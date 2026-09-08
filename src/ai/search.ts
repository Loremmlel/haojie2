import type { GameState, Player } from '../engine/types';
import {
  iterateCandidateGroups,
  attackCandidates,
  commandPriority,
  type CandidateGroup,
} from './candidates';
import { DIFFICULTIES } from './difficulty';
import { analyzePayload } from './threats';
import { evaluate, explainEvaluation, materialValue } from './evaluate';
import { decisionOwner, fingerprint, hash, imagined, positionKey } from './observation';
import { distribution } from './simulate';
import type { Outcome } from './simulate';
import type { Decision, Difficulty, Observation, PlanStep, SearchLimits } from './types';
interface Node {
  outcomes: Outcome[];
  score: number;
  path: PlanStep[];
  root: string;
  expandable: boolean;
  replyTested?: boolean;
}
interface Context {
  count: number;
  candidates: number;
  sampled: number;
  replies: number;
  depth: number;
  max: number;
  stopAt: number;
  deadline: number;
  side: Player;
  difficulty: Difficulty;
  cache: Map<string, number>;
  scenarioSeed: number;
}
// Common random scenarios reduce draw noise BETWEEN alternatives; never use the real PRNG.
function scenarioKey(
  ctx: Context,
  s: GameState,
  command: import('../engine/types').Command,
  sample: number,
): string {
  const u = s.units.find((v) => v.id === command.unitId);
  return JSON.stringify([
    ctx.scenarioSeed,
    sample,
    s.ply,
    s.summonSlots,
    command.type,
    command.unitId,
    command.cardId,
    u?.shots,
    u?.moves,
    u?.operations,
    u?.freeUsed,
  ]);
}
const time = () => (typeof performance === 'undefined' ? Date.now() : performance.now());
function stopped(ctx: Context): boolean {
  return ctx.count >= ctx.stopAt || time() >= ctx.deadline;
}
function value(ctx: Context, s: GameState): number {
  const key = positionKey(s);
  let v = ctx.cache.get(key);
  if (v === undefined) {
    v = evaluate(s, ctx.side);
    if (ctx.cache.size > 3000) ctx.cache.clear();
    ctx.cache.set(key, v);
  }
  return v;
}
function expectation(ctx: Context, out: Outcome[]): number {
  const mean = out.reduce((v, o) => v + o.weight * value(ctx, o.state), 0);
  // A small downside penalty, not a replacement for the expected value of random outcomes.
  const risk = ctx.difficulty === 'hard' ? 0.06 : 0;
  return (
    mean - risk * out.reduce((v, o) => v + o.weight * Math.max(0, mean - value(ctx, o.state)), 0)
  );
}
/** Each yield is a cancellation/coop-scheduling point. Worker and main-thread fallback use identical code. */
function* expand(
  ctx: Context,
  s: GameState,
  path: PlanStep[],
  level: Difficulty = ctx.difficulty,
  lean = false,
  salt = 0,
): Generator<void, Node[]> {
  const settings = DIFFICULTIES[level],
    result: Node[] = [],
    seen = new Set<string>();
  const work: (CandidateGroup & { index: number; accepted: number })[] = [];
  for (const group of iterateCandidateGroups(s, level)) {
    work.push({
      ...group,
      keep: lean || path.length ? 1 : group.keep,
      index: 0,
      accepted: 0,
    });
    yield;
  }
  // A rollout is a narrow policy, not another exhaustive root search at each atomic command.
  if (lean) {
    work.sort(
      (a, b) =>
        (b.priority ?? commandPriority(s, b.commands[0])) -
        (a.priority ?? commandPriority(s, a.commands[0])),
    );
    work.splice(3);
  }
  const localStart = ctx.count;
  const localLimit = lean ? Infinity : path.length ? 24 : Math.max(40, Math.floor(ctx.max * 0.16));
  let live = work.length;
  while (live > 0) {
    const passStart = result.length;
    live = 0;
    for (const group of work) {
      if (group.index >= group.commands.length || group.accepted >= group.keep) continue;
      live++;
      if (
        (stopped(ctx) || ctx.count - localStart >= localLimit) &&
        result.length &&
        !(s.phase === 'summon' && group.family === 'summon' && group.accepted < group.keep)
      )
        return result;
      const command = group.commands[group.index++],
        key = JSON.stringify(command);
      if (seen.has(key)) continue;
      seen.add(key);
      const d = distribution(
        s,
        command,
        lean ? 2 : command.type === 'summon' ? 64 : settings.chanceLimit,
        lean ? 1 : settings.chanceSamples,
        salt,
        lean ? scenarioKey(ctx, s, command, salt) : undefined,
      );
      ctx.count += d.attempts;
      if (d.sampled) ctx.sampled++;
      if (d.outcomes.length) {
        ctx.candidates++;
        group.accepted++;
        if (!lean) ctx.depth = Math.max(ctx.depth, path.length + 1);
        const nextPath = [...path, { before: fingerprint(s), command }];
        const only = d.outcomes.length === 1 ? d.outcomes[0].state : null;
        result.push({
          outcomes: d.outcomes,
          score:
            expectation(ctx, d.outcomes) -
            nextPath.filter((p) => p.command.type === 'move').length * 0.2,
          path: nextPath,
          root: path.length ? JSON.stringify(path[0].command) : key,
          expandable:
            !!only &&
            !only.winner &&
            only.active === s.active &&
            only.phase === s.phase &&
            decisionOwner(only) === decisionOwner(s),
        });
      }
      yield;
    }
    if (!lean && !path.length && !stopped(ctx))
      result.push(...(yield* attackContinuations(ctx, result.slice(passStart), level)));
  }
  return result;
}
/** Also evaluate a short complete firing operation. Six small shots must not lose to
 * a cosmetic move just because the beam cannot reach shot six. Keep the one-shot node too,
 * so interleaving marks/heals is still possible. Never continue through unknown dice or reactions. */
function* attackContinuations(
  ctx: Context,
  roots: Node[],
  level: Difficulty,
): Generator<void, Node[]> {
  const added: Node[] = [];
  for (const root of roots) {
    const first = root.path.at(-1)!.command;
    if (first.type !== 'attack' || !first.unitId || !root.expandable) continue;
    let current = root;
    for (let step = 0; step < 5 && !stopped(ctx); step++) {
      if (current.outcomes.length !== 1) break;
      const state = current.outcomes[0].state;
      if (state.pending.length || state.winner) break;
      let best: Node | undefined, focused: Node | undefined;
      const followups = attackCandidates(state, first.unitId);
      // A mark or a depleted shooter can hand off the same target to an ally. Keep
      // this as a legal, deterministic sequence; dice/reactions still break the line.
      if (state.units.some((u) => u.id === first.targetId) || first.targetId?.startsWith('base-'))
        for (const ally of state.units)
          if (ally.owner === state.active && ally.id !== first.unitId)
            followups.push(
              ...attackCandidates(state, ally.id).filter((c) => c.targetId === first.targetId),
            );
      followups.sort((a, b) => commandPriority(state, b) - commandPriority(state, a));
      for (const command of followups.slice(0, level === 'hard' ? 6 : 3)) {
        if (stopped(ctx)) break;
        const d = distribution(
          state,
          command,
          DIFFICULTIES[level].chanceLimit,
          DIFFICULTIES[level].chanceSamples,
        );
        ctx.count += d.attempts;
        if (d.sampled) ctx.sampled++;
        if (d.outcomes.length) {
          ctx.candidates++;
          const single = d.outcomes.length === 1 ? d.outcomes[0].state : null;
          const node: Node = {
            outcomes: d.outcomes,
            score: expectation(ctx, d.outcomes),
            root: root.root,
            path: [...current.path, { before: fingerprint(state), command }],
            expandable:
              !!single &&
              !single.winner &&
              single.active === state.active &&
              decisionOwner(single) === decisionOwner(state),
          };
          ctx.depth = Math.max(ctx.depth, node.path.length);
          if (command.targetId === first.targetId && (!focused || node.score > focused.score))
            focused = node;
          if (!best || node.score > best.score) best = node;
        }
        yield;
      }
      best = focused ?? best;
      if (!best) break;
      current = best;
      if (best.score > root.score + 0.01) added.push(best);
      if (!best.expandable) break;
    }
  }
  return added;
}
function bestDiverse(nodes: Node[], width: number, seen: Set<string>): Node[] {
  const chosen: Node[] = [],
    roots = new Map<string, number>();
  for (const node of [...nodes].sort((a, b) => b.score - a.score)) {
    if (!node.expandable) continue;
    const state = node.outcomes[0].state,
      key = positionKey(state);
    if (seen.has(key) || (roots.get(node.root) ?? 0) >= Math.max(2, Math.ceil(width / 3))) continue;
    roots.set(node.root, (roots.get(node.root) ?? 0) + 1);
    seen.add(key);
    chosen.push(node);
    if (chosen.length === width) break;
  }
  return chosen;
}
function selectForOwner(nodes: Node[], ctx: Context, owner: Player): Node | undefined {
  return nodes.sort((a, b) => (owner === ctx.side ? b.score - a.score : a.score - b.score))[0];
}
/** A legal, budget-limited turn rollout; opponent reactions are adversarial, never cooperative.
 * At chance nodes its policy adapts only after observing that sampled outcome. */
function* rollout(
  ctx: Context,
  initial: GameState,
  owner: Player,
  scenario = 0,
  steps = 100,
): Generator<void, GameState> {
  let state = initial;
  for (let i = 0; i < steps && !state.winner; i++) {
    if (stopped(ctx)) break;
    if (state.active !== owner && !state.pending.length) break;
    const mover = decisionOwner(state),
      nodes = yield* expand(ctx, state, [], 'easy', true, scenario);
    const choice = selectForOwner(nodes, ctx, mover);
    if (!choice) break;
    // Rollout sample is not selected by its score. Outcome choice is fixed independently.
    let draw = (hash(scenarioKey(ctx, state, choice.path[0].command, scenario)) + 0.5) / 4294967296;
    let selected = choice.outcomes.at(-1)!;
    for (const out of choice.outcomes) {
      draw -= out.weight;
      if (draw <= 0) {
        selected = out;
        break;
      }
    }
    state = selected.state;
  }
  return state;
}
export function* search(
  observation: Observation,
  side: Player,
  difficulty: Difficulty,
  limits: Partial<SearchLimits> = {},
): Generator<void, Decision> {
  const cfg = DIFFICULTIES[difficulty],
    s = imagined(observation),
    max = Math.max(40, limits.simulations ?? cfg.nodes);
  const ctx: Context = {
    count: 0,
    candidates: 0,
    sampled: 0,
    replies: 0,
    depth: 1,
    max,
    stopAt: max,
    deadline:
      limits.mode === 'timed'
        ? time() + Math.max(10, limits.milliseconds ?? cfg.decisionMs)
        : Infinity,
    side,
    difficulty,
    cache: new Map(),
    scenarioSeed: hash(positionKey(s)),
  };
  if (s.winner || decisionOwner(s) !== side)
    return {
      command: null,
      plan: [],
      stats: { simulations: 0, candidates: 0, depth: 0, replies: 0, sampled: 0, exhausted: false },
    };
  const fullDeadline = ctx.deadline;
  if (difficulty === 'hard') {
    if (Number.isFinite(fullDeadline))
      ctx.deadline = time() + Math.max(10, (fullDeadline - time()) * 0.4);
    ctx.stopAt = Math.max(40, Math.floor(max * 0.4));
  }
  const roots = yield* expand(ctx, s, []);
  const bestByRoot = new Map<string, Node>();
  for (const n of roots)
    if (!bestByRoot.has(n.root) || n.score > bestByRoot.get(n.root)!.score)
      bestByRoot.set(n.root, n);
  // Keep some budget for genuine adversarial replies instead of spending all of it on our moves.
  ctx.stopAt = difficulty === 'hard' ? Math.max(ctx.count, Math.floor(max * 0.4)) : max;
  const seen = new Set<string>();
  let beam = bestDiverse(roots, cfg.width, seen);
  for (let depth = 2; depth <= cfg.depth && beam.length && !stopped(ctx); depth++) {
    const children: Node[] = [];
    for (const parent of beam) {
      if (stopped(ctx)) break;
      const next = yield* expand(ctx, parent.outcomes[0].state, parent.path);
      for (const node of next) {
        const old = bestByRoot.get(node.root);
        if (!old || node.score > old.score + 0.01) bestByRoot.set(node.root, node);
      }
      children.push(...next);
    }

    beam = bestDiverse(children, cfg.width, seen);
  }
  let choices = [...bestByRoot.values()].sort((a, b) => b.score - a.score);
  ctx.stopAt = max;
  ctx.deadline = fullDeadline;
  let replyCandidates = 0,
    replySamples = 0;
  if (difficulty === 'hard' && s.phase !== 'summon' && choices.length > 1 && !stopped(ctx)) {
    const shortlist = choices.slice(0, cfg.replyRoots);
    const sums = shortlist.map(() => 0);
    // Publish only COMPLETE PAIRED scenarios. A completed first round remains useful even
    // if a later scenario runs out of budget. No half-turn scores or unequal sample counts.
    for (let sample = 0; sample < 2 && !stopped(ctx); sample++) {
      const roundScores: number[] = [];
      const roundDeadline = ctx.deadline;
      for (let index = 0; index < shortlist.length; index++) {
        if (stopped(ctx)) break;
        const node = shortlist[index];
        let draw = (hash(fingerprint(s) + ':reply:' + sample) + 0.5) / 4294967296;
        let state = node.outcomes.at(-1)!.state;
        for (const out of node.outcomes) {
          draw -= out.weight;
          if (draw <= 0) {
            state = out.state;
            break;
          }
        }
        if (node.outcomes.length > 1) ctx.sampled++;
        // Equal time slices in opt-in timed mode; normal production has a fixed work budget.
        ctx.deadline = Math.min(
          roundDeadline,
          time() + (roundDeadline - time()) / (shortlist.length - index),
        );
        if (!state.winner && state.active === side)
          state = yield* rollout(ctx, state, side, sample);
        if (!state.winner && state.active !== side)
          state = yield* rollout(ctx, state, state.active, sample);
        ctx.deadline = roundDeadline;
        if (
          state.winner ||
          (state.active === side && state.ply >= s.ply + 2 && !state.pending.length)
        ) {
          ctx.replies++;
          roundScores.push(value(ctx, state));
        } else break;
      }
      if (roundScores.length !== shortlist.length) break;
      replySamples++;
      roundScores.forEach((score, index) => (sums[index] += score));
      choices = shortlist
        .map((node, index) => ({ ...node, score: sums[index] / replySamples, replyTested: true }))
        .sort((a, b) => b.score - a.score);
      replyCandidates = shortlist.length;
    }
  }
  let selected = choices[0];
  if (
    difficulty === 'easy' &&
    s.phase !== 'summon' &&
    selected &&
    Math.abs(selected.score) < 90000
  ) {
    const near = choices.filter((n) => selected!.score - n.score <= cfg.noise).slice(0, 3);
    selected = near[hash(fingerprint(s)) % near.length];
  }
  const plan = selected?.path ?? [];
  return {
    command: plan[0]?.command ?? null,
    plan,
    ...(limits.trace
      ? {
          trace: {
            initial: { ...explainEvaluation(s, side) },
            chosen: plan[0]?.command ?? null,
            alternatives: choices.slice(0, 12).map((node) => ({
              command: node.path[0].command,
              score: node.score,
              stage: node.replyTested ? ('reply' as const) : ('static' as const),
              line: node.path.map((step) => step.command),
              outcomes: node.outcomes.map((out) => ({
                probability: out.weight,
                score: value(ctx, out.state),
                bases: out.state.bases,
                terms: { ...explainEvaluation(out.state, side) },
                payloads: out.state.units.flatMap((u) =>
                  (['execute', 'convert'] as const)
                    .filter((type) => u.effects.some((e) => e.type === type))
                    .map((type) =>
                      analyzePayload(out.state, u, type, (v) => materialValue(out.state, v)),
                    ),
                ),
              })),
            })),
          },
        }
      : {}),
    stats: {
      simulations: ctx.count,
      candidates: ctx.candidates,
      depth: ctx.depth,
      replies: ctx.replies,
      sampled: ctx.sampled,
      exhausted: stopped(ctx),
      mode: limits.mode ?? 'work',
      stopReason: ctx.count >= max ? 'nodes' : time() >= ctx.deadline ? 'time' : 'complete',
      replyCandidates,
      replySamples,
      selectedDepth: plan.length,
    },
  };
}
/** Synchronous convenience for Node tests and controlled benchmarks, not the UI. */
export function decide(
  observation: Observation,
  side: Player,
  difficulty: Difficulty,
  limits: Partial<SearchLimits> = {},
): Decision {
  const iterator = search(observation, side, difficulty, limits);
  let step = iterator.next();
  while (!step.done) step = iterator.next();
  return step.value;
}
