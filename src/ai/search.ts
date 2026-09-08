import type { GameState, Player } from '../engine/types';
import { iterateCandidateGroups, attackCandidates, type CandidateGroup } from './candidates';
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
  stopped: boolean;
  side: Player;
  difficulty: Difficulty;
  cache: Map<string, number>;
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
): Generator<void, Node[]> {
  const settings = DIFFICULTIES[level],
    result: Node[] = [],
    seen = new Set<string>();
  const work: (CandidateGroup & { index: number; accepted: number })[] = [];
  for (const group of iterateCandidateGroups(s, level)) {
    work.push({
      ...group,
      keep: lean ? 1 : path.length ? Math.min(3, group.keep) : group.keep,
      index: 0,
      accepted: 0,
    });
    yield;
  }
  let live = work.length,
    pass = 0;
  while (live > 0) {
    live = 0;
    for (const group of work) {
      if (group.index >= group.commands.length || group.accepted >= group.keep) continue;
      live++;
      if (
        stopped(ctx) &&
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
        command.type === 'summon' ? 64 : settings.chanceLimit,
        settings.chanceSamples,
      );
      ctx.count += d.attempts;
      if (d.sampled) ctx.sampled++;
      if (d.outcomes.length) {
        ctx.candidates++;
        group.accepted++;
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
            decisionOwner(only) === decisionOwner(s),
        });
      }
      yield;
    }
    if (pass++ === 0 && !lean && !stopped(ctx))
      result.push(...(yield* attackContinuations(ctx, result, level)));
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
      for (const command of attackCandidates(state, first.unitId).slice(
        0,
        level === 'hard' ? 6 : 3,
      )) {
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
          if (command.targetId === first.targetId) focused = node;
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
  steps = 60,
): Generator<void, GameState> {
  let state = initial;
  for (let i = 0; i < steps && !state.winner; i++) {
    if (stopped(ctx)) break;
    if (state.active !== owner && !state.pending.length) break;
    const mover = decisionOwner(state),
      nodes = yield* expand(ctx, state, [], 'easy', true);
    const choice = selectForOwner(nodes, ctx, mover);
    if (!choice) break;
    // Rollout sample is not selected by its score. Outcome choice is fixed independently.
    let draw = (hash(fingerprint(state) + i + owner) + 0.5) / 4294967296;
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
    deadline: time() + Math.max(10, limits.milliseconds ?? cfg.decisionMs),
    stopped: false,
    side,
    difficulty,
    cache: new Map(),
  };
  if (s.winner || decisionOwner(s) !== side)
    return {
      command: null,
      plan: [],
      stats: { simulations: 0, candidates: 0, depth: 0, replies: 0, sampled: 0, exhausted: false },
    };
  const fullDeadline = ctx.deadline;
  if (difficulty === 'hard') ctx.deadline = time() + Math.max(10, (fullDeadline - time()) * 0.52);
  const roots = yield* expand(ctx, s, []);
  const bestByRoot = new Map<string, Node>();
  for (const n of roots)
    if (!bestByRoot.has(n.root) || n.score > bestByRoot.get(n.root)!.score)
      bestByRoot.set(n.root, n);
  // Keep some budget for genuine adversarial replies instead of spending all of it on our moves.
  ctx.stopAt = difficulty === 'hard' ? Math.max(ctx.count, Math.floor(max * 0.55)) : max;
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
    ctx.depth = depth;
    beam = bestDiverse(children, cfg.width, seen);
  }
  let choices = [...bestByRoot.values()].sort((a, b) => b.score - a.score);
  ctx.stopAt = max;
  ctx.deadline = fullDeadline;
  if (difficulty === 'hard' && s.phase !== 'summon' && choices.length > 1 && !stopped(ctx)) {
    const shortlist = choices.slice(0, cfg.replyRoots),
      tested: Node[] = [];
    const remaining = ctx.deadline - time();
    for (let index = 0; index < shortlist.length; index++) {
      if (stopped(ctx)) break;
      const node = shortlist[index],
        globalDeadline = ctx.deadline,
        globalStop = ctx.stopAt;
      ctx.deadline = Math.min(globalDeadline, time() + remaining / shortlist.length);
      ctx.stopAt = Math.min(
        globalStop,
        ctx.count + Math.max(100, Math.floor((max - ctx.count) / (shortlist.length - index))),
      );
      let score = 0,
        weight = 0;
      for (const outcome of node.outcomes) {
        if (stopped(ctx)) break;
        let state = outcome.state;
        if (!state.winner && state.active === side) state = yield* rollout(ctx, state, side);
        if (!state.winner && state.active !== side) {
          state = yield* rollout(ctx, state, state.active);
          if (
            state.winner ||
            (state.active === side && state.ply >= s.ply + 2 && !state.pending.length)
          )
            ctx.replies++;
        }
        // Never rank a half-finished own turn against a fully completed opponent reply.
        if (
          state.winner ||
          (state.active === side && state.ply >= s.ply + 2 && !state.pending.length)
        ) {
          score += outcome.weight * value(ctx, state);
          weight += outcome.weight;
        }
      }
      ctx.deadline = globalDeadline;
      ctx.stopAt = globalStop;
      if (weight > 0.999999) tested.push({ ...node, score: score / weight, replyTested: true });
    }
    // Compare candidates at the same reply-tested stage. Unsearched roots are not assumed safe.
    if (tested.length >= 2) choices = tested.sort((a, b) => b.score - a.score);
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
