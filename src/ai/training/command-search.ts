import type { Player } from '../../engine/types';
import { allPieces, hasTrait } from '../../engine/core/traits';
import { decisionOwner } from '../observation';
import type { Observation } from '../types';
import { TrainingActionTree, type ActionNode } from './action-tree';
import {
  conditionSearchVisits,
  publicSearchSeed,
  searchCommands,
  searchRandom,
  type SearchPosition,
} from './mcts';
import { encodePosition } from './encoding/state';
import { bf16 } from './encoding/schema';
import { sampleSearchTransition } from './simulation';
import { trainingPosition } from './queries';

export interface SearchQuery {
  observation: Observation;
  actor: Player;
  node: ActionNode;
  optional: boolean;
  excluded: Map<string, number[]>;
}
export type SearchModel = (query: SearchQuery) => Promise<{ mass: number[]; value: number }>;
export interface CommandSearchPlan {
  observation: Observation;
  model: SearchModel;
  actor: Player;
  actions: { action: number[]; visits: number; prior: number }[];
}
const round = (value: number) => new Float32Array(new Uint32Array([bf16(value) << 16]).buffer)[0];
const orderedFrontier = (excluded: Map<string, number[]>) =>
  new Map(
    [...excluded].sort(([left], [right]) => {
      const a = left ? left.split(',').map(Number) : [];
      const b = right ? right.split(',').map(Number) : [];
      for (let i = 0; i < Math.min(a.length, b.length); ++i) if (a[i] !== b[i]) return a[i] - b[i];
      return a.length - b.length;
    }),
  );
interface State {
  observation: Observation;
  actor: Player;
  optional: boolean;
  offerInterrupt: boolean;
  cursor: number[];
  tree: TrainingActionTree;
  prediction: { mass: number[]; value: number };
}

/** TS 公开规则参照入口；与原生的完整命令采样、机会后继和访问分布使用同一流程。 */
export async function searchTrainingCommand(
  observation: Observation,
  model: SearchModel,
  options: {
    simulations?: number;
    depth?: number;
    seed?: number;
    offerInterrupt?: boolean;
    cursor?: number[];
    excluded?: Map<string, number[]>;
    cancelled?: () => boolean;
    plan?: CommandSearchPlan;
  } = {},
) {
  const simulations = options.simulations ?? 0;
  if (!Number.isSafeInteger(simulations) || simulations < 0)
    throw new Error('搜索预算须为非负整数，0 关闭搜索。');
  const depth = options.depth ?? 64;
  const cancelled = options.cancelled ?? (() => false);
  let perspective: Player;
  const positions = new Map<string, SearchPosition<State>>();
  const query = async (state: State, cursor: number[], excluded: Map<string, number[]>) => {
    const raw = state.tree.node(cursor);
    const pass = state.optional && !cursor.length;
    const count = raw.choices.length + Number(pass);
    const mask = Array.from(
      { length: count },
      (_, i) =>
        (i === raw.choices.length || raw.mask?.[i] !== false) &&
        !excluded.get(cursor.join(','))?.includes(i),
    );
    return {
      observation: state.observation,
      actor: state.actor,
      node: { ...raw, mask },
      optional: pass,
      excluded: orderedFrontier(excluded),
    };
  };
  const make = async (
    o: Observation,
    offerInterrupt: boolean,
    cursor: number[] = [],
    excluded = new Map<string, number[]>(),
    initial = false,
  ): Promise<SearchPosition<State>> => {
    const key = JSON.stringify([o, offerInterrupt, cursor, [...orderedFrontier(excluded)]]);
    const previous = positions.get(key);
    if (previous) return previous;
    let actor = decisionOwner(o);
    if (o.phase === 'shrine-draft' && o.shrineDraft?.committed[actor])
      actor = (3 - actor) as Player;
    const other = (3 - actor) as Player;
    const optional =
      offerInterrupt &&
      !o.pending.length &&
      o.phase !== 'shrine-draft' &&
      allPieces(trainingPosition(o)).some((u) => u.owner === other && hasTrait(u, 'u7'));
    if (optional) actor = other;
    if (initial) perspective = actor;
    const state: State = {
      observation: o,
      actor,
      optional,
      offerInterrupt,
      cursor,
      tree: new TrainingActionTree(o, actor),
      prediction: { mass: [], value: 0 },
    };
    const terminal = !!o.winner;
    if (!terminal) state.prediction = await model(await query(state, cursor, excluded));
    const position = {
      state,
      key,
      actor,
      cutoff: !terminal && !state.prediction.mass.some((mass) => mass > 0),
      value: terminal
        ? o.winner === 'draw'
          ? 0
          : o.winner === perspective
            ? 1
            : -1
        : state.prediction.value * (actor === perspective ? 1 : -1),
      terminal,
    };
    positions.set(key, position);
    return position;
  };
  const root = await make(
    observation,
    options.offerInterrupt ?? true,
    options.cursor ?? [],
    options.excluded ?? new Map(),
    true,
  );
  if (root.cutoff) throw new Error('no legal action at root');
  if (!simulations || observation.phase === 'shrine-draft')
    return {
      mass: root.state.prediction.mass,
      target: [],
      simulations: 0,
      expanded: 0,
      deepest: 0,
      opponentNodes: 0,
    };
  const seed =
    options.seed ??
    publicSearchSeed(encodePosition(observation, root.actor).globals, root.state.cursor);
  const plan = options.plan;
  if (plan?.observation === observation && plan.model === model && plan.actor === root.actor) {
    const q = await query(root.state, root.state.cursor, options.excluded ?? new Map());
    const visits = conditionSearchVisits(
      plan.actions,
      root.state.cursor,
      root.state.prediction.mass,
      q.node.mask!,
    );
    if (visits)
      return {
        target: visits.target.map(round),
        mass: visits.mass.map(round),
        simulations: 0,
        expanded: 0,
        deepest: 0,
        opponentNodes: 0,
        plan,
      };
  }
  let result;
  try {
    result = await searchCommands(
      root,
      {
        async sample(p, sampleSeed) {
          const s = p.state;
          const excluded = new Map<string, number[]>();
          if (p === root)
            for (const [key, values] of options.excluded ?? []) excluded.set(key, [...values]);
          const cursor = [...s.cursor];
          const random = searchRandom(sampleSeed ^ 0xa0761d65);
          let prior = 1;
          let visited = 0;
          let first = true;
          for (;;) {
            if (cancelled()) throw new Error('搜索已取消。');
            const q = await query(s, cursor, excluded);
            const initial = first ? s.prediction : undefined;
            first = false;
            const choices = q.node.choices;
            const remaining = q.node.mask!.flatMap((enabled, i) => (enabled ? [i] : []));
            if (!remaining.length) {
              excluded.delete(cursor.join(','));
              const index = cursor.pop();
              if (index === undefined) throw new Error('no legal action at root');
              const key = cursor.join(',');
              excluded.set(key, [...(excluded.get(key) ?? []), index]);
              continue;
            }
            let selected = remaining[0];
            if (remaining.length > 1) {
              if (visited >= 4096) throw new Error('search command parameter budget exceeded');
              ++visited;
              const prediction = initial ?? (await model(q));
              if (
                prediction.mass.length !== q.node.mask!.length ||
                prediction.mass.some(
                  (value, i) =>
                    !Number.isFinite(value) || value < 0 || (value !== 0 && !q.node.mask![i]),
                )
              )
                throw new Error('invalid search probability support');
              const mass = prediction.mass.map((value, i) => (q.node.mask![i] ? value : 0));
              const total = mass.reduce((a, b) => a + b, 0);
              const draw = ((random() + 0.5) / 4294967296) * total;
              let sum = 0;
              selected = mass.findIndex((value) => (sum += value) > draw);
              if (selected < 0) throw new Error('invalid search probability support');
              prior *= mass[selected] / total;
            }
            const path = [...cursor, selected];
            if (q.optional && selected === choices.length)
              return { action: path, key: path.join('/') + '/', prior: Math.max(1e-12, prior) };
            const choice = choices[selected];
            if (choice.next) {
              cursor.push(selected);
              continue;
            }
            try {
              sampleSearchTransition(s.observation, s.actor, choice.command, sampleSeed);
              return { action: path, key: path.join('/') + '/', prior: Math.max(1e-12, prior) };
            } catch (error) {
              if (choice.status !== 'uncertain') throw error;
              const key = cursor.join(',');
              excluded.set(key, [...(excluded.get(key) ?? []), selected]);
            }
          }
        },
        async step(p, path, sampleSeed) {
          const s = p.state;
          const node = s.tree.node(path.slice(0, -1));
          const index = path.at(-1)!;
          const pass = s.optional && path.length === 1 && index === node.choices.length;
          const next = pass
            ? s.observation
            : sampleSearchTransition(
                s.observation,
                s.actor,
                node.choices[index].command,
                sampleSeed,
              );
          return make(next, !s.optional);
        },
      },
      simulations,
      depth,
      seed,
      cancelled,
    );
  } catch (error) {
    if (error instanceof Error && error.message === 'no legal action at root')
      return {
        mass: root.state.prediction.mass,
        target: [],
        simulations: 0,
        expanded: 0,
        deepest: 0,
        opponentNodes: 0,
        emptyBranch: true,
      };
    if (!(error instanceof Error) || error.message !== 'search command parameter budget exceeded')
      throw error;
    return {
      mass: root.state.prediction.mass,
      target: [],
      simulations: 0,
      expanded: 0,
      deepest: 0,
      opponentNodes: 0,
      parameterLimit: true,
    };
  }
  const q = await query(root.state, root.state.cursor, options.excluded ?? new Map());
  const visits = conditionSearchVisits(
    result.actions,
    root.state.cursor,
    root.state.prediction.mass,
    q.node.mask!,
  );
  if (!visits) throw new Error('search did not visit a supported command');
  return {
    ...result,
    target: visits.target.map(round),
    mass: visits.mass.map(round),
    plan: { observation, model, actor: root.actor, actions: result.actions },
  };
}
