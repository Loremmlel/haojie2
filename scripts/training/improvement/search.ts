import { ensure } from '../../../src/engine/core/state';
import type { Command, Player } from '../../../src/engine/types';
import type { Observation } from '../../../src/ai/types';
import { decisionOwner, fingerprint, hash } from '../../../src/ai/observation';
import {
  sampleTrainingTransition,
  simulationRandomSource,
} from '../../../src/ai/training/simulation';
import type {
  DecodeOptions,
  DecodeResult,
  PolicyEvaluator,
} from '../../../src/ai/training/decoder';
import { checkPosition, terminalValue, windowBoundary } from '../search/puct';
import { beamDecode, type CompleteCandidate } from './beam';

interface Edge {
  candidate: CompleteCandidate;
  prior: number;
  visits: number;
  total: number;
  children: Map<string, Node>;
}
interface Node {
  observation: Observation;
  actor: Player;
  visits: number;
  edges?: Edge[];
}
export interface SearchOptions extends DecodeOptions {
  simulations?: number;
  horizon?: number;
  mode?: 'mcts' | 'gumbel';
  sampleSeed?: number;
  leafValue?: (observation: Observation, rootActor: Player) => Promise<number>;
  valuePhases?: string[];
}

/** 固定预算的逐轮减半访问表；最后一轮继续比较剩余两项，不在中途重抽Gumbel。 */
export function halvingVisits(count: number, budget: number) {
  ensure(
    Number.isSafeInteger(count) && count > 0 && Number.isSafeInteger(budget) && budget > 0,
    '减半预算无效。',
  );
  if (count === 1) return Array.from({ length: budget }, (_, i) => i);
  const levels = Math.ceil(Math.log2(count));
  const visits = Array(count).fill(0) as number[];
  const sequence: number[] = [];
  let active = count;
  while (sequence.length < budget) {
    const rounds = Math.max(1, Math.floor(budget / levels / active));
    for (let n = 0; n < rounds; n++) for (let i = 0; i < active; i++) sequence.push(visits[i]++);
    active = Math.max(2, Math.floor(active / 2));
  }
  return sequence.slice(0, budget);
}

/**
 * 公开规则引擎上的有限候选MCTS：完整命令为决策边，独立概率抽样为机会子节点。
 * 根与内部节点先验均来自网络；默认只用真实终局，未知截断估计0，不使用未训练V头。
 * 显式叶值回调须由调用方核验校准证书，并声明覆盖阶段；未覆盖时整次退回束，不混入零值。
 * 回报存根视角，按实际操作者取符号；重复同方命令、敌方反应不机械逐步翻转。
 * Gumbel臂只改变根预算分配（减半、无放回初始排序），内部仍共用PUCT，不宣称完整Gumbel AlphaZero。
 * 不支持的规则窗口显式退回已计算的束命令；取消和推理故障显式暂停。统计保留全部已耗成本。
 */
export async function policySearch(
  o: Observation,
  actor: Player,
  evaluate: PolicyEvaluator,
  options: SearchOptions = {},
) {
  const mode = options.mode ?? 'mcts',
    simulations = options.simulations ?? 16,
    horizon = options.horizon ?? 2;
  ensure(
    [simulations, horizon].every((n) => Number.isSafeInteger(n) && n > 0),
    '搜索预算无效。',
  );
  ensure(actor === decisionOwner(o), '根操作者不匹配。');
  const base = await beamDecode(o, actor, evaluate, options);
  const stats = {
    simulations: 0,
    transitions: 0,
    expansions: 0,
    terminalLeaves: 0,
    unknownLeaves: 0,
    sameActor: 0,
    changedActor: 0,
    chanceOutcomes: 0,
    candidates: base.candidates.length,
    valueCalls: 0,
    valueMs: 0,
  };
  const result = (decoded: DecodeResult, fallback?: string, edges?: unknown) => ({
    ...decoded,
    search: {
      mode,
      horizon,
      stats,
      fallback,
      edges,
      leaf: options.leafValue ? 'certified-behavior-value' : 'terminal-or-zero-estimate',
      candidateDomain: 'neural-beam-subset',
      changedFromBeam:
        !!decoded.command && JSON.stringify(decoded.command) !== JSON.stringify(base.command),
    },
  });
  if (base.status !== 'command') return result(base);
  if (windowBoundary(o)) return result(base, 'phase-outside-search');
  try {
    checkPosition(o);
  } catch (error) {
    return result(base, String(error));
  }
  const seed = options.sampleSeed ?? hash(`improvement-search:${fingerprint(o)}:${actor}`);
  const random = simulationRandomSource(seed);
  const noise = simulationRandomSource(hash(`gumbel:${seed}`));
  const makeEdges = (candidates: CompleteCandidate[]) => {
    const high = Math.max(...candidates.map((c) => c.logProbability));
    const p = candidates.map((c) => Math.exp(c.logProbability - high));
    const total = p.reduce((a, b) => a + b, 0);
    return candidates.map(
      (candidate, i): Edge => ({
        candidate,
        prior: p[i] / total,
        visits: 0,
        total: 0,
        children: new Map(),
      }),
    );
  };
  const root: Node = {
    observation: o,
    actor,
    visits: 0,
    edges: makeEdges(base.candidates),
  };
  const gumbel = root.edges!.map(() => -Math.log(-Math.log(Math.max(Number.MIN_VALUE, noise()))));
  const schedule = halvingVisits(root.edges!.length, simulations);
  const q = (edge: Edge) => (edge.visits ? edge.total / edge.visits : 0);
  const rootScore = (edge: Edge, i: number) =>
    Math.log(edge.prior) +
    gumbel[i] +
    (50 + Math.max(...root.edges!.map((e) => e.visits))) * q(edge);
  let failure: DecodeResult | undefined;
  const visit = async (node: Node, depth: number): Promise<number> => {
    ensure(!options.signal?.aborted, 'cancelled');
    checkPosition(node.observation);
    const ended = terminalValue(node.observation, actor);
    if (ended !== null) {
      stats.terminalLeaves++;
      return ended;
    }
    if (depth >= horizon || windowBoundary(node.observation)) {
      stats.unknownLeaves++;
      if (!options.leafValue) return 0;
      ensure(
        options.valuePhases?.includes(node.observation.phase),
        `uncovered-value-phase:${node.observation.phase}`,
      );
      const started = performance.now();
      stats.valueCalls++;
      base.stats.evaluations++;
      let value;
      try {
        value = await options.leafValue(node.observation, actor);
      } finally {
        stats.valueMs += performance.now() - started;
      }
      ensure(!options.signal?.aborted, 'cancelled');
      ensure(Number.isFinite(value) && Math.abs(value) <= 1, '价值输出无效。');
      return value;
    }
    if (!node.edges) {
      const expanded = await beamDecode(node.observation, node.actor, evaluate, {
        ...options,
        candidates: 4,
        width: 8,
        maxEvaluations: Math.min(options.maxEvaluations ?? 32, 8),
      });
      for (const key of Object.keys(base.stats) as (keyof DecodeResult['stats'])[]) {
        if (key.startsWith('max')) base.stats[key] = Math.max(base.stats[key], expanded.stats[key]);
        else base.stats[key] += expanded.stats[key];
      }
      if (expanded.status !== 'command') {
        failure = expanded;
        throw new Error('decoder-paused');
      }
      node.edges = makeEdges(expanded.candidates);
      stats.expansions++;
    }
    const sign = node.actor === actor ? 1 : -1;
    const score = (e: Edge, i: number) =>
      depth === 0 && mode === 'gumbel'
        ? e.visits === schedule[stats.simulations]
          ? rootScore(e, i)
          : -Infinity
        : sign * q(e) + (1.5 * e.prior * Math.sqrt(node.visits + 1)) / (1 + e.visits);
    // 两臂同样覆盖根候选，之后分别用PUCT与逐轮减半分配剩余预算。
    const edge =
      (depth === 0 && mode === 'mcts' ? node.edges.find((e) => !e.visits) : undefined) ??
      node.edges.reduce((best, e, i) =>
        score(e, i) > score(best, node.edges!.indexOf(best)) ? e : best,
      );
    const next = sampleTrainingTransition(
      node.observation,
      node.actor,
      edge.candidate.command,
      Math.floor(random() * 4294967296),
    );
    stats.transitions++;
    const nextActor = decisionOwner(next);
    if (nextActor === node.actor) stats.sameActor++;
    else stats.changedActor++;
    const key = JSON.stringify(next);
    let child = edge.children.get(key);
    if (!child) {
      child = { observation: next, actor: nextActor, visits: 0 };
      edge.children.set(key, child);
      stats.chanceOutcomes++;
    }
    const value = await visit(child, depth + 1);
    edge.total += value;
    edge.visits++;
    node.visits++;
    return value;
  };
  try {
    for (let i = 0; i < simulations; i++) {
      await visit(root, 0);
      stats.simulations++;
    }
    ensure(!options.signal?.aborted, 'cancelled');
  } catch (error) {
    if (options.signal?.aborted)
      return result({
        ...base,
        status: 'paused',
        command: undefined,
        reason: 'cancelled',
      });
    if (failure) {
      if (
        ['node-budget', 'inference-budget', 'depth-limit', 'no-command'].includes(failure.reason!)
      )
        return result(base, `interior-${failure.reason}`);
      return result({ ...failure, stats: base.stats });
    }
    if (String(error).includes('回合外巨大化')) return result(base, 'descendant-u7');
    if (String(error).includes('uncovered-value-phase:')) return result(base, String(error));
    throw error;
  }
  const chosen =
    mode === 'gumbel'
      ? root.edges!.reduce((best, e, i) =>
          rootScore(e, i) > rootScore(best, root.edges!.indexOf(best)) ? e : best,
        )
      : root.edges!.reduce((best, e) => (e.visits > best.visits ? e : best));
  const edges = root.edges!.map((e) => ({
    command: e.candidate.command,
    prior: e.prior,
    visits: e.visits,
    value: e.visits ? q(e) : null,
    chanceOutcomes: e.children.size,
  }));
  return result(
    {
      ...base,
      command: chosen.candidate.command,
      path: chosen.candidate.path,
      leafStatus: chosen.candidate.leafStatus,
    },
    undefined,
    edges,
  );
}
