import { ensure } from '../../../../src/engine/core/state';
import type { Command, Player } from '../../../../src/engine/types';
import type { Observation } from '../../../../src/ai/types';
import { decisionOwner, hash } from '../../../../src/ai/observation';
import { decide } from '../../../../src/ai/planning/search';
import { canonicalTrainingCommand } from '../../../../src/ai/training/action-tree';
import {
  sampleTrainingTransition,
  simulationRandomSource,
} from '../../../../src/ai/training/simulation';
import { checkPosition, search, searchAsync, terminalValue, windowBoundary } from '../puct';

/**
 * 自对弈启动用的教师辅助搜索：最多8个教师候选、16次PUCT，根每候选至少访问一次。
 * 同一公开局面的候选与续演共用教师首选和缓存，避免续演策略与展开排序相互矛盾。
 * 非支持窗口/巨大化明确回退教师，未知回报不变成价值标签；未预期失败须中断而非掩盖。
 * 无正式随机状态、无跨决策计划或可变输入。所有教师内部work和模拟转移单列。
 */
function context(observation: Observation, sampleSeed: number) {
  const stats = {
    teacherCalls: 0,
    teacherWork: 0,
    rolloutTransitions: 0,
    rolloutTerminal: 0,
    rolloutUnknown: 0,
    searchTransitions: 0,
    searchSimulations: 0,
  };
  const cache = new Map<string, { command: Command; candidates: Command[] }>();
  const teacher = (o: Observation) => {
    const key = JSON.stringify(o);
    const cached = cache.get(key);
    if (cached) return cached;
    ensure(stats.teacherCalls < 17, '教师调用超出固定上限。');
    const actor = decisionOwner(o);
    const decision = decide(o, actor, 'easy', { simulations: 40, mode: 'work', trace: true });
    stats.teacherCalls++;
    stats.teacherWork += decision.stats.simulations;
    // work请求不是原子概率分支的硬截断；完整召唤可超过40，实际成本照录并设异常保护。
    ensure(
      decision.command &&
        Number.isSafeInteger(decision.stats.simulations) &&
        decision.stats.simulations >= 0 &&
        decision.stats.simulations <= 256,
      '教师返回或预算异常。',
    );
    const candidates = [
      ...new Map(
        [decision.command, ...(decision.trace?.alternatives.map((a) => a.command) ?? [])]
          .map((c) => canonicalTrainingCommand(o, actor, c))
          .map((c) => [JSON.stringify(c), c]),
      ).values(),
    ].slice(0, 8);
    const value = { command: candidates[0], candidates };
    cache.set(key, value);
    return value;
  };
  const base = teacher(observation);
  let fallback: string | undefined;
  if (windowBoundary(observation)) fallback = 'phase-outside-search';
  try {
    checkPosition(observation);
  } catch (error) {
    if (String(error).includes('回合外巨大化')) fallback = 'unsupported-u7';
    else throw error;
  }
  const random = simulationRandomSource(hash(`bootstrap-rollout:${sampleSeed}`));
  return { stats, teacher, base, fallback, random };
}

type Context = ReturnType<typeof context>;

function fallbackResult(ctx: Context, reason: string) {
  return {
    command: ctx.base.command,
    mode: 'teacher-fallback' as const,
    reason,
    stats: ctx.stats,
    policy: null,
  };
}

function options(ctx: Context, sampleSeed: number) {
  return {
    simulations: 16,
    horizon: 2,
    sampleSeed,
    deferExpansion: true,
    coverRoot: true,
    candidateCommands: (o: Observation) => ctx.teacher(o).candidates,
  };
}

// 同步零叶值与网络叶值共用这一次真实模拟；未知与终局必须保持不同类型。
function rollout(ctx: Context, o: Observation, root: Player, remaining: number) {
  if (!remaining || windowBoundary(o)) {
    ctx.stats.rolloutUnknown++;
    return { observation: o, value: null };
  }
  const next = sampleTrainingTransition(
    o,
    decisionOwner(o),
    ctx.teacher(o).command,
    Math.floor(ctx.random() * 4294967296),
  );
  ctx.stats.rolloutTransitions++;
  checkPosition(next);
  const value = terminalValue(next, root);
  if (value === null) ctx.stats.rolloutUnknown++;
  else ctx.stats.rolloutTerminal++;
  return { observation: next, value };
}

function finish(ctx: Context, result: ReturnType<typeof search>) {
  const { stats, base } = ctx;
  stats.searchTransitions = result.stats.transitions;
  stats.searchSimulations = result.stats.simulations;
  ensure(stats.searchTransitions + stats.rolloutTransitions <= 32, '搜索转移超出上限。');
  if (result.status === 'paused') {
    if (!result.reason.includes('回合外巨大化')) throw new Error(result.reason);
    return fallbackResult(ctx, 'descendant-u7');
  }
  ensure(
    result.edges.every((e) => e.visits > 0),
    '根候选未全部访问。',
  );
  return {
    command: result.command,
    mode: 'search' as const,
    stats,
    policy: result.edges.map((e) => ({
      command: e.command,
      visits: e.visits,
      probability: e.visits / 16,
    })),
    rootValues: result.edges.map((e) => e.value),
    teacherCommand: base.command,
  };
}

export function bootstrapDecision(observation: Observation, sampleSeed: number) {
  const ctx = context(observation, sampleSeed);
  if (ctx.fallback) return fallbackResult(ctx, ctx.fallback);
  return finish(
    ctx,
    search(observation, {
      ...options(ctx, sampleSeed),
      leafValue: (o, root, remaining) => rollout(ctx, o, root, remaining).value ?? 0,
    }),
  );
}

/**
 * 将外部根视角叶值接入相同搜索；保留真实终局优先及所有原有教师/转移预算。
 * 新召唤窗口没有本轮价值训练覆盖，仍返回未知零估计；模型只估计play及强制反应。
 * 每决策至多16次值查询，公开局面缓存仅活到该决策结束；取消后不得落子。
 */
export async function bootstrapValueDecision(
  observation: Observation,
  sampleSeed: number,
  value: (observation: Observation, root: Player) => Promise<number>,
  signal?: AbortSignal,
) {
  const ctx = context(observation, sampleSeed);
  const valueStats = { calls: 0, cacheHits: 0, min: 0, max: 0 };
  if (ctx.fallback) return { ...fallbackResult(ctx, ctx.fallback), valueStats };
  const cache = new Map<string, number>();
  const result = await searchAsync(observation, {
    ...options(ctx, sampleSeed),
    signal,
    leafValue: async (o, root, remaining) => {
      const leaf = rollout(ctx, o, root, remaining);
      if (leaf.value !== null) return leaf.value;
      if (windowBoundary(leaf.observation)) return 0;
      const key = JSON.stringify(leaf.observation);
      const saved = cache.get(key);
      if (saved !== undefined) {
        valueStats.cacheHits++;
        return saved;
      }
      ensure(valueStats.calls < 16, '网络叶值查询超出固定上限。');
      const estimate = await value(leaf.observation, root);
      ensure(Number.isFinite(estimate) && Math.abs(estimate) <= 1, '网络叶值不是有限有界数。');
      valueStats.calls++;
      valueStats.min = Math.min(valueStats.min, estimate);
      valueStats.max = Math.max(valueStats.max, estimate);
      cache.set(key, estimate);
      return estimate;
    },
  });
  return { ...finish(ctx, result), valueStats };
}
