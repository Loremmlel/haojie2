import { ensure } from '../../../../src/engine/core/state';
import type { Command } from '../../../../src/engine/types';
import type { Observation } from '../../../../src/ai/types';
import { decisionOwner, hash } from '../../../../src/ai/observation';
import { decide } from '../../../../src/ai/planning/search';
import { canonicalTrainingCommand } from '../../../../src/ai/training/action-tree';
import {
  sampleTrainingTransition,
  simulationRandomSource,
} from '../../../../src/ai/training/simulation';
import { checkPosition, search, terminalValue, windowBoundary } from '../puct';

/**
 * 自对弈启动用的教师辅助搜索：最多8个教师候选、16次PUCT，根每候选至少访问一次。
 * 同一公开局面的候选与续演共用教师首选和缓存，避免续演策略与展开排序相互矛盾。
 * 非支持窗口/巨大化明确回退教师，未知回报不变成价值标签；未预期失败须中断而非掩盖。
 * 无正式随机状态、无跨决策计划或可变输入。所有教师内部work和模拟转移单列。
 */
export function bootstrapDecision(observation: Observation, sampleSeed: number) {
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
  if (fallback)
    return {
      command: base.command,
      mode: 'teacher-fallback' as const,
      reason: fallback,
      stats,
      policy: null,
    };
  const random = simulationRandomSource(hash(`bootstrap-rollout:${sampleSeed}`));
  const result = search(observation, {
    simulations: 16,
    horizon: 2,
    sampleSeed,
    deferExpansion: true,
    coverRoot: true,
    candidateCommands: (o) => teacher(o).candidates,
    leafValue: (o, root, remaining) => {
      if (!remaining || windowBoundary(o)) {
        stats.rolloutUnknown++;
        return 0;
      }
      const choice = teacher(o);
      const next = sampleTrainingTransition(
        o,
        decisionOwner(o),
        choice.command,
        Math.floor(random() * 4294967296),
      );
      stats.rolloutTransitions++;
      checkPosition(next);
      const value = terminalValue(next, root);
      if (value === null) stats.rolloutUnknown++;
      else stats.rolloutTerminal++;
      return value ?? 0;
    },
  });
  stats.searchTransitions = result.stats.transitions;
  stats.searchSimulations = result.stats.simulations;
  ensure(stats.searchTransitions + stats.rolloutTransitions <= 32, '搜索转移超出上限。');
  if (result.status === 'paused') {
    if (!result.reason.includes('回合外巨大化')) throw new Error(result.reason);
    return {
      command: base.command,
      mode: 'teacher-fallback' as const,
      reason: 'descendant-u7',
      stats,
      policy: null,
    };
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
