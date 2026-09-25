import { ensure } from '../../../src/engine/core/state';
import { decisionOwner } from '../../../src/ai/observation';
import { trainingDistribution } from '../../../src/ai/training/simulation';
import type { Observation } from '../../../src/ai/types';
import { commands, terminalValue, windowBoundary } from './puct';

/**
 * 有限命令深度的全宽期望极大极小参照；概率树必须完整枚举，否则明确失败。
 * 只用于小型验证夹具，不能称为游戏解；未知边界0与探针相同，不用教师或模型评分。
 * rootValues记录同一候选域中的全部等价最优动作，不把单个教师选择作为正确答案。
 */
export function reference(
  observation: Observation,
  horizon: number,
  limits: { maxActions?: number; maxActionNodes?: number } = {},
) {
  ensure(Number.isSafeInteger(horizon) && horizon > 0, '参照深度须为正整数。');
  for (const limit of Object.values(limits))
    ensure(Number.isSafeInteger(limit) && limit > 0, '参照工作量上限须为正整数。');
  const rootActor = decisionOwner(observation);
  const cache = new Map<string, number>();
  const stats = {
    transitionAttempts: 0,
    actions: 0,
    actionNodes: 0,
    terminalLeaves: 0,
    cutoffLeaves: 0,
  };
  const expand = (o: Observation) => {
    const result = commands(o, limits.maxActionNodes);
    stats.actionNodes += result.nodes;
    ensure(result.commands.length, '参照遇到非终局无命令。');
    return result.commands;
  };
  const value = (o: Observation, depth: number): number => {
    const ended = terminalValue(o, rootActor);
    if (ended !== null) {
      stats.terminalLeaves++;
      return ended;
    }
    if (!depth || windowBoundary(o)) {
      stats.cutoffLeaves++;
      return 0;
    }
    const key = `${depth}:${JSON.stringify(o)}`;
    const previous = cache.get(key);
    if (previous !== undefined) return previous;
    const values = expand(o).map((command) => actionValue(o, command, depth));
    const best = decisionOwner(o) === rootActor ? Math.max(...values) : Math.min(...values);
    cache.set(key, best);
    return best;
  };
  const actionValue = (
    o: Observation,
    command: ReturnType<typeof expand>[number],
    depth: number,
  ) => {
    ensure(stats.actions < (limits.maxActions ?? Infinity), '精确参照动作预算耗尽。');
    const distribution = trainingDistribution(o, decisionOwner(o), command, 0);
    stats.transitionAttempts += distribution.attempts;
    stats.actions++;
    ensure(!distribution.sampled && distribution.outcomes.length > 0, '参照需要非空完整概率树。');
    ensure(
      Math.abs(distribution.outcomes.reduce((n, out) => n + out.weight, 0) - 1) < 1e-9,
      '概率和不是1。',
    );
    return distribution.outcomes.reduce(
      (sum, out) => sum + out.weight * value(out.observation, depth - 1),
      0,
    );
  };
  const rootValues = expand(observation).map((command) => ({
    command,
    value: actionValue(observation, command, horizon),
  }));
  return { rootValues, best: Math.max(...rootValues.map((r) => r.value)), stats };
}
