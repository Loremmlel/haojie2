import { ensure } from '../../../../src/engine/core/state';
import { decisionOwner, hash } from '../../../../src/ai/observation';
import { decide } from '../../../../src/ai/planning/search';
import {
  sampleTrainingTransition,
  simulationRandomSource,
  trainingDistribution,
} from '../../../../src/ai/training/simulation';
import type { Observation } from '../../../../src/ai/types';
import type { Command } from '../../../../src/engine/types';
import { checkPosition, terminalValue, windowBoundary, type ProbeOptions } from '../puct';

/**
 * 实验叶端最多续演一条完整命令，只返回实际终局回报；未知0只作估计，不生成标签。
 * 教师只负责为当前实际操作者选招，不提供价值真值；其工作量、机会转移和结果分开计数。
 * 仅接收公开局面，不改输入；随机源独立于主搜索及正式对局，未支持局面/非法命令明确失败。
 */
export function terminalRollout(sampleSeed: number) {
  const random = simulationRandomSource(hash(`terminal-rollout:${sampleSeed}`));
  const stats = {
    calls: 0,
    teacherCalls: 0,
    teacherWork: 0,
    teacherMs: 0,
    transitions: 0,
    transitionMs: 0,
    reactionCalls: 0,
    terminal: 0,
    wins: 0,
    losses: 0,
    draws: 0,
    unknown: 0,
  };
  const leafValue: NonNullable<ProbeOptions['leafValue']> = (o, root, remaining) => {
    stats.calls++;
    ensure(remaining <= 1 && remaining >= 0, '短续演只接受至多一条剩余命令。');
    if (remaining === 0 || windowBoundary(o)) {
      stats.unknown++;
      return 0;
    }
    checkPosition(o);
    const actor = decisionOwner(o);
    if (o.pending.length) stats.reactionCalls++;
    const start = performance.now();
    const decision = decide(o, actor, 'easy', { simulations: 40, mode: 'work' });
    stats.teacherMs += performance.now() - start;
    stats.teacherCalls++;
    stats.teacherWork += decision.stats.simulations;
    ensure(decision.command, '续演教师未返回完整命令。');
    ensure(decision.stats.simulations <= 40, '续演教师工作量超限。');
    const transitionStart = performance.now();
    const next = sampleTrainingTransition(
      o,
      actor,
      decision.command,
      Math.floor(random() * 4294967296),
    );
    stats.transitionMs += performance.now() - transitionStart;
    stats.transitions++;
    checkPosition(next);
    const value = terminalValue(next, root);
    if (value === null) stats.unknown++;
    else {
      stats.terminal++;
      if (value === 1) stats.wins++;
      else if (value === -1) stats.losses++;
      else stats.draws++;
    }
    return value ?? 0;
  };
  return { leafValue, stats };
}

/** 离线核对一个指定命令的即时终局概率；不枚举其它命令，非终局只记未知，不宣称全局最优。 */
export function immediateCertificate(o: Observation, command: Command) {
  const actor = decisionOwner(o);
  const distribution = trainingDistribution(o, actor, command, 0);
  if (distribution.sampled || !distribution.outcomes.length)
    return { status: 'unresolved' as const, reason: '没有完整概率树' };
  ensure(
    Math.abs(distribution.outcomes.reduce((n, r) => n + r.weight, 0) - 1) < 1e-9,
    '概率和不是1。',
  );
  let value = 0,
    terminalProbability = 0,
    winProbability = 0;
  for (const r of distribution.outcomes) {
    const ended = terminalValue(r.observation, actor);
    if (ended !== null) {
      terminalProbability += r.weight;
      value += r.weight * ended;
      if (ended === 1) winProbability += r.weight;
    }
  }
  return {
    status: 'exact' as const,
    value,
    terminalProbability,
    winProbability,
    attempts: distribution.attempts,
  };
}
