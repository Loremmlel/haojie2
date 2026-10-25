import { allocateBudget, emptyBudget, type TurnBudget } from '../../../../src/ai/budget';
import { cachedDecision } from '../../../../src/ai/planning/plan-cache';
import { decide } from '../../../../src/ai/planning/search';
import { decisionOwner, imagined } from '../../../../src/ai/observation';
import { DIFFICULTIES } from '../../../../src/ai/difficulty';
import { cloneRuleData } from '../../../../src/engine/core/clone';
import type { Player } from '../../../../src/engine/types';
import type { Difficulty, Observation, PlanStep } from '../../../../src/ai/types';

// 嵌入式 V8 仅补齐平台 API，算法、预算和缓存收尾复用游戏 AI。
if (typeof globalThis.structuredClone === 'undefined') {
  let cloning = false;
  globalThis.structuredClone = <T>(value: T): T => {
    if (cloning) throw new Error('AI 测评只允许规则纯数据');
    cloning = true;
    try {
      return cloneRuleData(value);
    } finally {
      cloning = false;
    }
  };
}
let difficulty: Difficulty;
let budgets: Record<Player, TurnBudget>;
let plans: Record<Player, PlanStep[]>;
export function reset(level: Difficulty) {
  if (!['easy', 'medium', 'hard'].includes(level)) throw new Error('未知 AI 难度');
  difficulty = level;
  budgets = { 1: emptyBudget(), 2: emptyBudget() };
  plans = { 1: [], 2: [] };
  return DIFFICULTIES[level];
}
export function next(observation: Observation, actor: Player, optional: boolean) {
  if (optional) return decide(observation, actor, difficulty, { mode: 'work' }).command;
  const side = decisionOwner(observation);
  if (actor !== side) throw new Error('AI 决策所有者不符');
  if (budgets[side].ply !== observation.ply) {
    budgets[side] = { ...emptyBudget(), ply: observation.ply };
    plans[side] = [];
  }
  const budget = budgets[side];
  if (budget.commands >= 200) throw new Error('AI 本回合达到 200 条命令');
  const decision =
    cachedDecision(observation, plans[side]) ??
    decide(observation, side, difficulty, {
      ...allocateBudget(imagined(observation), difficulty, budget),
      mode: 'work',
      trace: false,
    });
  if (!decision.command) throw new Error('AI 没有合法候选');
  plans[side] = decision.plan.slice(1);
  budget.nodes += decision.stats.simulations;
  budget.commands++;
  return decision.command;
}
