import { TrainingTeacher } from '../../../../src/ai/training/teacher';
import { DIFFICULTIES } from '../../../../src/ai/difficulty';
import type { Difficulty, Observation } from '../../../../src/ai/types';
import { cloneRuleData } from '../../../../src/engine/core/clone';
import { decide } from '../../../../src/ai/planning/search';
import type { Player } from '../../../../src/engine/types';

// 嵌入式 V8 不提供 Web structuredClone；公开协议只含规则纯数据，复用同一复制器。
// 防止未来引入非数据类型后递归回退；不能用 JSON 序列化静默删 undefined。
if (typeof globalThis.structuredClone === 'undefined') {
  let cloning = false;
  globalThis.structuredClone = <T>(value: T): T => {
    if (cloning) throw new Error('教师适配只允许规则纯数据');
    cloning = true;
    try {
      return cloneRuleData(value);
    } finally {
      cloning = false;
    }
  };
}

// 独立 V8 适配仅提供真实教师；不实现规则副本，不接收权威 RNG。
let teacher: TrainingTeacher;
export function reset(difficulty: Difficulty, simulations?: number) {
  teacher = new TrainingTeacher(difficulty, simulations);
  return DIFFICULTIES[difficulty];
}
export function next(observation: Observation, actor?: Player, optional = false) {
  // 原生产搜索在非决策方窗口返回 null；显式保留该策略，不另造巨大化教师。
  if (optional && actor)
    return decide(observation, actor, teacher.difficulty, { mode: 'work' }).command;
  return teacher.next(observation).command;
}
