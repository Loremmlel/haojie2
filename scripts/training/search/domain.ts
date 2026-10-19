import { ensure } from '../../../src/engine/core/state';
import { allPieces, hasTrait } from '../../../src/engine/core/traits';
import type { Command, Player } from '../../../src/engine/types';
import { decisionOwner } from '../../../src/ai/observation';
import { TrainingActionTree } from '../../../src/ai/training/action-tree';
import { trainingPosition } from '../../../src/ai/training/queries';
import type { Observation } from '../../../src/ai/types';

/** 有限候选搜索只接受经典公开局面；不支持窗口由解码器明确回退。 */
export function checkPosition(observation: Observation) {
  const position = trainingPosition(observation);
  ensure(!observation.mode, '搜索探针仅支持经典模式。');
  ensure(observation.phase !== 'shrine-draft', '搜索探针不支持暗选。');
  ensure(!allPieces(position).some((u) => hasTrait(u, 'u7')), '搜索探针不支持回合外巨大化。');
}

/** 完整遍历分步动作树后才返回；上限不足时抛错，不把截断前缀当合法动作子集。 */
export function commands(observation: Observation, maxNodes = 4096) {
  checkPosition(observation);
  const tree = new TrainingActionTree(observation, decisionOwner(observation));
  const result = new Map<string, Command>();
  let nodes = 0;
  const visit = (cursor: number[]) => {
    ensure(++nodes <= maxNodes && cursor.length <= 256, '完整命令枚举超出预算。');
    const node = tree.node(cursor);
    node.choices.forEach((choice, index) => {
      if (choice.next) visit([...cursor, index]);
      else result.set(JSON.stringify(choice.command), choice.command);
    });
  };
  if (!observation.winner) visit([]);
  return { commands: [...result.values()], nodes };
}

export function terminalValue(observation: Observation, player: Player): number | null {
  return observation.winner === undefined
    ? null
    : observation.winner === 'draw'
      ? 0
      : observation.winner === player
        ? 1
        : -1;
}

/** 有限深度搜索只跨 play 及其强制反应；新召唤/合成窗口由调用方处理。 */
export const windowBoundary = (observation: Observation) =>
  !observation.pending.length && observation.phase !== 'play';
