import { HEIGHT, WIDTH } from '../../../engine/core/geometry';
import { ensure } from '../../../engine/core/state';
import { SYNTHESIS_RECIPES } from '../../../engine/setup/synthesis';
import type { Player } from '../../../engine/types';
import type { Observation } from '../../types';
import type { ActionNode } from '../action-tree';
import { COMMANDS, DIRECTIONS, kindIndex, MODES } from './schema';
import { createPositionEncoder, createSamplingPositionEncoder } from './state';

export const DECISION_STAGES = [
  'action',
  'material',
  'chosen',
  'target',
  'point',
  'death',
  'row',
  'column',
  'direction',
  'path',
];
export interface EncodedDecision {
  entities: number[][];
  kinds: number[];
  entity_mask: boolean[];
  globals: number[];
  candidates: number[][];
  sources: number[];
  targets: number[];
  candidate_mask: boolean[];
}

/** 浏览器、数据准备与网络服务共用同一输入编码；不接受教师标签作为特征。 */
export function encodeDecision(
  observation: Observation,
  viewer: Player,
  node: ActionNode,
): EncodedDecision {
  return createDecisionEncoder(observation, viewer)(node);
}

/** 同一不可变公开观察的参数节点共用固定编码；返回张量相互独立。 */
export function createDecisionEncoder(observation: Observation, viewer: Player) {
  const encode = createPositionEncoder(observation, viewer);
  return decisionEncoder(observation, viewer, encode);
}
/** 仅供同一决策内的只读前向；固定实体行共享，候选与前缀仍逐节点独立。 */
export function createSamplingEncoder(
  observation: Observation,
  viewer: Player,
  reuseBuffers = false,
) {
  return decisionEncoder(
    observation,
    viewer,
    createSamplingPositionEncoder(observation, viewer, reuseBuffers),
    reuseBuffers,
  );
}
function decisionEncoder(
  observation: Observation,
  viewer: Player,
  encode: ReturnType<typeof createPositionEncoder>,
  reuseBuffers = false,
) {
  // 仅内部同步前向显式启用；默认出口仍独立拥有，不能把借用张量留给异步消费者。
  const storage: number[][] = [];
  const workspace = {
    candidates: [] as number[][],
    sources: [] as number[],
    targets: [] as number[],
    entity_mask: [] as boolean[],
    candidate_mask: [] as boolean[],
  };
  return (node: ActionNode): EncodedDecision => {
    ensure(node.choices.length > 0, '空动作分支须回溯，不能伪造合法候选。');
    const state = encode(node.prefix);
    const index = (id?: string) => {
      if (id === undefined) return -1;
      const result = state.indices.get(id);
      if (result === undefined) ensure(false, `动作引用缺失的公开实体 ${id}。`);
      return result;
    };
    const sources = reuseBuffers ? workspace.sources : [],
      targetIndices = reuseBuffers ? workspace.targets : [],
      candidates = reuseBuffers ? workspace.candidates : [];
    sources.length = targetIndices.length = candidates.length = node.choices.length;
    node.choices.forEach((choice, i) => {
      const c = choice.command;
      const row = reuseBuffers
        ? (storage[i] ??= Array<number>(64)).fill(0)
        : Array<number>(64).fill(0);
      row[COMMANDS.indexOf(c.type)] = 1;
      if (c.mode !== undefined) {
        const mode = MODES.indexOf(c.mode);
        if (mode < 0) ensure(false, `候选模式未编码：${c.mode}。`);
        row[23 + mode] = 1;
      }
      row[36] = (c.x ?? 0) / WIDTH;
      row[37] = (c.y ?? 0) / HEIGHT;
      row[38] = (c.row ?? 0) / HEIGHT;
      row[39] = (c.column ?? 0) / WIDTH;
      row[40] = c.ultimate === undefined ? -1 : Number(c.ultimate);
      row[41] = c.charge === undefined ? -1 : Number(c.charge);
      row[42] = c.direction === undefined ? 0 : (DIRECTIONS.indexOf(c.direction) + 1) / 4;
      row[43] = kindIndex(c.ability) / 128;
      row[44] = kindIndex(c.chosenKind) / 128;
      row[45] = kindIndex(c.shrineKind) / 128;
      row[46] =
        c.recipeId === undefined
          ? 0
          : (SYNTHESIS_RECIPES.findIndex((r) => r.id === c.recipeId) + 1) / 8;
      row[47] = c.parity === undefined ? 0 : c.parity === 'odd' ? 1 : -1;
      row[48] = (DECISION_STAGES.indexOf(node.stage) + 1) / 16;
      row[49] = Number(choice.status !== 'parameter');
      row[50] = Number(choice.status === 'uncertain');
      row[51] = c.offerIndices?.[0] === undefined ? -1 : c.offerIndices[0] / 16;
      row[52] = c.offerIndices?.[1] === undefined ? -1 : c.offerIndices[1] / 16;
      row[53] = (c.materialIds?.length ?? 0) / 3;
      row[54] = (c.sacrificeIds?.length ?? 0) / 2;
      row[55] = (c.path?.length ?? 0) / 117;
      row[56] = state.reference(c.secondId) / 256;
      row[57] = state.reference(c.deathId) / 256;
      row[58] = Number(choice.key === 'commit-path');
      row[59] = c.player === undefined ? 0 : c.player === viewer ? 1 : -1;
      const point = c.path?.at(-1);
      row[60] = (point?.x ?? 0) / WIDTH;
      row[61] = (point?.y ?? 0) / HEIGHT;
      sources[i] = index(
        c.unitId ??
          c.cardId ??
          (c.type === 'react' ? observation.pending[0]?.source.id : undefined),
      );
      targetIndices[i] = index(choice.subject ?? c.targetId ?? c.deathId);
      candidates[i] = row;
    });
    const entity_mask = reuseBuffers ? workspace.entity_mask : [];
    const candidate_mask = reuseBuffers ? workspace.candidate_mask : [];
    entity_mask.length = state.entities.length;
    candidate_mask.length = candidates.length;
    entity_mask.fill(true);
    candidate_mask.fill(true);
    return {
      entities: state.entities,
      kinds: state.kinds,
      globals: state.globals,
      entity_mask,
      candidates,
      sources,
      targets: targetIndices,
      candidate_mask,
    };
  };
}
