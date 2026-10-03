import { visibleShrineDraft } from '../engine/online/player-view';
import { deploymentRows } from '../engine/core/geometry';
import type { GameState, Player } from '../engine/types';
import type { Observation } from './types';
import { registerPublicPosition } from '../engine/runtime/position';
import type { GamePosition } from '../engine/types';
const publicPositions = new WeakMap<Observation, GamePosition>();
const frozen = new WeakSet<object>();

/** 冻结共享的公开子树一次；视图无权威根引用，下一条命令通过独立运行分支写入。 */
function freezePublic<T>(value: T): T {
  if (!value || typeof value !== 'object' || frozen.has(value)) return value;
  frozen.add(value);
  for (const child of Object.values(value)) freezePublic(child);
  return Object.freeze(value);
}

export function publicPositionView(s: GameState, viewer: Player = decisionOwner(s)): Observation {
  const view = freezePublic(fields(s, viewer));
  // 只共享公开身份与数字槽；登记表没有权威根或随机状态的反向引用。
  const position = registerPublicPosition(freezePublic({ ...view, log: [], events: [] }), s);
  publicPositions.set(view, position);
  return view;
}

/** 只接受由白名单入口登记的视图；普通外部观察仍在训练查询入口验证。 */
export function viewPosition(view: Observation): GamePosition | undefined {
  return publicPositions.get(view);
}
export const decisionOwner = (s: Pick<GameState, 'pending' | 'active'>): Player =>
  s.pending[0]?.owner ?? s.active;
function fields(s: GameState, viewer: Player = decisionOwner(s)): Observation {
  // 明确使用白名单：未来在 GameState 添加秘密字段，也不能自动向 AI 暴露。
  return {
    version: s.version,
    serial: s.serial,
    ply: s.ply,
    active: s.active,
    phase: s.phase,
    summonSlots: s.summonSlots,
    turns: s.turns,
    bases: s.bases,
    baseEffects: s.baseEffects,
    heads: s.heads,
    hands: s.hands,
    bonus: s.bonus,
    deployRows: { 1: deploymentRows(s, 1), 2: deploymentRows(s, 2) },
    units: s.units,
    pending: s.pending,
    deaths: s.deaths,
    hazards: s.hazards,
    siphons: s.siphons,
    iceMarks: s.iceMarks,
    ...(s.mode ? { mode: s.mode } : {}),
    ...(s.landmarks ? { landmarks: s.landmarks } : {}),
    ...(s.auras ? { auras: s.auras } : {}),
    ...(s.regularSummons !== undefined ? { regularSummons: s.regularSummons } : {}),
    ...(s.summonOffer ? { summonOffer: s.summonOffer } : {}),
    ...(s.shrineSetupDone ? { shrineSetupDone: s.shrineSetupDone } : {}),
    ...(s.clockFrames ? { clockFrames: s.clockFrames } : {}),
    ...(s.shrineDraft
      ? {
          shrineDraft: visibleShrineDraft(s.shrineDraft, viewer),
        }
      : {}),
    ...(s.winner ? { winner: s.winner } : {}),
  };
}
export function observe(s: GameState, viewer: Player = decisionOwner(s)): Observation {
  return structuredClone(fields(s, viewer));
}
export function imagined(o: Observation): GameState {
  return { ...structuredClone(o), seed: 1, rng: 1, events: [], log: [] };
}
export function hash(text: string): number {
  let result = 2166136261;
  for (let i = 0; i < text.length; i++) result = Math.imul(result ^ text.charCodeAt(i), 16777619);
  return result >>> 0;
}
/** 供界面复用的紧凑缓存键；搜索去重还保留完整局面文本。 */
export function fingerprint(s: GameState | Observation): string {
  const text = JSON.stringify('rng' in s ? fields(s) : s);
  return `${hash(text).toString(36)}:${text.length}`;
}
export function positionKey(s: GameState): string {
  const o = fields(s);
  // 事件标识影响 serial，但不影响规则；保留对象 ID 及反应、死亡记录的关联。
  return JSON.stringify({ ...o, serial: 0 });
}
