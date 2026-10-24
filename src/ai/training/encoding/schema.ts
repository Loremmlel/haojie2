import { CATALOG } from '../../../engine/catalog';
import { ensure } from '../../../engine/core/state';
import type { Command, Kind } from '../../../engine/types';

/** 编码语义变化必须更换版本；词表随数据文件保存并核对，不能静默重排旧模型输入。 */
export const ENCODING = 'haojie-public-tables-v3';
export const GLOBAL_ROLE = 32;
export const KIND_VOCAB = CATALOG.map((d) => d.id);
const kindIndices = new Map(KIND_VOCAB.map((kind, i) => [kind, i + 1]));
const kindKeys = new Map(KIND_VOCAB.map((kind) => [String(kind), kind]));
export const ROLES = [
  'base',
  'unit',
  'landmark',
  'card',
  'death',
  'effect',
  'equipment',
  'ability',
  'attacked',
  'guard',
  'received',
  'hazard',
  'siphon',
  'ice',
  'reaction',
  'snapshot',
  'frame',
  'aura',
  'draft-offer',
  'draft-choice',
  'summon-offer',
  'prefix',
  'argument',
  'path',
] as const;
export type EntityRole = (typeof ROLES)[number];
export const COMMANDS: Command['type'][] = [
  'end',
  'summon',
  'begin',
  'reroll',
  'deploy',
  'move',
  'attack',
  'charge',
  'skill',
  'cast',
  'equip',
  'craft',
  'synthesize',
  'skip-synthesis',
  'react',
  'finish-mode',
  'choose-shrine',
  'finish-shrine-setup',
  'activate-aura',
  'extra-summon',
  'choose-summons',
  'clock',
  'shatter',
];
export const MODES = [
  'none',
  'move',
  'attack',
  'skill',
  'damage',
  'heal',
  'summon',
  'single',
  'double',
  'row',
  'column',
  'pull',
  'normal',
];
export const PHASES = ['shrine-draft', 'shrine-setup', 'synthesis', 'summon', 'play'];
export const EFFECTS = [
  'attack',
  'immune',
  'execute',
  'convert',
  'mark',
  'freeze',
  'burn',
  'stun',
  'inner-fire',
];
export const REACTIONS = ['death-shot', 'reflect', 'bounce', 'hut-spawn', 'hit-pull'];
export const DIRECTIONS = ['up', 'down', 'left', 'right'] as const;
export const UNIT_FIELDS = [
  'x',
  'y',
  'hp',
  'maxHp',
  'size',
  'born',
  'offset',
  'deployedAt',
  'chargedOnDeploy',
  'mode',
  'operations',
  'shots',
  'moves',
  'bonusAttacks',
  'bonusSequence',
  'weaponFirstUsed',
  'charge',
  'readyCharge',
  'chargeType',
  'lastCharge',
  'upgrades',
  'kills',
  'attackBonus',
  'rangeBonus',
  'guardUsed',
  'rerollUsedPly',
  'silenced',
  'freeUsed',
  'onceUsed',
  'extraOperations',
  'bannerHp',
  'overMaxFromBanner',
  'bladeQualified',
  'expiresAt',
  'hookReadyAt',
  'hookExpiresAt',
  'dormantSince',
  'rebuildTicks',
] as const;

export function kindIndex(kind: Kind | undefined): number {
  if (kind === undefined) return 0;
  const index = kindIndices.get(kind);
  if (index === undefined || index > 128)
    ensure(false, `编码词表不支持棋子 ${String(kind)}，请升级编码版本。`);
  return index;
}

export function kindFromKey(key: string): Kind {
  const kind = kindKeys.get(key);
  if (kind === undefined) ensure(false, `未知能力/装备词表键 ${key}。`);
  return kind;
}

export function category(
  value: string | undefined,
  vocabulary: readonly string[],
): number | undefined {
  if (value === undefined) return undefined;
  const index = vocabulary.indexOf(value);
  if (index < 0) ensure(false, `编码词表不支持 ${value}。`);
  return index + 1;
}

export function knownKeys(value: object, allowed: readonly string[], label: string): void {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    ensure(false, `${label}必须是对象。`);
  for (const key of Object.keys(value))
    if (!allowed.includes(key)) ensure(false, `${label}含未编码字段 ${key}。`);
}

/** 仅连续量使用有限、不截断的缩放。身份、索引和存在位不经浮点投影。 */
export function numeric(value: number | boolean): number {
  if (typeof value === 'boolean') return Number(value);
  ensure(typeof value === 'number' && Number.isFinite(value), '编码数值必须有限。');
  return (Math.sign(value) * Math.log1p(Math.abs(value))) / 8;
}

const roundedFloat = new Float32Array(1);
const roundedBits = new Uint32Array(roundedFloat.buffer);
/** 连续量按 BF16 最近偶数舍入，输出保存原始位。 */
export function bf16(value: number): number {
  roundedFloat[0] = value;
  const bits = roundedBits[0];
  if ((bits & 0x7fffffff) > 0x7f800000) return (bits >>> 16) | 0x40;
  return ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16) & 0xffff;
}

/** 原始类型化行；调试 JSON 将64位整数写成十进制字符串，避免传输中丢失精度。 */
export class FeatureRow {
  readonly ids = new BigInt64Array(60);
  present = 0n;
  integers = 0n;
  readonly continuous = new Uint16Array(52);

  clone(): FeatureRow {
    const out = new FeatureRow();
    out.ids.set(this.ids);
    out.present = this.present;
    out.integers = this.integers;
    out.continuous.set(this.continuous);
    return out;
  }

  toJSON() {
    return {
      ids: Array.from(this.ids, String),
      present: this.present.toString(),
      integers: this.integers.toString(),
      continuous: Array.from(this.continuous),
    };
  }
}

/** 候选只包含离散选择，保留全部字段及原始顺序。 */
export class ActionRow {
  readonly ids = new BigInt64Array(64);

  reset(): ActionRow {
    this.ids.fill(0n);
    return this;
  }

  toJSON() {
    return Array.from(this.ids, String);
  }
}

export function valuesInto(row: FeatureRow, values: (number | boolean | undefined)[]): void {
  ensure(values.length <= 52, '实体字段超过52项，请升级编码结构，禁止截断。');
  for (let i = 0; i < values.length; i++) {
    valueInto(row, i, values[i]);
  }
}

/** 布尔量及可表示整数直接保存；其余数值只占用独立 BF16 连续槽。 */
export function valueInto(row: FeatureRow, i: number, value: number | boolean | undefined): void {
  if (value === undefined) return;
  ensure(i >= 0 && i < 52, '实体字段超过52项，请升级编码结构，禁止截断。');
  const n = Number(value);
  ensure(Number.isFinite(n), '编码数值必须有限。');
  const bit = 1n << BigInt(i);
  row.present |= bit;
  row.integers &= ~bit;
  row.ids[8 + i] = 0n;
  row.continuous[i] = 0;
  if (Number.isInteger(n)) {
    const integer = BigInt(n);
    if (integer >= -(1n << 63n) && integer < 1n << 63n) {
      row.ids[8 + i] = integer;
      row.integers |= bit;
      return;
    }
  }
  row.continuous[i] = bf16(numeric(n));
}

export const ENCODING_SCHEMA = {
  encoding: ENCODING,
  row_ids: 60,
  row_values: 52,
  action_ids: 64,
  global_role: GLOBAL_ROLE,
  kind_count: 256,
  kinds: KIND_VOCAB,
  roles: ROLES,
  commands: COMMANDS,
  modes: MODES,
  phases: PHASES,
  effects: EFFECTS,
  reactions: REACTIONS,
  unit_fields: UNIT_FIELDS,
};
