import type { GamePosition, Unit, Landmark, Kind } from '../types';
import { cloneRuleData } from '../core/clone';
import { CATALOG } from '../catalog';

export type EntityHandle = number & { readonly entityHandle: unique symbol };
export type KindCode = number & { readonly kindCode: unique symbol };
const kindCodes = new Map<Kind, KindCode>(
  Object.values(CATALOG).map((d, i) => [d.id, (i + 1) as KindCode]),
);
interface EntitySlot {
  id: string;
  index: number;
  landmark: boolean;
}
interface EntityTable {
  graphClone: boolean;
  units: Unit[];
  landmarks: Landmark[] | undefined;
  count: number;
  slots: readonly (EntitySlot | undefined)[];
  identities: ReadonlyMap<string, EntityHandle>;
}
const tables = new WeakMap<GamePosition, EntityTable>();
const branchSources = new WeakMap<GamePosition, Partial<GamePosition>>();

/** 句柄属于一个运行分支。布局可共享，槽保存位置而非对象，原位替换不会留下旧实体引用。 */
function table(s: GamePosition, rebuild = false): EntityTable | undefined {
  const t = tables.get(s);
  if (!t) return;
  if (
    !rebuild &&
    t.units === s.units &&
    t.landmarks === s.landmarks &&
    t.count === s.units.length + (s.landmarks?.length ?? 0)
  )
    return t;
  // 既有身份表只读共享；出现新身份才复制，未移动的不可变槽也可继续共享。
  let added: Map<string, EntityHandle> | undefined;
  const slots: (EntitySlot | undefined)[] = Array(t.slots.length);
  const add = (u: Unit, index: number, landmark: boolean) => {
    let h = (added ?? t.identities).get(u.id);
    if (h === undefined) {
      h = slots.length as EntityHandle;
      (added ??= new Map(t.identities)).set(u.id, h);
      slots.push(undefined);
    }
    if (!slots[h]) {
      const previous = t.slots[h];
      slots[h] =
        previous?.id === u.id && previous.index === index && previous.landmark === landmark
          ? previous
          : { id: u.id, index, landmark };
    }
  };
  s.units.forEach((u, i) => add(u, i, false));
  s.landmarks?.forEach((u, i) => add(u, i, true));
  t.units = s.units;
  t.landmarks = s.landmarks;
  t.count = s.units.length + (s.landmarks?.length ?? 0);
  t.slots = slots;
  if (added) t.identities = added;
  return t;
}
function slotUnit(s: GamePosition, slot?: EntitySlot) {
  const u = slot ? (slot.landmark ? s.landmarks?.[slot.index] : s.units[slot.index]) : undefined;
  return u?.id === slot?.id ? u : undefined;
}
export function entityHandle(s: GamePosition, id: string): EntityHandle | undefined {
  let t = table(s),
    h = t?.identities.get(id);
  if (t && (h === undefined || !slotUnit(s, t.slots[h]))) {
    // 查询不存在的身份不代表布局失效；只有同长度替换实际引入了该身份才重建。
    if (!s.units.some((u) => u.id === id) && !s.landmarks?.some((u) => u.id === id)) return;
    t = table(s, true);
    h = t?.identities.get(id);
  }
  return h !== undefined && slotUnit(s, t?.slots[h]) ? h : undefined;
}
export function entityAt(s: GamePosition, handle: EntityHandle): Unit | undefined {
  let t = table(s),
    u = slotUnit(s, t?.slots[handle]);
  if (t && !u) {
    t = table(s, true);
    u = slotUnit(s, t?.slots[handle]);
  }
  return u;
}
export function entityKind(s: GamePosition, handle: EntityHandle): KindCode | undefined {
  const u = entityAt(s, handle);
  return u && kindCodes.get(u.kind);
}
export function pieceById(s: GamePosition, id: string | undefined): Unit | undefined {
  if (id === undefined) return;
  if (!tables.has(s))
    return (
      s.units.find((u) => u.id === id) ??
      s.landmarks?.find((u) => u.id === id && u.hp > 0 && u.dormantSince === undefined)
    );
  const h = entityHandle(s, id);
  if (h === undefined) return;
  const t = table(s)!,
    slot = t.slots[h],
    u = slotUnit(s, slot);
  if (slot?.landmark && (u!.hp <= 0 || (u as Landmark).dormantSince !== undefined)) return;
  return u;
}

/** 冷字段仅在规则实际写入前分离；只读属性/路径/编码不再触发复制。 */
export function own<K extends keyof GamePosition>(s: GamePosition, key: K): GamePosition[K] {
  const source = branchSources.get(s);
  if (source && s[key] === source[key] && s[key] && typeof s[key] === 'object') {
    if (key === 'clockFrames') {
      const frames = s.clockFrames!;
      s.clockFrames = { 1: { ...frames[1] }, 2: { ...frames[2] } };
    } else s[key] = cloneRuleData(s[key]);
  }
  return s[key];
}

function register<S extends GamePosition>(s: S, previous?: EntityTable, graphClone = false): S {
  tables.set(s, {
    graphClone: previous?.graphClone ?? graphClone,
    units: previous ? s.units : [],
    landmarks: previous ? s.landmarks : undefined,
    count: previous?.count ?? -1,
    slots: previous?.slots ?? [],
    identities: previous?.identities ?? new Map(),
  });
  return s;
}

/** 实体采用紧凑结构复制；频繁属性读取不经过描述符，嵌套可变历史按已知类型隔离。 */
function copyUnit<T extends Unit>(u: T): T {
  return {
    ...u,
    attacked: u.attacked.slice(),
    effects: u.effects.map((e) => ({ ...e })),
    equipment: u.equipment.slice(),
    ...(u.traits ? { traits: u.traits.slice() } : {}),
    ...(u.guardSourceIds ? { guardSourceIds: u.guardSourceIds.slice() } : {}),
    ...(u.equipmentIds ? { equipmentIds: { ...u.equipmentIds } } : {}),
    ...(u.receivedDamage ? { receivedDamage: u.receivedDamage.map((v) => ({ ...v })) } : {}),
    ...(u.abilityUsage
      ? {
          abilityUsage: Object.fromEntries(
            Object.entries(u.abilityUsage).map(([k, v]) => [k, { ...v }]),
          ),
        }
      : {}),
    ...(u.abilityCharges
      ? {
          abilityCharges: Object.fromEntries(
            Object.entries(u.abilityCharges).map(([k, v]) => [k, { ...v }]),
          ),
        }
      : {}),
  };
}

/** 仅在外部局面进入拥有型运行器时复制；后续规则、预检和编码直接消费同一类型化数据。 */
export function importRuntimePosition<S extends GamePosition>(source: S): S {
  const copy = cloneRuleData(source);
  const seen = new WeakSet<object>();
  // JS 外部宿主允许跨实体别名及环；只在边界识别，使用明确的整图复制兼容入口。
  const aliased = (v: unknown): boolean => {
    if (!v || typeof v !== 'object') return false;
    if (seen.has(v)) return true;
    seen.add(v);
    if (!Array.isArray(v) && Object.getPrototypeOf(v) !== Object.prototype) return true;
    return Object.values(v).some(aliased);
  };
  return register(copy, undefined, aliased(copy));
}

export function isRuntimePosition(s: GamePosition): boolean {
  return tables.has(s);
}

/** 外部输入已经整图复制时直接交付这份独立快照；移除运行标记，允许宿主自由修改。 */
export function exportOwnedPosition<S extends GamePosition>(s: S): S {
  if (branchSources.has(s)) throw new Error('共享分支必须先复制再导出。');
  tables.delete(s);
  return s;
}

export function forkRuntimePosition<S extends GamePosition>(source: S): S {
  if (tables.get(source)?.graphClone) return register(cloneRuleData(source), table(source));
  const result = { ...source };
  // 只保留实际共享的子树，不能让连续运行局面反向持有全部旧根。
  branchSources.set(result, {
    hands: source.hands,
    pending: source.pending,
    deaths: source.deaths,
    hazards: source.hazards,
    siphons: source.siphons,
    iceMarks: source.iceMarks,
    log: source.log,
    auras: source.auras,
    clockFrames: source.clockFrames,
    shrineDraft: source.shrineDraft,
    summonOffer: source.summonOffer,
    shrineSetupDone: source.shrineSetupDone,
  });
  result.turns = { ...source.turns };
  result.bases = { ...source.bases };
  result.heads = { ...source.heads };
  result.bonus = { ...source.bonus };
  result.deployRows = { ...source.deployRows };
  result.baseEffects = {
    1: source.baseEffects[1].map((e) => ({ ...e })),
    2: source.baseEffects[2].map((e) => ({ ...e })),
  };
  result.units = source.units.map(copyUnit);
  if (source.landmarks) result.landmarks = source.landmarks.map(copyUnit);
  return register(result, table(source));
}

/** 公开根只登记公开实体，不保留完整状态、随机源或运行器引用。 */
export function registerPublicPosition<S extends GamePosition>(s: S, source?: GamePosition): S {
  return tables.has(s) ? s : register(s, source ? table(source) : undefined);
}
