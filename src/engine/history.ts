import { applyCommand } from './game';
import { normalizeHornStorage } from './migrations';
import { definition, isStored } from './catalog';
import { cells, inside, key, basePoint, equal } from './geometry';
import type { Command, GameState, Player, Unit } from './types';
export interface Session {
  format: 'haojie-session-v2';
  present: GameState;
  past: GameState[];
  future: GameState[];
}
const LIMIT = 60;
export const createSession = (present: GameState): Session => ({
  format: 'haojie-session-v2',
  present,
  past: [],
  future: [],
});
export function dispatch(s: Session, c: Command): Session {
  return {
    ...s,
    present: applyCommand(s.present, c),
    past: [...s.past, s.present].slice(-LIMIT),
    future: [],
  };
}
export function undo(s: Session): Session {
  if (!s.past.length) return s;
  return {
    ...s,
    present: s.past.at(-1)!,
    past: s.past.slice(0, -1),
    future: [s.present, ...s.future].slice(0, LIMIT),
  };
}
export function redo(s: Session): Session {
  if (!s.future.length) return s;
  return {
    ...s,
    present: s.future[0],
    past: [...s.past, s.present].slice(-LIMIT),
    future: s.future.slice(1),
  };
}
const object = (v: unknown): v is Record<string, any> =>
  !!v && typeof v === 'object' && !Array.isArray(v);
const num = (v: unknown, min = 0, max = 1e9) =>
  typeof v === 'number' && Number.isFinite(v) && v >= min && v <= max;
const int = (v: unknown, min = 0, max = Number.MAX_SAFE_INTEGER) =>
  num(v, min, max) && Number.isSafeInteger(v);
const text = (v: unknown, max = 150) => typeof v === 'string' && v.length > 0 && v.length <= max;
const player = (p: unknown) => p === 1 || p === 2;
const list = (v: unknown, max = 2000): v is any[] => Array.isArray(v) && v.length <= max;
const validKind = (k: unknown, unit = false) => {
  try {
    const d = definition(k as Unit['kind']);
    return !unit || !isStored(d);
  } catch {
    return false;
  }
};
function effects(v: unknown) {
  return (
    list(v, 1000) &&
    v.every(
      (e) =>
        object(e) &&
        [
          'attack',
          'immune',
          'execute',
          'convert',
          'mark',
          'freeze',
          'burn',
          'stun',
          'inner-fire',
        ].includes(e.type) &&
        int(e.from) &&
        int(e.until) &&
        e.until > e.from &&
        player(e.owner) &&
        (e.amount === undefined || num(e.amount)) &&
        (e.sourceId === undefined || text(e.sourceId)) &&
        (e.global === undefined || typeof e.global === 'boolean'),
    )
  );
}
function validUnit(u: unknown, dead = false): u is Unit {
  if (
    !object(u) ||
    !text(u.id) ||
    !validKind(u.kind, true) ||
    !player(u.owner) ||
    !inside(u as Unit) ||
    ![1, 2].includes(u.size)
  )
    return false;
  if (
    !num(u.hp, dead ? 0 : 0.000001) ||
    !num(u.maxHp, dead ? 0 : 0.000001) ||
    u.hp > u.maxHp ||
    !int(u.born, -1) ||
    !int(u.offset) ||
    !int(u.deployedAt) ||
    typeof u.chargedOnDeploy !== 'boolean' ||
    u.offset % 2 !== 0
  )
    return false;
  if (
    !['none', 'attack', 'move', 'skill', 'charge'].includes(u.mode) ||
    ![
      u.operations,
      u.shots,
      u.moves,
      u.bonusAttacks,
      u.charge,
      u.readyCharge,
      u.upgrades,
      u.kills,
      u.rangeBonus,
    ].every((v) => int(v)) ||
    !num(u.attackBonus, -1e6) ||
    !int(u.lastCharge, -1) ||
    !int(u.freeUsed, -1)
  )
    return false;
  if (
    !['move', 'attack', 'skill'].includes(u.chargeType) ||
    !['bonusSequence', 'weaponFirstUsed', 'guardUsed', 'silenced', 'onceUsed'].every(
      (k) => typeof u[k] === 'boolean',
    )
  )
    return false;
  if (
    !list(u.attacked) ||
    !u.attacked.every((v) => text(v)) ||
    !list(u.equipment, 1) ||
    !u.equipment.every((v) => validKind(v) && definition(v).weapon !== undefined) ||
    !effects(u.effects)
  )
    return false;
  return (
    (u.group === undefined || text(u.group)) &&
    ['expiresAt', 'hookReadyAt', 'hookExpiresAt'].every((k) => u[k] === undefined || int(u[k]))
  );
}
export function validState(s: unknown): s is GameState {
  if (
    !object(s) ||
    s.version !== 2 ||
    !player(s.active) ||
    !int(s.ply, 1) ||
    !int(s.seed, 1, 4294967295) ||
    !int(s.rng, 1, 4294967295) ||
    !int(s.serial, 1)
  )
    return false;
  if (
    !['summon', 'play'].includes(s.phase) ||
    !int(s.summonSlots, -1, 1000) ||
    (s.winner !== undefined && !player(s.winner) && s.winner !== 'draw')
  )
    return false;
  if (
    ![s.turns, s.bases, s.baseEffects, s.heads, s.hands, s.bonus, s.deployRows].every(object) ||
    !list(s.units, 920) ||
    !list(s.pending, 2000)
  )
    return false;
  const ids = new Set<string>(),
    occupied = new Map<string, Unit[]>();
  for (const raw of s.units) {
    if (!validUnit(raw) || ids.has(raw.id)) return false;
    const u = raw;
    ids.add(u.id);
    for (const p of cells(u)) {
      if (!inside(p)) return false;
      const old = occupied.get(key(p)) ?? [];
      const transit =
        u.kind === 'u12' ||
        u.kind === 'u12p' ||
        old.some((v) => v.kind === 'u12' || v.kind === 'u12p');
      if (
        old.length &&
        !transit &&
        !(u.kind === 'u25' && u.size === 1 && old.every((v) => v.kind === 'u25' && v.size === 1))
      )
        return false;
      if (
        (equal(p, basePoint(1)) || equal(p, basePoint(2))) &&
        !(
          u.kind === 'u12' &&
          s.pending.some((r: any) => r.kind === 'bounce' && r.source?.id === u.id)
        )
      )
        return false;
      occupied.set(key(p), [...old, u]);
    }
  }
  for (const p of [1, 2] as Player[]) {
    if (
      !int(s.turns[p]) ||
      !num(s.bases[p], 0, 300) ||
      !int(s.heads[p]) ||
      !int(s.bonus[p]) ||
      !effects(s.baseEffects[p]) ||
      !list(s.deployRows[p], 13) ||
      !s.deployRows[p].every((r: any) => int(r, 1, 13)) ||
      !list(s.hands[p], 2000)
    )
      return false;
    for (const c of s.hands[p]) {
      if (
        !object(c) ||
        !text(c.id) ||
        ids.has(c.id) ||
        !validKind(c.kind) ||
        !int(c.drawnAt) ||
        !int(c.summonedPly, 1) ||
        (c.expiresAt !== undefined && !int(c.expiresAt, 1))
      )
        return false;
      ids.add(c.id);
      if (c.group !== undefined && !text(c.group)) return false;
    }
  }
  if (
    !s.pending.every(
      (r: any) =>
        object(r) &&
        ['death-shot', 'reflect', 'bounce', 'hut-spawn'].includes(r.kind) &&
        player(r.owner) &&
        validUnit(r.source, true) &&
        num(r.amount),
    )
  )
    return false;
  if (
    !list(s.deaths, 240) ||
    !s.deaths.every(
      (r: any) =>
        object(r) &&
        text(r.id) &&
        validKind(r.kind, true) &&
        player(r.owner) &&
        int(r.ply, 1) &&
        typeof r.revived === 'boolean',
    )
  )
    return false;
  if (
    !list(s.hazards) ||
    !s.hazards.every(
      (h: any) =>
        object(h) &&
        text(h.id) &&
        player(h.owner) &&
        ['row', 'column'].includes(h.axis) &&
        int(h.line, 1, h.axis === 'row' ? 13 : 9) &&
        int(h.due, 1),
    )
  )
    return false;
  if (
    !list(s.iceMarks) ||
    !s.iceMarks.every(
      (m: any) =>
        object(m) &&
        text(m.id) &&
        text(m.sourceId) &&
        player(m.owner) &&
        inside({ x: m.x, y: m.y }) &&
        int(m.due, 1),
    )
  )
    return false;
  if (
    !list(s.siphons) ||
    !s.siphons.every(
      (l: any) =>
        object(l) &&
        ['id', 'sourceId', 'fromId', 'toId'].every((k) => text(l[k])) &&
        player(l.owner),
    )
  )
    return false;
  if (
    !list(s.log, 180) ||
    !s.log.every((l: any) => typeof l === 'string' && l.length <= 2000) ||
    !list(s.events, 5000)
  )
    return false;
  return s.events.every(
    (e: any) =>
      object(e) &&
      text(e.id) &&
      [
        'spawn',
        'move',
        'attack',
        'damage',
        'heal',
        'death',
        'skill',
        'shield',
        'turn',
        'summon',
      ].includes(e.type) &&
      ['from', 'to'].every(
        (k) => e[k] === undefined || (object(e[k]) && num(e[k].x, 0, 10) && num(e[k].y, 0, 14)),
      ) &&
      (e.text === undefined || typeof e.text === 'string') &&
      (e.path === undefined ||
        (list(e.path, 200) && e.path.every((p: any) => object(p) && inside({ x: p.x, y: p.y })))),
  );
}
export function parseSession(text: string): Session {
  if (text.length > 24_000_000) throw new Error('存档超过24MB，请使用本游戏导出的存档。');
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new Error('存档不是有效JSON。');
  }
  if (object(value) && (value.format === 'haojie2-session-v1' || value.present?.version === 1))
    throw new Error(
      '这是旧版“豪杰棋局”存档。浩劫2.0的操作规则不同，不自动迁移；原文件没有被修改。',
    );
  if (
    !object(value) ||
    value.format !== 'haojie-session-v2' ||
    !validState(value.present) ||
    !list(value.past, LIMIT) ||
    !list(value.future, LIMIT) ||
    !value.past.every(validState) ||
    !value.future.every(validState)
  )
    throw new Error('存档结构不兼容或数据损坏，当前棋局未被替换。');
  return normalizeHornStorage(value as Session);
}
