import { applyCommand } from './game';
import { definition } from './catalog';
import { cells, inside, basePoint, equal, key } from './geometry';
import type { Command, GameState, Player, Unit } from './types';

export interface Session {
  format: 'haojie2-session-v1';
  present: GameState;
  past: GameState[];
  future: GameState[];
}
const LIMIT = 60;
export const createSession = (present: GameState): Session => ({
  format: 'haojie2-session-v1',
  present,
  past: [],
  future: [],
});
export function dispatch(session: Session, command: Command): Session {
  return {
    ...session,
    present: applyCommand(session.present, command),
    past: [...session.past, session.present].slice(-LIMIT),
    future: [],
  };
}
export function undo(session: Session): Session {
  if (!session.past.length) return session;
  return {
    ...session,
    present: session.past[session.past.length - 1],
    past: session.past.slice(0, -1),
    future: [session.present, ...session.future].slice(0, LIMIT),
  };
}
export function redo(session: Session): Session {
  if (!session.future.length) return session;
  return {
    ...session,
    present: session.future[0],
    past: [...session.past, session.present].slice(-LIMIT),
    future: session.future.slice(1),
  };
}
const record = (v: unknown): v is Record<string, any> =>
  !!v && typeof v === 'object' && !Array.isArray(v);
const integer = (v: unknown, min = 0, max = Number.MAX_SAFE_INTEGER) =>
  typeof v === 'number' && Number.isSafeInteger(v) && v >= min && v <= max;
const player = (v: unknown) => v === 1 || v === 2;
const effects = (v: unknown): boolean =>
  Array.isArray(v) &&
  v.length <= 500 &&
  v.every(
    (e) =>
      record(e) &&
      ['attack', 'immune', 'execute', 'convert', 'mark'].includes(e.type) &&
      integer(e.from) &&
      integer(e.until) &&
      e.until > e.from &&
      player(e.owner) &&
      (e.amount === undefined || integer(e.amount)),
  );
function validUnit(u: any, dead = false): u is Unit {
  if (!record(u) || typeof u.id !== 'string' || !player(u.owner) || !inside(u as Unit))
    return false;
  try {
    if (definition(u.kind).spell) return false;
  } catch {
    return false;
  }
  if (
    !integer(u.hp, dead ? 0 : 1) ||
    !integer(u.maxHp, 1) ||
    u.hp > u.maxHp ||
    !integer(u.born) ||
    !integer(u.spent)
  )
    return false;
  if (
    ![u.charge, u.upgrades, u.kills, u.attackBonus, u.rangeBonus].every((v) => integer(v)) ||
    !integer(u.lastCharge, -1)
  )
    return false;
  return (
    typeof u.fired === 'boolean' &&
    typeof u.guardUsed === 'boolean' &&
    Array.isArray(u.attacked) &&
    u.attacked.every((id: unknown) => typeof id === 'string') &&
    effects(u.effects) &&
    (u.expiresAt === undefined || integer(u.expiresAt, 1))
  );
}
/** Structural validation for local save files, not a network authentication boundary. */
function validState(s: unknown): s is GameState {
  if (
    !record(s) ||
    s.version !== 1 ||
    !player(s.active) ||
    !integer(s.ply, 1) ||
    !integer(s.seed, 1, 4294967295) ||
    !integer(s.rng, 1, 4294967295) ||
    !integer(s.serial, 1)
  )
    return false;
  if (s.winner !== undefined && !player(s.winner) && s.winner !== 'draw') return false;
  if (![s.turns, s.bases, s.baseEffects, s.hands, s.bonus, s.deployRows].every(record))
    return false;
  const ids = new Set<string>(),
    occupied = new Set<string>();
  if (!Array.isArray(s.units) || s.units.length > 115) return false;
  for (const u of s.units) {
    if (!validUnit(u) || ids.has(u.id)) return false;
    ids.add(u.id);
    for (const p of cells(u)) {
      if (!inside(p) || equal(p, basePoint(1)) || equal(p, basePoint(2)) || occupied.has(key(p)))
        return false;
      occupied.add(key(p));
    }
  }
  for (const p of [1, 2] as Player[]) {
    if (
      !integer(s.turns[p]) ||
      !integer(s.bases[p], 0, 300) ||
      !integer(s.bonus[p]) ||
      !effects(s.baseEffects[p])
    )
      return false;
    if (!Array.isArray(s.deployRows[p]) || s.deployRows[p].some((r: unknown) => !integer(r, 1, 13)))
      return false;
    if (!Array.isArray(s.hands[p]) || s.hands[p].length > 1000) return false;
    for (const c of s.hands[p]) {
      if (!record(c) || typeof c.id !== 'string' || ids.has(c.id) || !integer(c.drawnAt))
        return false;
      ids.add(c.id);
      try {
        if (!definition(c.kind)) return false;
      } catch {
        return false;
      }
      if (c.expiresAt !== undefined && !integer(c.expiresAt, 1)) return false;
    }
  }
  if (
    !Array.isArray(s.pending) ||
    !s.pending.every(
      (r: any) =>
        record(r) &&
        ['death-shot', 'reflect'].includes(r.kind) &&
        player(r.owner) &&
        validUnit(r.source, true) &&
        integer(r.amount),
    )
  )
    return false;
  return (
    Array.isArray(s.log) &&
    s.log.length <= 160 &&
    s.log.every((v: unknown) => typeof v === 'string') &&
    Array.isArray(s.events)
  );
}
export function parseSession(text: string): Session {
  if (text.length > 12_000_000) throw new Error('存档过大，请使用本游戏导出的JSON存档。');
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new Error('存档不是有效的JSON文件。');
  }
  if (
    !record(value) ||
    value.format !== 'haojie2-session-v1' ||
    !validState(value.present) ||
    !Array.isArray(value.past) ||
    !Array.isArray(value.future) ||
    value.past.length > LIMIT ||
    value.future.length > LIMIT ||
    !value.past.every(validState) ||
    !value.future.every(validState)
  )
    throw new Error('存档格式不兼容或数据损坏；当前对局未被替换。');
  return value as Session;
}
