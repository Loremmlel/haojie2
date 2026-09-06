import { definition, SUMMON_POOL } from './catalog';
import { adjacent, cells, other, attackPath, targetAt } from './geometry';
import type {
  Effect,
  GameEvent,
  GameState,
  Kind,
  Player,
  Point,
  Stats,
  Target,
  Unit,
} from './types';

export class RuleError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'RuleError';
  }
}
export function ensure(condition: unknown, message: string): asserts condition {
  if (!condition) throw new RuleError(message);
}
export const faction = (p: Player) => (p === 1 ? '苍穹方' : '赤焰方');
export const activeEffect = (s: GameState, e: Effect) => e.from <= s.ply && e.until > s.ply;
export function emit(s: GameState, event: Omit<GameEvent, 'id'>, message?: string): void {
  const snapshot = { ...event, id: `e${s.serial++}` };
  if (event.from) snapshot.from = { x: event.from.x, y: event.from.y };
  if (event.to) snapshot.to = { x: event.to.x, y: event.to.y };
  if (event.type === 'attack' && event.from && 'kind' in event.from && event.to) {
    const target = targetAt(s, event.to);
    const route = target && attackPath(s, event.from as Unit, target, 117);
    if (route) snapshot.path = route.map((p) => ({ x: p.x, y: p.y }));
  }
  s.events.push(snapshot);
  if (message) {
    s.log.push(`${s.ply} · ${message}`);
    if (s.log.length > 160) s.log.shift();
  }
}
/** Xorshift32 is part of the saved state. Undo restores randomness, too. */
export function random(s: GameState): number {
  let x = s.rng;
  x ^= x << 13;
  x ^= x >>> 17;
  x ^= x << 5;
  s.rng = x >>> 0;
  return s.rng / 4294967296;
}
export function template(kind: Kind, owner: Player, born: number, at: Point, id = 'preview'): Unit {
  const d = definition(kind);
  return {
    id,
    kind,
    owner,
    x: at.x,
    y: at.y,
    hp: d.health,
    maxHp: d.health,
    born,
    spent: 0,
    attacked: [],
    charge: 0,
    lastCharge: -1,
    fired: false,
    upgrades: 0,
    kills: 0,
    attackBonus: 0,
    rangeBonus: 0,
    guardUsed: false,
    effects: [],
  };
}
export function addUnit(s: GameState, kind: Kind, owner: Player, at: Point): Unit {
  const u = template(kind, owner, s.turns[owner], at, `u${s.serial++}`);
  s.units.push(u);
  emit(
    s,
    { type: 'spawn', to: at, unitId: u.id, owner },
    `${faction(owner)}部署了${definition(kind).name}`,
  );
  return u;
}
export function draw(s: GameState, owner: Player, count: number): void {
  for (let i = 0; i < count; i++) {
    let kind: Kind = SUMMON_POOL[Math.floor(random(s) * SUMMON_POOL.length)];
    if (kind === 3 && random(s) >= 1 / 3) kind = '3p';
    const d = definition(kind);
    s.hands[owner].push({
      id: `c${s.serial++}`,
      kind,
      drawnAt: s.turns[owner],
      ...(d.spell ? { expiresAt: s.turns[owner] + d.spell } : {}),
    });
    emit(s, { type: 'skill', owner, text: d.name }, `${faction(owner)}召唤：${d.name}`);
  }
}
export function getStats(s: GameState, u: Unit): Stats {
  const d = definition(u.kind);
  let attack = d.attack + u.attackBonus,
    range = d.range + u.rangeBonus;
  let actions = u.kind === 4 ? 1 : d.actions,
    move = d.move;
  const age = s.turns[u.owner] - u.born;
  const sleeping = age <= 0 || (u.kind === 23 && (age < 2 || age % 2 !== 0));
  if (u.kind === '3p') {
    const n = s.units.filter((v) =>
      cells(v).some((p) => Math.max(Math.abs(p.x - u.x), Math.abs(p.y - u.y)) <= 1),
    ).length;
    attack = Math.max(0, 40 - 5 * n) + u.attackBonus;
    range = n + u.rangeBonus;
  }
  if (u.kind === 4 && u.charge >= 5) range++;
  if (u.kind === 5) move = u.charge > 0 ? 1 : 0;
  if (u.kind === 12 && s.units.some((v) => v.owner !== u.owner && adjacent(u, v))) {
    actions--;
    move--;
  }
  attack += u.effects
    .filter((e) => e.type === 'attack' && activeEffect(s, e))
    .reduce((sum, e) => sum + (e.amount ?? 0), 0);
  actions = sleeping ? 0 : Math.max(0, actions);
  return {
    attack: Math.max(0, attack),
    range: Math.max(0, range),
    actions,
    remaining: Math.max(0, actions - u.spent),
    move,
    sleeping,
  };
}
export function findUnit(s: GameState, id?: string): Unit {
  const u = s.units.find((v) => v.id === id);
  ensure(u, '请选择仍在场上的随从。');
  return u;
}
export function actor(s: GameState, id: string): Unit {
  const u = findUnit(s, id);
  ensure(u.owner === s.active, '现在不是这枚棋子所属玩家的回合。');
  ensure(getStats(s, u).remaining > 0, '这枚棋子本回合不能继续行动。');
  return u;
}
export function point(x?: number, y?: number): Point {
  ensure(Number.isInteger(x) && Number.isInteger(y), '请选择棋盘格。');
  return { x: x!, y: y! };
}
export function asTarget(u: Unit): Target {
  return { id: u.id, owner: u.owner, x: u.x, y: u.y, unit: u };
}
export function refreshDeployment(s: GameState, owner: Player): void {
  s.deployRows[owner] = [];
  for (let y = 1; y <= 13; y++) {
    const count = (p: Player) =>
      s.units.filter((u) => u.owner === p && cells(u).some((c) => c.y === y)).length;
    if ((owner === 1 ? y <= 8 : y >= 6) || count(owner) - count(other(owner)) >= 2)
      s.deployRows[owner].push(y);
  }
}
