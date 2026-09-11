import { enrichEvent } from './event-facts';
import { simulationRandom } from './random';
import { definition, isStored, SUMMON_POOL, ULTIMATE_POOL } from './catalog';
import type {
  Card,
  Command,
  Effect,
  GameEvent,
  GameState,
  Kind,
  Mode,
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
export function ensure(value: unknown, message: string): asserts value {
  if (!value) throw new RuleError(message);
}
export const faction = (p: Player) => (p === 1 ? '苍穹方' : '赤焰方');
export const now = (s: GameState, u: Unit) => s.ply + u.offset;
export const activeEffect = (s: GameState, e: Effect, u?: Unit) =>
  e.from <= (e.global || !u ? s.ply : now(s, u)) && e.until > (e.global || !u ? s.ply : now(s, u));
export const has = (s: GameState, u: Unit, type: Effect['type']) =>
  u.effects.some((e) => e.type === type && activeEffect(s, e, u));
export const allegiance = (s: GameState, u: Unit): Player | 0 =>
  has(s, u, 'freeze') ? 0 : u.owner;
export const passive = (s: GameState, u: Unit) => !u.silenced && !has(s, u, 'freeze');
export const hasWeapon = (u: Unit, k: Kind) => u.equipment.includes(k);
export const age = (s: GameState, u: Unit) => s.turns[u.owner] + u.offset / 2 - u.born;
export const isRunner = (u: Unit) => !u.silenced && (u.kind === 'u12' || u.kind === 'u12p');
export function emit(s: GameState, event: Omit<GameEvent, 'id'>, message?: string) {
  const snap = enrichEvent(s, { ...event, id: `e${s.serial++}` });
  if (event.from) snap.from = { x: event.from.x, y: event.from.y };
  if (event.to) snap.to = { x: event.to.x, y: event.to.y };
  if (event.path) snap.path = event.path.map((p) => ({ x: p.x, y: p.y }));
  s.events.push(snap);
  if (message) {
    s.log.push(`${s.ply} · ${message}`);
    if (s.log.length > 180) s.log.shift();
  }
}
export function random(s: GameState, boundaries: readonly number[] = [0, 1]) {
  const supplied = simulationRandom(s, boundaries);
  if (supplied !== undefined) return supplied;
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
    size: d.size ?? 1,
    born,
    offset: 0,
    deployedAt: 0,
    chargedOnDeploy: ['u1', 'u12', 'u12p'].includes(String(kind)),
    mode: 'none',
    operations: 0,
    shots: 0,
    moves: 0,
    attacked: [],
    bonusAttacks: 0,
    bonusSequence: false,
    weaponFirstUsed: false,
    charge: 0,
    readyCharge: 0,
    chargeType: d.move % 1 ? 'move' : kind === 21 ? 'skill' : 'attack',
    lastCharge: -1,
    upgrades: 0,
    kills: 0,
    attackBonus: 0,
    rangeBonus: 0,
    guardUsed: false,
    effects: [],
    equipment: [],
    silenced: false,
    freeUsed: -1,
    onceUsed: false,
  };
}
export function addUnit(s: GameState, kind: Kind, owner: Player, at: Point, group?: string): Unit {
  const u = template(kind, owner, s.turns[owner], at, `u${s.serial++}`);
  u.deployedAt = s.ply;
  if (group) u.group = group;
  if (['u1', 'u12', 'u12p'].includes(String(kind))) u.born--;
  s.units.push(u);
  emit(
    s,
    { type: 'spawn', to: at, unitId: u.id, owner, ultimate: definition(kind).tier !== 'normal' },
    `${faction(owner)}部署${definition(kind).name}`,
  );
  return u;
}
export function draw(s: GameState, owner: Player, count: number, ultimate = false): Card[] {
  const result: Card[] = [];
  for (let i = 0; i < count; i++) {
    const pool = ultimate ? ULTIMATE_POOL : SUMMON_POOL;
    let kind =
      pool[
        Math.floor(
          random(
            s,
            Array.from({ length: pool.length + 1 }, (_, i) => i / pool.length),
          ) * pool.length,
        )
      ];
    if (kind === 3 && random(s, [0, 1 / 3, 1]) >= 1 / 3) kind = '3p';
    if (kind === 'u12' && random(s, [0, 0.1, 1]) >= 0.1) kind = 'u12p';
    const group = kind === 'u25' ? `group${s.serial++}` : undefined;
    const d = definition(kind),
      limit = d.spell ?? d.weapon;
    for (let j = 0; j < (group ? 8 : 1); j++) {
      const c: Card = {
        id: `c${s.serial++}`,
        kind,
        drawnAt: s.turns[owner],
        summonedPly: s.ply,
        ...(limit !== undefined && limit >= 0 ? { expiresAt: s.turns[owner] + limit } : {}),
        ...(group ? { group } : {}),
      };
      s.hands[owner].push(c);
      result.push(c);
    }
    emit(
      s,
      { type: 'summon', owner, text: d.name, ultimate },
      `${ultimate ? '终极' : '普通'}召唤：${d.name}${group ? ' ×8' : ''}`,
    );
  }
  return result;
}
export function getStats(s: GameState, u: Unit): Stats {
  const d = definition(u.kind),
    enabled = !u.silenced;
  let attack = d.attack + u.attackBonus,
    range = d.range + u.rangeBonus,
    actions = d.actions === 1 / 3 ? 1 : d.actions,
    move = d.move;
  const frozen = has(s, u, 'freeze'),
    stunned = has(s, u, 'stun');
  const a = age(s, u);
  const sleeping = a <= 0 || (enabled && u.kind === 23 && (a < 2 || a % 2 !== 0));
  if (enabled && u.kind === '3p') {
    const n = s.units.filter((v) => {
      for (let dx = 0; dx < v.size; dx++)
        for (let dy = 0; dy < v.size; dy++)
          if (Math.max(Math.abs(v.x + dx - u.x), Math.abs(v.y + dy - u.y)) <= 1) return true;
      return false;
    }).length;
    attack = Math.max(0, 40 - 5 * n) + u.attackBonus;
    range = n + u.rangeBonus;
  }
  if (enabled && u.kind === 4 && u.charge >= 5) range++;
  if (enabled && u.kind === 'u2') attack += u.charge * 15;
  if (enabled && u.kind === 'u6' && hasWeapon(u, 'u5')) attack = 10 + u.attackBonus;
  if (hasWeapon(u, 'u5')) range++;
  if (hasWeapon(u, 'u11')) attack += 5;
  if (
    enabled &&
    u.kind === 12 &&
    s.units.some(
      (v) =>
        allegiance(s, v) !== u.owner &&
        v.id !== u.id &&
        Array.from({ length: u.size * u.size }, (_, i) => ({
          x: u.x + (i % u.size),
          y: u.y + Math.floor(i / u.size),
        })).some((p) =>
          Array.from({ length: v.size * v.size }, (_, i) => ({
            x: v.x + (i % v.size),
            y: v.y + Math.floor(i / v.size),
          })).some((q) => Math.abs(p.x - q.x) + Math.abs(p.y - q.y) === 1),
        ),
    )
  ) {
    actions--;
    move--;
  }
  attack += u.effects
    .filter((e) => e.type === 'attack' && activeEffect(s, e, u))
    .reduce((a, e) => a + (e.amount ?? 0), 0);
  if (has(s, u, 'inner-fire')) attack = u.hp;
  const operationLimit = enabled && u.kind === 'u27' && a === 1 ? 2 : 1;
  const locked = sleeping || frozen || stunned;
  const operationsLeft = locked ? 0 : Math.max(0, operationLimit - u.operations);
  const availableAttack = locked
    ? 0
    : u.mode === 'attack'
      ? Math.max(0, actions - u.shots)
      : u.mode === 'none' && (operationsLeft > 0 || u.bonusAttacks > 0)
        ? actions
        : 0;
  return {
    attack: Math.max(0, attack),
    range: Math.max(0, range),
    actions: Math.max(0, actions),
    remaining: availableAttack,
    move,
    mode: u.mode,
    operationLimit,
    operationsLeft,
    frozen,
    stunned,
    sleeping,
  };
}
export function findUnit(s: GameState, id?: string): Unit {
  const u = s.units.find((u) => u.id === id);
  ensure(u, '请选择仍在场上的随从。');
  return u;
}
export const asTarget = (u: Unit): Target => ({
  id: u.id,
  owner: u.owner,
  x: u.x,
  y: u.y,
  unit: u,
});
export function actor(s: GameState, id?: string) {
  const u = findUnit(s, id),
    stats = getStats(s, u);
  ensure(s.phase === 'play', '请先完成回合开始阶段。');
  ensure(u.owner === s.active, '不是该随从的回合。');
  ensure(
    !stats.sleeping && !stats.frozen && !stats.stunned,
    '该随从正在疲劳、休整、冰冻或眩晕中。',
  );
  return u;
}
export function chooseMode(s: GameState, u: Unit, mode: Mode) {
  const stats = getStats(s, u);
  ensure(
    u.mode === 'none' || u.mode === mode,
    '本回合已选择另一操作模式，剩余攻击不能换成移动或技能。',
  );
  if (u.mode === 'none') {
    if (mode === 'attack' && stats.operationsLeft <= 0 && u.bonusAttacks > 0) {
      u.bonusAttacks--;
      u.bonusSequence = true;
    } else ensure(stats.operationsLeft > 0, '本回合操作已用完。');
    u.mode = mode;
    u.shots = 0;
    u.moves = 0;
  }
  if (mode === 'attack') ensure(getStats(s, u).remaining > 0, '本次攻击操作次数已用完。');
}
export function finishOperation(u: Unit) {
  if (!u.bonusSequence) u.operations++;
  u.bonusSequence = false;
  u.mode = 'none';
  u.shots = 0;
  u.moves = 0;
}
export function point(x?: number, y?: number): Point {
  ensure(Number.isInteger(x) && Number.isInteger(y), '请选择棋盘格。');
  return { x: x!, y: y! };
}
export function resetUnit(s: GameState, u: Unit) {
  u.operations = 0;
  u.mode = 'none';
  u.shots = 0;
  u.moves = 0;
  u.attacked = [];
  u.bonusAttacks = 0;
  u.bonusSequence = false;
  u.weaponFirstUsed = false;
  u.readyCharge = u.charge;
  u.effects = u.effects.filter((e) => e.until > (e.global ? s.ply : now(s, u)));
}
export function addEffect(
  s: GameState,
  u: Unit,
  type: Effect['type'],
  owner: Player,
  delay: number,
  duration: number,
  amount?: number,
  sourceId?: string,
) {
  const base = now(s, u);
  u.effects.push({
    type,
    owner,
    from: base + delay,
    until: base + delay + duration,
    ...(amount === undefined ? {} : { amount }),
    ...(sourceId === undefined ? {} : { sourceId }),
  });
}
export function storageRemaining(s: GameState, c: Card) {
  return c.expiresAt === undefined ? null : c.expiresAt - s.turns[s.active];
}
