import { definition, isStored } from '../engine/catalog';
import { attackPath, basePoint, cells, distance, other } from '../engine/geometry';
import { allegiance, getStats, passive } from '../engine/state';
import type { Card, GameState, Player, Point, Unit } from '../engine/types';
const passives: Partial<Record<Unit['kind'], number>> = {
  2: 16,
  3: 40,
  4: 10,
  6: 12,
  7: 18,
  10: 12,
  11: 15,
  12: 12,
  14: 6,
  16: 8,
  19: 12,
  20: 12,
  21: 12,
  23: 4,
  24: 14,
  26: 14,
  u3: 26,
  u4: 30,
  u6: 32,
  u8: 28,
  u13: 18,
  u14: 38,
  u15: 30,
  u18: 28,
  u19: 20,
  u20: 25,
  u21: 24,
  u22: 20,
  u23: 18,
  u24: 17,
  firelord: 65,
};
export function unitValue(s: GameState, u: Unit): number {
  const st = getStats(s, u),
    ratio = u.hp / u.maxHp;
  let value =
    14 +
    u.hp * 0.38 +
    u.maxHp * 0.12 +
    st.attack * Math.sqrt(st.actions) * (0.65 + 0.3 * ratio) +
    Math.min(8, st.range) * 2;
  if (passive(s, u)) value += passives[u.kind] ?? 0;
  if (u.kind === 'grave' || u.kind === 'wall') value = 4 + u.hp * 0.16;
  if (u.kind === 'u25') value -= 7; // Eight bodies are useful, not eight kill rewards.
  if (u.kind === 'u12' || u.kind === 'u12p') value += 30;
  if (u.expiresAt !== undefined) value *= 0.35;
  if (st.frozen) value *= 0.5;
  else if (st.sleeping || st.stunned) value *= 0.82;
  value += u.charge * (u.kind === 4 ? 10 : u.kind === 'u6' ? 20 : u.chargeType === 'move' ? 8 : 7);
  value += u.equipment.includes('u5')
    ? 20
    : u.equipment.includes('u16')
      ? 16
      : u.equipment.includes('u28')
        ? 22
        : 0;
  for (const e of u.effects) {
    if (e.type === 'immune') value += 12;
    if (e.type === 'execute' || e.type === 'convert') value += st.actions ? 24 : 2;
    if (e.type === 'attack') value += Math.min(15, e.amount ?? 0) * 0.8;
    if (e.type === 'burn') value -= Math.min(u.hp, 12) * 0.5;
    if (e.type === 'mark') value -= 3;
  }
  return Math.max(1, value);
}
export function cardValue(s: GameState, c: Card, owner: Player): number {
  const d = definition(c.kind);
  if (!isStored(d))
    return (
      (18 + d.health * 0.5 + d.attack * Math.sqrt(d.actions) * 0.65 + (passives[c.kind] ?? 0)) *
      0.78
    );
  const remaining = (c.expiresAt ?? s.turns[owner] + 20) - s.turns[owner];
  const values: Partial<Record<Unit['kind'], number>> = {
    8: 22,
    17: 15,
    18: 28,
    22: 35,
    25: 38,
    u5: 34,
    u9: 44,
    u11: 30,
    u16: 30,
    u17: 30,
    u26: 34,
    u28: 36,
  };
  return (values[c.kind] ?? 20) * Math.min(1, 0.65 + Math.max(0, remaining - 1) * 0.13);
}
/** Position estimate complements exact command simulation; it is deliberately not a second rules engine. */
export function placementValue(s: GameState, u: Unit, p: Point): number {
  const st = getStats(s, u),
    enemyBase = basePoint(other(u.owner)),
    home = basePoint(u.owner);
  let nearest = distance(p, enemyBase),
    danger = 0,
    help = 0;
  for (const v of s.units) {
    if (v.id === u.id) continue;
    const d = Math.max(0, Math.min(...cells(v).map((q) => distance(p, q))) - (u.size - 1));
    const vs = getStats(s, v);
    if (allegiance(s, v) !== u.owner) {
      nearest = Math.min(nearest, d);
      if (d <= Math.min(vs.range, 7) && !vs.frozen && !vs.stunned)
        danger += Math.max(2, vs.attack) * Math.max(1, vs.actions) * 0.15;
    } else if (d <= st.range) help += Math.min(8, (v.maxHp - v.hp) * 0.1 + 2);
  }
  const support = [2, 3, 6, 'u3', 'u13', 'u14', 'u15', 'u19', 'u21', 'u22'].includes(u.kind);
  const reach = Math.min(st.range, 6),
    engage = Math.max(0, nearest - reach);
  let score = -(engage * 2.8 + distance(p, enemyBase) * 0.3);
  if (st.attack > 0 || ['u6', 'u12', 'u12p', 'firelord'].includes(String(u.kind)))
    score += Math.min(8, reach - nearest + 2) * 1.5;
  if (support) score += help - Math.max(0, 2 - nearest) * 5;
  score -= Math.min(45, danger) * (1 - u.hp / (u.hp + 55));
  // Defenders converge on actual intruders, rather than every unit sitting on its base.
  const intruder = s.units.some((v) => v.owner !== u.owner && distance(v, home) <= 5);
  if (intruder) score += Math.max(0, 7 - distance(p, home)) * 1.6;
  return score;
}
/** Project only availability for threat estimation. Actual future combat uses engine transitions. */
function futureAttacker(s: GameState, u: Unit): [GameState, Unit] {
  if (u.owner === s.active) return [s, u];
  return [
    { ...s, ply: s.ply + 1, turns: { ...s.turns, [u.owner]: s.turns[u.owner] + 1 } },
    { ...u, mode: 'none', operations: 0, shots: 0, readyCharge: u.charge, attacked: [] },
  ];
}
export function baseThreat(s: GameState, defender: Player): number {
  const target = { id: `base-${defender}`, owner: defender, ...basePoint(defender) };
  let threat = 0;
  for (const raw of s.units) {
    if (raw.owner === defender) continue;
    const [view, u] = futureAttacker(s, raw),
      st = getStats(view, u);
    if (
      !st.remaining ||
      st.frozen ||
      st.stunned ||
      st.sleeping ||
      (u.kind === 4 && !u.silenced && u.readyCharge < 2)
    )
      continue;
    if (Math.min(...cells(u).map((p) => distance(p, target))) > st.range) continue;
    if (!attackPath(view, u, target, st.range)) continue;
    let damage = u.kind === 'u27' ? 10 : st.attack;
    if (u.kind === 1 && !u.silenced) damage += 10;
    if (u.kind === 10 && !u.silenced) damage = s.bases[defender] === 300 ? 5 : 0;
    if (u.kind === 9 && !u.silenced) threat += u.attacked.includes(target.id) ? 0 : damage;
    else threat += damage * st.remaining;
  }
  return threat;
}
export function evaluate(s: GameState, side: Player): number {
  if (s.winner) return s.winner === 'draw' ? 0 : s.winner === side ? 100000 : -100000;
  let score = 0;
  for (const p of [1, 2] as Player[]) {
    const sign = p === side ? 1 : -1,
      hp = s.bases[p];
    let value =
      hp * 2.8 + Math.min(8, s.heads[p]) * 21 + Math.max(0, s.heads[p] - 8) * 12 + s.bonus[p] * 28;
    value += s.hands[p].reduce((sum, c) => sum + cardValue(s, c, p), 0);
    value += s.deployRows[p].filter((y) => (p === 1 ? y > 8 : y < 6)).length * 9;
    const threat = baseThreat(s, p);
    value -= threat * 0.65;
    if (threat >= hp) value -= 650 + (threat - hp) * 3;
    else if (threat > hp * 0.6) value -= 90;
    score += sign * value;
  }
  for (const u of s.units) {
    const sign = u.owner === side ? 1 : -1;
    let value = unitValue(s, u) + placementValue(s, u, u);
    const st = getStats(s, u);
    if (u.owner === s.active && !st.sleeping && !st.frozen)
      value += st.operationsLeft * 2 + st.remaining * 0.4;
    if (u.group) {
      const stack = s.units.filter((v) => v.group === u.group && v.x === u.x && v.y === u.y).length;
      value -= (stack - 1) * 0.7;
    }
    for (const h of s.hazards)
      if (cells(u).some((p) => (h.axis === 'row' ? p.y === h.line : p.x === h.line)))
        value -= Math.min(u.hp, 20) * 0.5;
    for (const m of s.iceMarks)
      if (m.owner !== u.owner && cells(u).some((p) => p.x === m.x && p.y === m.y)) value -= 20;
    score += sign * value;
  }
  // Free persistent siphons and delayed effects have value before their damage arrives.
  for (const l of s.siphons) {
    const sideOf = (id: string) =>
      id.startsWith('base-') ? Number(id.slice(5)) : s.units.find((u) => u.id === id)?.owner;
    const from = sideOf(l.fromId),
      to = sideOf(l.toId);
    if (from) score += (from === side ? -1 : 1) * 18;
    if (to) {
      const u = s.units.find((u) => u.id === l.toId),
        missing = u ? u.maxHp - u.hp : 300 - s.bases[to as Player];
      score += (to === side ? 1 : -1) * Math.min(missing, 20);
    }
  }
  return score;
}
