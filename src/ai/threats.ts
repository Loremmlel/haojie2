import {
  attackPath,
  basePoint,
  cells,
  distance,
  other,
  targets,
  topTarget,
  canPlace,
  neighbors,
} from '../engine/geometry';
import { activeEffect, asTarget, has, passive } from '../engine/state';
import type { GameState, Player, Target, Unit } from '../engine/types';
import { actionWindow, hitDistance, isFrontHit, statsFor } from './spatial';
/** Damage/utility estimate, not a replacement for exact distribution() at search nodes. */
export function readyAttack(s: GameState, u: Unit): boolean {
  const st = statsFor(s, u);
  return (
    st.remaining > 0 &&
    !st.sleeping &&
    !st.frozen &&
    !st.stunned &&
    !(passive(s, u) && ((u.kind === 4 && u.readyCharge < 2) || u.kind === 'firelord'))
  );
}
export function hitPackets(
  s: GameState,
  u: Unit,
  t: Target,
): { damage: number; probability: number }[] {
  const st = statsFor(s, u);
  if (t.unit && has(s, t.unit, 'immune')) return [{ damage: 0, probability: 1 }];
  let packets = [{ damage: st.attack, probability: 1 }];
  if (passive(s, u)) {
    if (u.kind === 1)
      packets = [
        { damage: st.attack, probability: 2 / 3 },
        { damage: st.attack + 20, probability: 1 / 4 },
        { damage: st.attack + 60, probability: 1 / 12 },
      ];
    if (u.kind === 'u1')
      packets = [
        { damage: st.attack, probability: 7 / 15 },
        { damage: st.attack * 2, probability: 1 / 3 },
        { damage: 100, probability: 1 / 5 },
      ];
    if (u.kind === 'u8') {
      const p = Math.min(1, 0.2 + 0.2 * u.kills);
      packets = [
        { damage: st.attack, probability: 1 - p },
        { damage: st.attack * 2, probability: p },
      ];
    }
    if (u.kind === 'u27' && !t.unit) packets = [{ damage: 10, probability: 1 }];
    if (u.kind === 10)
      packets = [
        {
          damage:
            st.attack > 0
              ? st.attack
              : (t.unit ? t.unit.hp === t.unit.maxHp : s.bases[t.owner] === 300)
                ? 5
                : 0,
          probability: 1,
        },
      ];
  }
  if (t.unit && passive(s, t.unit)) {
    if (t.unit.kind === 24 && isFrontHit(s, u, t, t.id))
      packets = packets.map((p) => ({ ...p, damage: Math.min(10, p.damage) }));
    if (t.unit.kind === 'u18')
      packets = packets.map((p) => ({ ...p, damage: p.damage <= 10 ? 0 : p.damage }));
  }
  return packets;
}
export function attackPressure(s: GameState, u: Unit, t: Target, ignoreId = ''): number {
  if (!readyAttack(s, u) || !Number.isFinite(hitDistance(s, u, t, ignoreId))) return 0;
  if (u.kind === 9 && passive(s, u) && u.attacked.includes(t.id)) return 0;
  if (t.unit && has(s, t.unit, 'immune')) return 0;
  const packets = hitPackets(s, u, t);
  let amount = packets.reduce((v, p) => v + p.damage * p.probability, 0);
  const one = passive(s, u) && (u.kind === 9 || u.kind === 4 || u.kind === 10);
  const count = one ? 1 : statsFor(s, u).remaining;
  amount *= count;
  if (t.unit) {
    if (passive(s, u) && u.kind === 'u4') amount += 12;
    if (u.equipment.includes('u5')) amount += 15;
    if (passive(s, u) && u.kind === 'u6' && !u.equipment.includes('u5')) amount += 15;
  }
  return amount;
}
export function incoming(s: GameState, target: Unit, nextFullTurn = false): number {
  const view = actionWindow(s, other(target.owner), nextFullTurn),
    t = asTarget({ ...target });
  let total = 0;
  for (const v of view.units)
    if (v.owner !== target.owner && v.id !== target.id)
      total += attackPressure(view, v, t, target.id);
  return total;
}
export function baseThreat(s: GameState, defender: Player): number {
  const view = actionWindow(s, other(defender));
  const target = { id: `base-${defender}`, owner: defender, ...basePoint(defender) };
  return view.units
    .filter((u) => u.owner !== defender)
    .reduce((v, u) => v + attackPressure(view, u, target), 0);
}
/** Marks must be cashed in before their real global expiry, by another non-catapult's hit. */
export function markFollowUp(s: GameState, target: Target, owner: Player, until: number): number {
  const view = actionWindow(s, owner);
  if (view.ply >= until) return 0;
  let best = 0;
  for (const u of view.units) {
    if (
      u.owner !== owner ||
      u.kind === 10 ||
      !readyAttack(view, u) ||
      (u.kind === 9 && !u.silenced && u.attacked.includes(target.id)) ||
      !Number.isFinite(hitDistance(view, u, target))
    )
      continue;
    const hp = target.unit?.hp ?? view.bases[target.owner];
    const guarded =
      target.unit &&
      !target.unit.guardUsed &&
      view.units.some(
        (v) =>
          v.kind === 3 &&
          v.owner === target.owner &&
          passive(view, v) &&
          attackPath(view, v, target, statsFor(view, v).range),
      );
    const packets = hitPackets(view, u, target);
    let probability = packets.reduce(
      (n, p) => n + (p.damage < hp || guarded ? p.probability : 0),
      0,
    );
    if (!probability) continue; // An already lethal follow-up cannot redeem five extra damage.
    const hit = packets.reduce((n, p) => n + p.probability * p.damage, 0);
    // A legal follow-up is not a commitment: a doomed shooter may sensibly retreat instead.
    if (
      target.unit &&
      hit + 5 < target.unit.hp &&
      incoming(view, u) >= u.hp &&
      statsFor(view, u).move >= 1 &&
      neighbors(u).some((p) => canPlace(view, u, p) && incoming(view, { ...u, ...p }) < u.hp)
    )
      probability *= 0.25;
    best = Math.max(best, probability);
  }
  return best;
}
export interface PayloadAnalysis {
  carrierId: string;
  type: 'execute' | 'convert';
  value: number;
  window: number;
  survival: number;
  targets: { id: string; probability: number; value: number; reason: string }[];
}
/** Exposes the same opportunity calculation to optional CLI traces. Never reads a future roll. */
export function analyzePayload(
  s: GameState,
  u: Unit,
  type: 'execute' | 'convert',
  valueOf: (u: Unit) => number,
): PayloadAnalysis {
  const result: PayloadAnalysis = {
    carrierId: u.id,
    type,
    value: 0,
    window: s.ply,
    survival: 0,
    targets: [],
  };
  const effect = u.effects.find((e) => e.type === type);
  if (!effect) return result;
  const view = actionWindow(s, u.owner, effect.from > s.ply + u.offset);
  result.window = view.ply;
  const carrier = view.units.find((v) => v.id === u.id);
  if (!carrier || !activeEffect(view, effect, carrier) || !readyAttack(view, carrier))
    return result;
  for (const victim of view.units) {
    if (victim.owner === u.owner || !topTarget(view, asTarget(victim))) continue;
    let reason = '可兑现';
    if (has(view, victim, 'immune')) reason = '金身保护';
    else if (!Number.isFinite(hitDistance(view, carrier, asTarget(victim))))
      reason = '无合法攻击路径';
    else if (
      view.units.some(
        (v) =>
          v.kind === 'u15' &&
          v.owner === victim.owner &&
          passive(view, v) &&
          attackPath(view, v, asTarget(victim), statsFor(view, v).range),
      )
    )
      reason = '免疫塔保护';
    const probability =
      reason !== '可兑现'
        ? 0
        : type === 'execute'
          ? 1
          : hitPackets(view, carrier, asTarget(victim)).reduce(
              (n, p) => n + (p.damage > 0 && p.damage < victim.hp ? p.probability : 0),
              0,
            );
    if (!probability && reason === '可兑现') reason = '没有造成正伤害后存活的分支';
    result.targets.push({
      id: victim.id,
      probability,
      value: valueOf(victim) * probability * (type === 'convert' ? 1.3 : 0.85),
      reason,
    });
  }
  result.survival = Math.max(0.12, 1 - 0.75 * Math.min(1, incoming(s, u) / Math.max(1, u.hp)));
  result.value =
    Math.max(0, ...result.targets.map((t) => t.value)) *
    (effect.from > s.ply + u.offset ? 0.55 : 0.8) *
    result.survival;
  result.targets.sort((a, b) => b.value - a.value);
  return result;
}
export function payloadOpportunity(
  s: GameState,
  u: Unit,
  type: 'execute' | 'convert',
  valueOf: (u: Unit) => number,
): number {
  return analyzePayload(s, u, type, valueOf).value;
}
