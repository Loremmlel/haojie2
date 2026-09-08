import { definition, isStored } from '../engine/catalog';
import { unitActions, cardActions, reactionAction, actionError } from '../engine/options';
import type { ActionSpec, SelectionStep } from '../engine/options';
import {
  ALL_CELLS,
  basePoint,
  attackPath,
  canPlace,
  cells,
  distance,
  equal,
  movementPath,
  neighbors,
  occupants,
  targets,
  topTarget,
} from '../engine/geometry';
import { allegiance, asTarget, getStats, isRunner, template } from '../engine/state';
import type { Command, GameState, Point, Target, Unit } from '../engine/types';
import { DIFFICULTIES } from './difficulty';
import type { Difficulty } from './types';
import { placementValue, unitValue, materialValue } from './evaluate';
import {
  markFollowUp,
  payloadOpportunity,
  hitPackets,
  readyAttack,
  attackPressure,
} from './threats';
import { has, now, passive } from '../engine/state';
import { decisionOwner } from './observation';
import { hitDistance } from './spatial';
export interface CandidateGroup {
  family: string;
  commands: Command[];
  keep: number;
}
const releaseCache = new WeakMap<GameState, Map<string, number>>();
/** Removing a cheap screen may unlock a loaded ally's otherwise blocked base shot. */
function laneRelease(s: GameState, blocker: Unit): number {
  let cache = releaseCache.get(s);
  if (!cache) {
    cache = new Map();
    releaseCache.set(s, cache);
  }
  const old = cache.get(blocker.id);
  if (old !== undefined) return old;
  const owner = decisionOwner(s),
    defender = owner === 1 ? 2 : 1;
  const target = { id: `base-${defender}`, owner: defender as 1 | 2, ...basePoint(defender) };
  let value = 0;
  for (const ally of s.units)
    if (
      ally.owner === owner &&
      readyAttack(s, ally) &&
      !Number.isFinite(hitDistance(s, ally, target)) &&
      Number.isFinite(hitDistance(s, ally, target, blocker.id))
    ) {
      const damage = attackPressure(s, ally, target, blocker.id);
      value = Math.max(value, damage >= s.bases[defender] ? 200 : Math.min(80, damage * 0.8));
    }
  cache.set(blocker.id, value);
  return value;
}
function targetRank(s: GameState, a: ActionSpec, c: Command, t: Target): number {
  const owner = decisionOwner(s),
    friend = (t.unit ? allegiance(s, t.unit) : t.owner) === owner;
  const caster =
    s.units.find((u) => u.id === c.unitId) ??
    (c.type === 'react' ? s.pending[0]?.source : undefined);
  const cardKind = s.hands[s.active].find((v) => v.id === c.cardId)?.kind;
  if (caster?.kind === 10 && !caster.silenced && !friend) {
    const follow = markFollowUp(s, t, owner, s.ply + 2);
    const immediate = t.unit ? t.unit.hp === t.unit.maxHp : s.bases[t.owner] === 300;
    return (
      (immediate ? 5 : 0) +
      follow * (t.unit ? 12 + materialValue(s, t.unit) * 0.25 : 15) +
      (t.unit && immediate ? materialValue(s, t.unit) * 0.08 : 0)
    );
  }
  if (c.type === 'cast' && (cardKind === 18 || cardKind === 22) && t.unit) {
    const type = cardKind === 18 ? 'execute' : 'convert';
    const carrier = {
      ...t.unit,
      effects: [
        ...t.unit.effects,
        { type, owner, from: now(s, t.unit) + 2, until: now(s, t.unit) + 4 },
      ],
    } as Unit;
    const view = { ...s, units: s.units.map((v) => (v.id === carrier.id ? carrier : v)) };
    return payloadOpportunity(view, carrier, type, (v) => materialValue(view, v));
  }
  if (!t.unit) return friend ? (300 - s.bases[t.owner]) * 0.6 : 80 + (300 - s.bases[t.owner]);
  const u = t.unit,
    value = unitValue(s, u);
  if (c.type === 'attack' || c.type === 'react') {
    if (friend) return (u.maxHp - u.hp) * 2;
    const shots = caster ? getStats(s, caster).remaining : 1;
    const packets = caster ? hitPackets(s, caster, t) : [];
    const burst = packets.length === 1 && caster?.kind !== 9 && caster?.kind !== 10 ? shots : 1;
    const killChance = packets.reduce(
      (p, hit) => p + (hit.damage * burst >= u.hp ? hit.probability : 0),
      0,
    );
    return value + killChance * (value * 0.7 + 20) + laneRelease(s, u);
  }
  if (a.id === 'siphon')
    return c.targetId
      ? friend
        ? (u.maxHp - u.hp) * 3 + 10
        : -value
      : friend
        ? -value
        : value + 30;
  if (a.id === 'reforge-two' || a.id === 'sacrifice')
    return -value + (a.id === 'sacrifice' ? getStats(s, u).attack : 0);
  return friend ? value : value + 30;
}
function pointRank(s: GameState, a: ActionSpec, c: Command, p: Point, u?: Unit): number {
  if (
    c.type === 'move' ||
    c.type === 'deploy' ||
    a.id === 'dash' ||
    a.id === 'revive' ||
    a.id === 'hut-spawn'
  ) {
    const k = s.hands[s.active].find((v) => v.id === c.cardId)?.kind;
    const ghost =
      u ??
      template(
        a.id === 'hut-spawn'
          ? 20
          : c.deathId
            ? (s.deaths.find((d) => d.id === c.deathId)?.kind ?? 1)
            : (k ?? 1),
        decisionOwner(s),
        0,
        p,
      );
    let value = placementValue(s, ghost, p);
    if (isRunner(ghost))
      value += occupants(s, p)
        .filter((v) => v.id !== ghost.id && v.owner !== ghost.owner)
        .reduce((n, v) => n + Math.min(30, v.hp) + 10, 0);
    return value;
  }
  if (a.id === 'hook') {
    const victim = s.units.find((v) => v.id === c.targetId);
    if (victim) return -placementValue(s, victim, p);
  }
  if (a.id === 'blast' || a.id === 'cross' || a.id === 'ice-mark') {
    let value = 0;
    for (const t of targets(s)) {
      const hit = (t.unit ? cells(t.unit) : [t]).some((q) =>
        a.id === 'blast'
          ? q.x >= p.x && q.x <= p.x + 1 && q.y >= p.y && q.y <= p.y + 1
          : a.id === 'cross'
            ? (q.x === p.x || q.y === p.y) &&
              !!u &&
              Math.abs(q.x - u.x) <= 5 &&
              Math.abs(q.y - u.y) <= 5
            : equal(q, p),
      );
      if (hit)
        value +=
          (t.owner === decisionOwner(s) ? -1 : 1) * (t.unit ? Math.min(t.unit.hp, 20) + 15 : 40);
    }
    return value;
  }
  return u ? -distance(p, u) : 0;
}
function points(s: GameState, a: ActionSpec, c: Command): Point[] {
  const u =
    s.units.find((v) => v.id === c.unitId) ??
    (c.type === 'react' ? s.pending[0]?.source : undefined);
  if ((c.type === 'move' && u && isRunner(u)) || a.id === 'bounce') return u ? neighbors(u) : [];
  const all = ALL_CELLS.filter((p) => {
    if (c.type === 'deploy') {
      const card = s.hands[s.active].find((v) => v.id === c.cardId);
      return !!card && canPlace(s, template(card.kind, s.active, s.turns[s.active], p), p, true);
    }
    if (c.type === 'move' && u) {
      const st = getStats(s, u);
      const limit = st.move % 1 ? st.move * 2 : st.move;
      return (
        distance(u, p) <= limit && !!movementPath(s, u, p, limit, u.kind === 13 && !u.silenced)
      );
    }
    if (a.id === 'dash' && u) return distance(u, p) <= 6 && !!movementPath(s, u, p, 6);
    if (a.id === 'blast') return p.x < 9 && p.y < 13;
    if ((a.id === 'cross' || a.id === 'ice-mark') && u)
      return Math.abs(p.x - u.x) <= 5 && Math.abs(p.y - u.y) <= 5;
    if (a.id === 'hook' && u) {
      const v = s.units.find((v) => v.id === c.targetId);
      return (
        !!v &&
        !equal(v, p) &&
        canPlace(s, v, p) &&
        !!attackPath(s, u, asTarget({ ...v, ...p }), getStats(s, u).range)
      );
    }
    if (a.id === 'wall' && u)
      return (
        canPlace(s, template('wall', u.owner, 0, p), p) &&
        !!attackPath(s, u, p, getStats(s, u).range)
      );
    if (a.id === 'revive' || a.id === 'hut-spawn')
      return !!u && distance(p, u) <= getStats(s, u).range + 1;
    return true;
  });
  const ranked = all.map((p) => ({ p, score: pointRank(s, a, c, p, u) }));
  return ranked.sort((a, b) => b.score - a.score).map((v) => v.p);
}
function choices(s: GameState, a: ActionSpec, c: Command, step: SelectionStep): Command[] {
  if (step.kind === 'target') {
    const u =
      s.units.find((v) => v.id === c.unitId) ??
      (c.type === 'react' ? s.pending[0]?.source : undefined);
    const view = u && a.id === 'dash' && c.x !== undefined ? { ...u, x: c.x, y: c.y! } : u;
    return targets(s)
      .filter((t) => {
        if (step.unitOnly && !t.unit) return false;
        const side = t.unit ? allegiance(s, t.unit) : t.owner,
          owner = decisionOwner(s);
        if (step.relation === 'friend' && side !== owner) return false;
        if (step.relation === 'enemy' && side === owner) return false;
        if (c.sacrificeIds?.includes(t.id) || c.targetId === t.id) return false;
        if (
          step.field === 'sacrificeIds' &&
          (!t.unit || t.unit.kind === 'u25' || t.unit.hp * 2 < t.unit.maxHp)
        )
          return false;
        if (a.id === 'sacrifice' && (t.id === u?.id || t.unit?.kind === 'u25')) return false;
        // Friendly spells can target a particular clone, attacks still obey top-of-stack rules.
        if (c.type === 'attack' || c.type === 'react') {
          if (!topTarget(s, t)) return false;
          if (
            view &&
            side === owner &&
            t.unit &&
            !view.silenced &&
            (view.kind === 2 || view.kind === 'u21') &&
            t.unit.hp === t.unit.maxHp
          )
            return false;
          if (
            view?.kind === 10 &&
            passive(s, view) &&
            t.unit?.kind === 'u18' &&
            passive(s, t.unit) &&
            t.unit.hp === t.unit.maxHp &&
            getStats(s, view).attack <= 10 &&
            !has(s, view, 'execute') &&
            !has(s, view, 'convert') &&
            !view.equipment.includes('u28')
          )
            return false;
          // Don't spend visible actions and beam slots on zero-damage, zero-effect hits.
          // Real on-hit mechanics (marks, execution, burning, equipment) remain candidates.
          if (
            view &&
            side !== owner &&
            getStats(s, view).attack === 0 &&
            !(passive(s, view) && (view.kind === 10 || view.kind === 'u6')) &&
            !has(s, view, 'execute') &&
            !has(s, view, 'convert') &&
            !view.equipment.includes('u5') &&
            !view.equipment.includes('u28') &&
            !t.unit?.effects.some((e) => e.type === 'mark' && e.owner === owner && e.until > s.ply)
          )
            return false;
          if (
            view?.kind === 10 &&
            !view.silenced &&
            side !== owner &&
            getStats(s, view).attack === 0 &&
            !has(s, view, 'execute') &&
            !has(s, view, 'convert') &&
            !(t.unit ? t.unit.hp === t.unit.maxHp : s.bases[t.owner] === 300) &&
            !markFollowUp(s, t, owner, s.ply + 2)
          )
            return false;
          if (
            side === owner &&
            !(view && !view.silenced && (view.kind === 2 || view.kind === 'u21') && t.unit) &&
            !(t.unit?.kind === 16 && !t.unit.silenced && t.id !== view?.id)
          )
            return false;
        }
        if (
          step.range &&
          view &&
          !view.equipment.includes('u28') &&
          !attackPath(s, view, t, getStats(s, view).range)
        )
          return false;
        return true;
      })
      .sort((x, y) => targetRank(s, a, c, y) - targetRank(s, a, c, x))
      .map((t) =>
        step.field === 'sacrificeIds'
          ? { ...c, sacrificeIds: [...(c.sacrificeIds ?? []), t.id] }
          : { ...c, [step.field ?? 'targetId']: t.id },
      );
  }
  if (step.kind === 'point') return points(s, a, c).map((p) => ({ ...c, ...p }));
  if (step.kind === 'death')
    return s.deaths
      .filter((d) => d.owner === s.active && !d.revived && s.ply - d.ply <= 4 && s.ply > d.ply)
      .sort((a, b) => definition(b.kind).attack - definition(a.kind).attack)
      .map((d) => ({ ...c, deathId: d.id }));
  const lines = Array.from({ length: step.kind === 'row' ? 13 : 9 }, (_, i) => ({
    ...c,
    [step.kind]: i + 1,
  }));
  const lineValue = (command: Command) =>
    targets(s).reduce((score, t) => {
      const hit = (t.unit ? cells(t.unit) : [t]).some((p) =>
        step.kind === 'row' ? p.y === command.row : p.x === command.column,
      );
      if (!hit) return score;
      return (
        score +
        (t.owner === decisionOwner(s) ? -1 : 1) *
          (t.unit ? Math.min(20, t.unit.hp) + (t.unit.hp <= 20 ? materialValue(s, t.unit) : 0) : 25)
      );
    }, 0);
  return lines.sort((a, b) => lineValue(b) - lineValue(a));
}
function expand(s: GameState, a: ActionSpec, partial: number): Command[] {
  let drafts = [a.command];
  for (let i = 0; i < a.steps.length; i++) {
    const last = i === a.steps.length - 1;
    drafts = drafts.flatMap((c) => {
      const next = choices(s, a, c, a.steps[i]);
      return last ? next : next.slice(0, partial);
    });
  }
  if (
    a.command.type === 'deploy' &&
    s.hands[s.active].find((c) => c.id === a.command.cardId)?.kind === 1
  )
    return drafts.flatMap((c) => [c, { ...c, charge: true }]);
  return drafts;
}
/** Candidate abstraction shared by all levels; every final command is still engine-validated.
 * Groups retain setup skills and multiple actors instead of pruning solely by immediate damage. */
export function* iterateCandidateGroups(
  s: GameState,
  difficulty: Difficulty,
): Generator<CandidateGroup> {
  if (s.winner) return;
  const settings = DIFFICULTIES[difficulty];
  const make = (a: ActionSpec): CandidateGroup | null => {
    if (actionError(s, a)) return null;
    const commands = expand(s, a, settings.partial);
    return commands.length
      ? {
          family: `${a.command.unitId ?? a.command.cardId ?? 'system'}:${a.id}`,
          commands,
          keep: a.command.type === 'attack' ? Math.max(8, settings.perAction) : settings.perAction,
        }
      : null;
  };
  const emit = function* (a: ActionSpec) {
    const group = make(a);
    if (group) yield group;
  };
  if (s.pending.length) {
    yield* emit(reactionAction(s)!);
    if (s.pending[0].kind !== 'bounce')
      yield { family: 'reaction-skip', commands: [{ type: 'react' }], keep: 1 };
    return;
  }
  if (s.phase === 'summon') {
    yield {
      family: 'summon',
      keep: 2,
      commands:
        s.summonSlots > 0
          ? [
              { type: 'summon', ultimate: false },
              ...(s.heads[s.active] >= 2 ? [{ type: 'summon' as const, ultimate: true }] : []),
            ]
          : [{ type: 'begin' }],
    };
    for (const c of s.hands[s.active])
      for (const a of cardActions(s, c)) if (a.command.type === 'reroll') yield* emit(a);
    return;
  }
  const transit = s.units.find((u) => u.kind === 'u12p' && u.mode === 'move' && !canPlace(s, u, u));
  if (transit) {
    for (const a of unitActions(s, transit)) if (a.id === 'move') yield* emit(a);
    return;
  }
  const actions = s.units.filter((u) => u.owner === s.active).flatMap((u) => unitActions(s, u));
  actions.sort((a, b) => Number(b.command.type === 'attack') - Number(a.command.type === 'attack'));
  for (const a of actions) yield* emit(a);
  for (const c of s.hands[s.active]) for (const a of cardActions(s, c)) yield* emit(a);
  const hearts = s.hands[s.active].filter((c) => c.kind === 'u28');
  if (hearts.length >= 3)
    yield {
      family: 'craft',
      commands: [{ type: 'craft', cardIds: hearts.slice(0, 3).map((c) => c.id) }],
      keep: 1,
    };
  yield { family: 'end', commands: [{ type: 'end' }], keep: 1 };
}
export function candidateGroups(s: GameState, difficulty: Difficulty): CandidateGroup[] {
  return [...iterateCandidateGroups(s, difficulty)];
}

/** Cheap tactical follow-ups: reuse the same target rules without enumerating all movements. */
export function attackCandidates(s: GameState, id: string): Command[] {
  const unit = s.units.find((u) => u.id === id);
  if (!unit || s.pending.length) return [];
  const action = unitActions(s, unit).find((a) => a.command.type === 'attack');
  return action && !actionError(s, action) ? expand(s, action, 1) : [];
}
