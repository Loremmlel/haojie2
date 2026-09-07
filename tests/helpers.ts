import assert from 'node:assert/strict';
import { createGame, applyCommand, definition, isStored } from '../src/engine';
import { template, random } from '../src/engine/state';
import type { Card, GameState, Kind, Player, Unit } from '../src/engine';
export function fixture(): GameState {
  const s = createGame(19);
  s.phase = 'play';
  s.summonSlots = 0;
  s.ply = 5;
  s.turns = { 1: 3, 2: 2 };
  s.heads = { 1: 6, 2: 0 };
  s.log = [];
  s.events = [];
  return s;
}
export function add(s: GameState, kind: Kind, owner: Player, x: number, y: number): Unit {
  const u = template(kind, owner, 0, { x, y }, `test${s.serial++}`);
  s.units.push(u);
  return u;
}
export function card(s: GameState, kind: Kind): string {
  const d = definition(kind),
    limit = d.spell ?? d.weapon,
    id = `card${s.serial++}`;
  s.hands[s.active].push({
    id,
    kind,
    drawnAt: s.turns[s.active],
    summonedPly: s.ply,
    ...(limit !== undefined && limit >= 0 ? { expiresAt: s.turns[s.active] + limit } : {}),
  });
  return id;
}
export function unit(s: GameState, id: string) {
  const u = s.units.find((u) => u.id === id);
  assert.ok(u, `unit ${id} must be alive`);
  return u;
}
export function pass(s: GameState): GameState {
  const n = structuredClone(s);
  n.phase = 'play';
  n.summonSlots = 0;
  n.hands[n.active] = n.hands[n.active].filter((c) => isStored(definition(c.kind)));
  return applyCommand(n, { type: 'end' });
}
export function round(s: GameState): GameState {
  const n = pass(pass(s));
  n.phase = 'play';
  n.summonSlots = 0;
  return n;
}
export const strike = (s: GameState, u: Unit, v: Unit) =>
  applyCommand(s, { type: 'attack', unitId: u.id, targetId: v.id });
export function seedFor(min: number, max: number) {
  for (let i = 1; i < 1e6; i++) {
    const s = fixture();
    s.rng = i;
    const r = random(s);
    if (r >= min && r < max) return i;
  }
  throw new Error('seed not found');
}
