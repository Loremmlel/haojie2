import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ALL_CELLS,
  CATALOG,
  applyCommand,
  attackPath,
  canPlace,
  commandError,
  createDemoGame,
  createGame,
  createSession,
  definition,
  dispatch,
  getStats,
  parseSession,
  redo,
  undo,
} from '../src/engine';
import { asTarget, random, template } from '../src/engine/state';
import type { Command, GameState, Kind, Player, Unit } from '../src/engine';

function fixture(): GameState {
  const s = createGame(19);
  s.units = [];
  s.hands = { 1: [], 2: [] };
  s.turns = { 1: 3, 2: 2 };
  s.ply = 5;
  s.log = [];
  s.events = [];
  return s;
}
function add(s: GameState, kind: Kind, owner: Player, x: number, y: number): Unit {
  const u = template(kind, owner, 0, { x, y }, `test-u-${s.serial++}`);
  s.units.push(u);
  return u;
}
function card(s: GameState, kind: Kind): string {
  const id = `test-c-${s.serial++}`,
    d = definition(kind),
    turn = s.turns[s.active];
  s.hands[s.active].push({
    id,
    kind,
    drawnAt: turn,
    ...(d.spell ? { expiresAt: turn + d.spell } : {}),
  });
  return id;
}
function unit(s: GameState, id: string): Unit {
  const u = s.units.find((u) => u.id === id);
  assert.ok(u);
  return u;
}
function pass(s: GameState): GameState {
  const next = structuredClone(s);
  next.hands[next.active] = next.hands[next.active].filter((c) => definition(c.kind).spell);
  return applyCommand(next, { type: 'end' });
}
function round(s: GameState): GameState {
  return pass(pass(s));
}
const strike = (s: GameState, u: Unit, v: Unit) =>
  applyCommand(s, { type: 'attack', unitId: u.id, targetId: v.id });

test('117 cells, 300-health bases, exactly 26 summon definitions plus three variants', () => {
  const s = createGame(7);
  assert.equal(ALL_CELLS.length, 117);
  assert.deepEqual(s.bases, { 1: 300, 2: 300 });
  assert.equal(s.hands[1].length, 2);
  assert.equal(CATALOG.filter((d) => typeof d.id === 'number').length, 26);
  assert.equal(CATALOG.length, 29);
});
test('same seed produces the same opening; invalid commands are atomic', () => {
  assert.deepEqual(createGame(712), createGame(712));
  const s = fixture(),
    before = structuredClone(s);
  assert.throws(() => applyCommand(s, { type: 'move', unitId: 'missing', x: 1, y: 1 }));
  assert.deepEqual(s, before);
});
test('minions must deploy before ending; deployment fatigue lasts until next own turn', () => {
  let s = fixture();
  const id = card(s, 9);
  assert.match(commandError(s, { type: 'end' })!, /部署/);
  s = applyCommand(s, { type: 'deploy', cardId: id, x: 2, y: 4 });
  const u = s.units[0];
  assert.equal(getStats(s, u).remaining, 0);
  s = round(s);
  assert.equal(getStats(s, unit(s, u.id)).remaining, 2);
});
test('deployment respects sides, base occupancy and all four big-unit squares', () => {
  const s = fixture(),
    id = card(s, 5);
  assert.ok(commandError(s, { type: 'deploy', cardId: id, x: 9, y: 4 }));
  assert.ok(commandError(s, { type: 'deploy', cardId: id, x: 2, y: 8 }));
  assert.ok(commandError(s, { type: 'deploy', cardId: id, x: 5, y: 1 }));
  const n = applyCommand(s, { type: 'deploy', cardId: id, x: 2, y: 4 });
  assert.equal(n.units.length, 1);
  assert.equal(canPlace(n, template(1, 1, 3, { x: 3, y: 5 }), { x: 3, y: 5 }, true), false);
});
test('01 charge trades 10 maximum/current HP for immediate action', () => {
  const s = fixture(),
    id = card(s, 1),
    n = applyCommand(s, { type: 'deploy', cardId: id, x: 1, y: 3, charge: true });
  assert.equal(n.units[0].hp, 40);
  assert.equal(n.units[0].maxHp, 40);
  assert.equal(getStats(n, n.units[0]).remaining, 1);
});
test('01 attacks use the specified three deterministic critical-hit tiers', () => {
  const wanted = [
    { low: 0, high: 1 / 12, damage: 80 },
    { low: 1 / 12, high: 1 / 3, damage: 40 },
    { low: 1 / 3, high: 1, damage: 20 },
  ];
  for (const tier of wanted) {
    let s = fixture(),
      seed = 1;
    while (true) {
      const copy = { ...s, rng: seed };
      const r = random(copy);
      if (r >= tier.low && r < tier.high) break;
      seed++;
    }
    s.rng = seed;
    const a = add(s, 1, 1, 2, 3),
      b = add(s, 5, 2, 3, 4);
    const n = strike(s, a, b);
    assert.equal(unit(n, b.id).hp, 111 - tier.damage);
  }
});
test('attacks cannot pierce enemies but can detour or pass allies', () => {
  const s = fixture(),
    a = add(s, 1, 1, 1, 2),
    b = add(s, 1, 2, 3, 2),
    block = add(s, 1, 2, 2, 2);
  assert.equal(attackPath(s, a, asTarget(b), 2), null);
  assert.equal(attackPath(s, a, asTarget(b), 4)?.length, 5);
  block.owner = 1;
  assert.equal(attackPath(s, a, asTarget(b), 2)?.length, 3);
});
test('02 friendly attack heals without exceeding maximum', () => {
  const s = fixture(),
    a = add(s, 2, 1, 2, 3),
    b = add(s, 1, 1, 3, 3);
  b.hp = 42;
  const n = strike(s, a, b);
  assert.equal(unit(n, b.id).hp, 50);
  assert.equal(unit(n, a.id).spent, 1);
});
test('02 death reaction belongs to the deceased owner and pauses ordinary actions', () => {
  const s = fixture(),
    a = add(s, 1, 1, 2, 3),
    b = add(s, 2, 2, 3, 3);
  b.hp = 1;
  let n = strike(s, a, b);
  assert.equal(n.pending[0].owner, 2);
  assert.equal(n.active, 1);
  assert.ok(commandError(n, { type: 'end' }));
  n = applyCommand(n, { type: 'react', targetId: a.id });
  assert.equal(unit(n, a.id).hp, 30);
  assert.equal(n.pending.length, 0);
  assert.equal(n.active, 1);
});
test('03 guardian rescues once at one HP; it also protects itself', () => {
  let s = fixture();
  const attacker = add(s, 4, 1, 3, 4),
    guardian = add(s, 3, 2, 3, 6),
    friend = add(s, 1, 2, 4, 6);
  attacker.charge = 5;
  friend.hp = 5;
  s = strike(s, attacker, friend);
  assert.equal(unit(s, friend.id).hp, 1);
  assert.equal(unit(s, friend.id).guardUsed, true);
  unit(s, attacker.id).spent = 0;
  unit(s, attacker.id).charge = 5;
  s = strike(s, unit(s, attacker.id), unit(s, friend.id));
  assert.equal(
    s.units.some((u) => u.id === friend.id),
    false,
  );
  unit(s, attacker.id).spent = 0;
  unit(s, attacker.id).charge = 5;
  s = strike(s, unit(s, attacker.id), unit(s, guardian.id));
  assert.equal(unit(s, guardian.id).hp, 1);
});
test('03 draw transformation is deterministic and both forms can occur', () => {
  const seen = new Set<Kind>();
  for (let seed = 1; seed < 1000; seed++) {
    const s = createGame(seed);
    for (const c of s.hands[1]) if (c.kind === 3 || c.kind === '3p') seen.add(c.kind);
  }
  assert.deepEqual([...seen].sort(), [3, '3p'].sort());
});
test('03-prime dynamically counts self, both factions and big units only once', () => {
  const s = fixture(),
    a = add(s, '3p', 1, 4, 4);
  add(s, 5, 2, 5, 4);
  add(s, 'grave', 1, 3, 3);
  const stats = getStats(s, a);
  assert.equal(stats.attack, 25);
  assert.equal(stats.range, 3);
});
test('04 cannon charges automatically, moving preserves charge, firing clears it', () => {
  let s = fixture();
  const a = add(s, 4, 1, 2, 3),
    b = add(s, 5, 2, 2, 7);
  a.charge = 1;
  assert.ok(commandError(s, { type: 'attack', unitId: a.id, targetId: b.id }));
  s = applyCommand(s, { type: 'move', unitId: a.id, x: 2, y: 4 });
  assert.equal(unit(s, a.id).charge, 1);
  s = round(s);
  assert.equal(unit(s, a.id).charge, 2);
  s = strike(s, unit(s, a.id), unit(s, b.id));
  assert.equal(unit(s, a.id).charge, 0);
  assert.equal(unit(s, b.id).hp, 11);
});
test('04 five charge grants exactly one extra range', () => {
  const s = fixture(),
    a = add(s, 4, 1, 2, 3),
    b = add(s, 1, 2, 2, 8);
  a.charge = 4;
  assert.ok(commandError(s, { type: 'attack', unitId: a.id, targetId: b.id }));
  a.charge = 5;
  assert.equal(commandError(s, { type: 'attack', unitId: a.id, targetId: b.id }), null);
});
test('05 big unit needs a banking action before a one-cell whole-footprint move', () => {
  let s = fixture();
  const a = add(s, 5, 1, 2, 3);
  assert.ok(commandError(s, { type: 'move', unitId: a.id, x: 3, y: 3 }));
  s = applyCommand(s, { type: 'skill', unitId: a.id });
  s = round(s);
  s = applyCommand(s, { type: 'move', unitId: a.id, x: 3, y: 3 });
  assert.equal(unit(s, a.id).x, 3);
  assert.equal(unit(s, a.id).charge, 0);
});
test('06 encouragement is next-own-turn-only and excludes catapults', () => {
  let s = fixture();
  const a = add(s, 6, 1, 3, 4),
    b = add(s, 1, 1, 4, 4),
    c = add(s, 10, 1, 2, 4);
  s = applyCommand(s, { type: 'skill', unitId: a.id });
  assert.equal(getStats(s, unit(s, b.id)).attack, 20);
  s = round(s);
  assert.equal(getStats(s, unit(s, b.id)).attack, 30);
  assert.equal(getStats(s, unit(s, c.id)).attack, 0);
  s = round(s);
  assert.equal(getStats(s, unit(s, b.id)).attack, 20);
});
test('07 hook moves an enemy to a legal in-range cell atomically', () => {
  const s = fixture(),
    a = add(s, 7, 1, 3, 4),
    b = add(s, 1, 2, 3, 6);
  const n = applyCommand(s, { type: 'skill', unitId: a.id, targetId: b.id, x: 4, y: 4 });
  assert.equal(unit(n, b.id).x, 4);
  assert.equal(unit(n, b.id).y, 4);
  assert.equal(unit(n, a.id).spent, 1);
  assert.ok(commandError(s, { type: 'skill', unitId: a.id, targetId: b.id, x: 9, y: 13 }));
});
test('08 blast hits each big minion once and can damage friendly bases', () => {
  const s = fixture(),
    big = add(s, 5, 1, 2, 3),
    id = card(s, 8),
    n = applyCommand(s, { type: 'cast', cardId: id, x: 2, y: 3 });
  assert.equal(unit(n, big.id).hp, 91);
  const s2 = fixture(),
    id2 = card(s2, 8),
    n2 = applyCommand(s2, { type: 'cast', cardId: id2, x: 4, y: 1 });
  assert.equal(n2.bases[1], 280);
});
test('09 archer must use different attack targets', () => {
  const s = fixture(),
    a = add(s, 9, 1, 3, 4),
    b = add(s, 5, 2, 3, 6);
  const n = strike(s, a, b);
  assert.match(commandError(n, { type: 'attack', unitId: a.id, targetId: b.id })!, /同一目标/);
});
test('10 marks trigger immediately at full health, otherwise on another allied hit', () => {
  let s = fixture();
  const a = add(s, 10, 1, 2, 4),
    b = add(s, 5, 2, 3, 5),
    c = add(s, 9, 1, 2, 5);
  s = strike(s, a, b);
  assert.equal(unit(s, b.id).hp, 106);
  assert.equal(unit(s, b.id).effects.length, 0);
  s = strike(s, unit(s, a.id), unit(s, b.id));
  assert.equal(unit(s, b.id).hp, 106);
  assert.equal(unit(s, b.id).effects.length, 1);
  s = strike(s, unit(s, c.id), unit(s, b.id));
  assert.equal(unit(s, b.id).hp, 91);
  assert.equal(unit(s, b.id).effects.length, 0);
});
test('10 damaged bases retain a mark until an allied non-catapault hit', () => {
  let s = fixture();
  const a = add(s, 10, 1, 4, 10),
    b = add(s, 9, 1, 6, 10);
  s.bases[2] = 200;
  s = applyCommand(s, { type: 'attack', unitId: a.id, targetId: 'base-2' });
  assert.equal(s.bases[2], 200);
  assert.equal(s.baseEffects[2].length, 1);
  s = applyCommand(s, { type: 'attack', unitId: b.id, targetId: 'base-2' });
  assert.equal(s.bases[2], 185);
  assert.equal(s.baseEffects[2].length, 0);
});
test('11 death grants an additional summon on the owner next turn', () => {
  const s = fixture(),
    a = add(s, 1, 1, 3, 4),
    b = add(s, 11, 2, 3, 5);
  b.hp = 1;
  let n = strike(s, a, b);
  assert.equal(n.bonus[2], 1);
  n = pass(n);
  assert.equal(n.active, 2);
  assert.equal(n.hands[2].length, 3);
  assert.equal(n.bonus[2], 0);
});
test('12 adjacency reduces action cap and movement; death creates a 70 HP grave', () => {
  const s = fixture(),
    a = add(s, 12, 1, 3, 4),
    b = add(s, 1, 2, 3, 5);
  assert.equal(getStats(s, a).actions, 1);
  assert.equal(getStats(s, a).move, 1);
  s.active = 2;
  const n = strike(s, b, a),
    grave = n.units.find((u) => u.kind === 'grave');
  assert.ok(grave);
  assert.equal(grave.hp, 70);
  assert.equal(grave.owner, 1);
  assert.deepEqual([grave.x, grave.y], [3, 4]);
});
test('13 straight mover requires exactly three unobstructed cardinal cells', () => {
  const s = fixture(),
    a = add(s, 13, 1, 2, 3);
  assert.ok(commandError(s, { type: 'move', unitId: a.id, x: 4, y: 3 }));
  assert.ok(commandError(s, { type: 'move', unitId: a.id, x: 3, y: 5 }));
  assert.equal(commandError(s, { type: 'move', unitId: a.id, x: 5, y: 3 }), null);
  add(s, 1, 1, 3, 3);
  assert.ok(commandError(s, { type: 'move', unitId: a.id, x: 5, y: 3 }));
});
test('14 sacrifice uses sacrificed attack, costs max HP and chooses first enemy in a column', () => {
  const s = fixture(),
    a = add(s, 14, 1, 3, 4),
    b = add(s, 26, 1, 2, 4),
    target = add(s, 5, 2, 3, 6);
  b.attackBonus = 5;
  const n = applyCommand(s, { type: 'skill', unitId: a.id, targetId: b.id, column: 3 });
  assert.equal(unit(n, a.id).maxHp, 20);
  assert.equal(unit(n, a.id).hp, 20);
  assert.equal(unit(n, target.id).hp, 86);
  assert.equal(
    n.units.some((u) => u.id === b.id),
    false,
  );
});
test('15 upgrades are permanent, consume actions and stop at three', () => {
  let s = fixture();
  const a = add(s, 15, 1, 3, 4);
  s = applyCommand(s, { type: 'skill', unitId: a.id, mode: 'attack' });
  s = applyCommand(s, { type: 'skill', unitId: a.id, mode: 'range' });
  s = round(s);
  s = applyCommand(s, { type: 'skill', unitId: a.id, mode: 'attack' });
  assert.equal(getStats(s, unit(s, a.id)).attack, 25);
  assert.equal(getStats(s, unit(s, a.id)).range, 3);
  assert.ok(commandError(s, { type: 'skill', unitId: a.id, mode: 'range' }));
});
test('16 friendly damage creates an optional equal-damage enemy reaction', () => {
  const s = fixture(),
    a = add(s, 9, 1, 2, 4),
    mirror = add(s, 16, 1, 3, 4),
    b = add(s, 5, 2, 3, 5);
  let n = strike(s, a, mirror);
  assert.equal(n.pending[0].amount, 10);
  n = applyCommand(n, { type: 'react', targetId: b.id });
  assert.equal(unit(n, b.id).hp, 101);
  assert.equal(n.pending.length, 0);
});
test('17 gold body is immune during opponent turn, expires at next own start', () => {
  let s = fixture();
  const a = add(s, 1, 1, 3, 4),
    b = add(s, 9, 2, 3, 5),
    id = card(s, 17);
  s = applyCommand(s, { type: 'cast', cardId: id, targetId: a.id });
  s = pass(s);
  s = strike(s, unit(s, b.id), unit(s, a.id));
  assert.equal(unit(s, a.id).hp, 50);
  s = pass(s);
  assert.equal(unit(s, a.id).effects.length, 0);
});
test('18 execute activates next own turn, bypasses damage immunity, never executes bases', () => {
  let s = fixture();
  const a = add(s, 9, 1, 3, 4),
    b = add(s, 5, 2, 3, 6),
    id = card(s, 18);
  s = applyCommand(s, { type: 'cast', cardId: id, targetId: a.id });
  s = strike(s, unit(s, a.id), unit(s, b.id));
  assert.equal(unit(s, b.id).hp, 101);
  s = round(s);
  unit(s, b.id).effects.push({ type: 'immune', owner: 2, from: s.ply, until: s.ply + 3 });
  s = strike(s, unit(s, a.id), unit(s, b.id));
  assert.equal(
    s.units.some((u) => u.id === b.id),
    false,
  );
});
test('19 temporary barricade expires at next own turn', () => {
  let s = fixture();
  const a = add(s, 19, 1, 3, 4);
  s = applyCommand(s, { type: 'skill', unitId: a.id, x: 4, y: 4 });
  assert.equal(s.units.filter((u) => u.kind === 'wall').length, 1);
  s = round(s);
  assert.equal(s.units.filter((u) => u.kind === 'wall').length, 0);
});
test('20 death retaliation uses the enemy killer as target', () => {
  const s = fixture(),
    a = add(s, 26, 1, 3, 4),
    b = add(s, 20, 2, 3, 5),
    n = strike(s, a, b);
  assert.equal(unit(n, a.id).hp, 35);
  assert.equal(unit(n, a.id).maxHp, 55);
  assert.equal(unit(n, a.id).kills, 1);
});
test('21 two separate charges unlock a later six-cell move-and-attack action', () => {
  let s = fixture();
  const a = add(s, 21, 1, 2, 3),
    b = add(s, 5, 2, 5, 8);
  s = applyCommand(s, { type: 'skill', unitId: a.id, mode: 'charge' });
  s = round(s);
  s = applyCommand(s, { type: 'skill', unitId: a.id, mode: 'charge' });
  assert.ok(
    commandError(s, { type: 'skill', unitId: a.id, mode: 'dash', x: 5, y: 6, targetId: b.id }),
  );
  s = round(s);
  s = applyCommand(s, { type: 'skill', unitId: a.id, mode: 'dash', x: 5, y: 6, targetId: b.id });
  assert.equal(unit(s, a.id).hp, 25);
  assert.equal(unit(s, a.id).spent, 1);
  assert.equal(unit(s, a.id).charge, 0);
  assert.equal(unit(s, b.id).hp, 81);
});
test('22 conversion waits until next own turn, damages first, sleeps the surviving new ally', () => {
  let s = fixture();
  const a = add(s, 9, 1, 3, 4),
    b = add(s, 1, 2, 3, 6),
    id = card(s, 22);
  s = applyCommand(s, { type: 'cast', cardId: id, targetId: a.id });
  s = round(s);
  s = strike(s, unit(s, a.id), unit(s, b.id));
  assert.equal(unit(s, b.id).owner, 1);
  assert.equal(unit(s, b.id).hp, 40);
  assert.equal(getStats(s, unit(s, b.id)).remaining, 0);
});
test('22 a lethal conversion hit does not resurrect the victim', () => {
  let s = fixture();
  const a = add(s, 9, 1, 3, 4),
    b = add(s, 1, 2, 3, 6),
    id = card(s, 22);
  b.hp = 5;
  s = applyCommand(s, { type: 'cast', cardId: id, targetId: a.id });
  s = round(s);
  s = strike(s, unit(s, a.id), unit(s, b.id));
  assert.equal(
    s.units.some((u) => u.id === b.id),
    false,
  );
});
test('23 lone fighter alternates resting and six-action turns; excludes all eight allied neighbors', () => {
  const s = fixture(),
    a = add(s, 23, 1, 4, 4);
  a.born = 1;
  assert.equal(getStats(s, a).remaining, 6);
  s.turns[1] = 4;
  assert.equal(getStats(s, a).remaining, 0);
  s.turns[1] = 5;
  assert.equal(getStats(s, a).remaining, 6);
  assert.equal(canPlace(s, template(1, 1, 5, { x: 5, y: 5 }), { x: 5, y: 5 }, true), false);
  assert.equal(canPlace(s, template(1, 2, 2, { x: 5, y: 5 }), { x: 5, y: 5 }), true);
});
test('24 frontal attack is capped at ten, same-row flanking is not', () => {
  const s = fixture(),
    a = add(s, 26, 1, 3, 4),
    b = add(s, 24, 2, 3, 6);
  assert.equal(unit(strike(s, a, b), b.id).hp, 40);
  a.x = 1;
  a.y = 6;
  assert.equal(unit(strike(s, a, b), b.id).hp, 30);
});
test('25 reforge offers one free draw or two draws for two distinct half-health allies', () => {
  const s = fixture(),
    id = card(s, 25),
    n = applyCommand(s, { type: 'cast', cardId: id, mode: 'single' });
  assert.equal(n.hands[1].length, 1);
  const s2 = fixture(),
    a = add(s2, 1, 1, 2, 4),
    b = add(s2, 1, 1, 3, 4),
    id2 = card(s2, 25);
  a.hp = 25;
  b.hp = 25;
  assert.ok(
    commandError(s2, { type: 'cast', cardId: id2, mode: 'double', sacrificeIds: [a.id, a.id] }),
  );
  const n2 = applyCommand(s2, {
    type: 'cast',
    cardId: id2,
    mode: 'double',
    sacrificeIds: [a.id, b.id],
  });
  assert.equal(n2.units.length, 0);
  assert.equal(n2.hands[1].length, 2);
});
test('26 kill growth repeats health, attack, range in order', () => {
  let s = fixture();
  const a = add(s, 26, 1, 3, 4);
  for (let i = 0; i < 4; i++) {
    unit(s, a.id).spent = 0;
    const b = add(s, 'grave', 2, 3, 5);
    b.hp = 1;
    s = strike(s, unit(s, a.id), b);
  }
  const u = unit(s, a.id);
  assert.equal(u.kills, 4);
  assert.equal(u.maxHp, 65);
  assert.equal(u.hp, 65);
  assert.equal(getStats(s, u).attack, 25);
  assert.equal(getStats(s, u).range, 5);
});
test('spell storage includes the draw turn and expires at the correct own start', () => {
  let s = fixture();
  const id = card(s, 8);
  s = round(s);
  s = round(s);
  s = round(s);
  assert.ok(s.hands[1].some((c) => c.id === id));
  s = round(s);
  assert.equal(
    s.hands[1].some((c) => c.id === id),
    false,
  );
});
test('extra deployment rows are sampled only at own turn start', () => {
  let s = fixture();
  add(s, 1, 1, 1, 10);
  add(s, 1, 1, 3, 10);
  assert.equal(s.deployRows[1].includes(10), false);
  s = round(s);
  assert.equal(s.deployRows[1].includes(10), true);
  s.units.pop();
  assert.equal(s.deployRows[1].includes(10), true);
  s = round(s);
  assert.equal(s.deployRows[1].includes(10), false);
});
test('base destruction ends the match and rejects later commands', () => {
  const s = fixture(),
    a = add(s, 9, 1, 5, 10);
  s.bases[2] = 10;
  const n = applyCommand(s, { type: 'attack', unitId: a.id, targetId: 'base-2' });
  assert.equal(n.winner, 1);
  assert.ok(commandError(n, { type: 'end' }));
});
test('undo and redo restore random numbers, hands, reactions and full state', () => {
  const s = createDemoGame(),
    a = s.units.find((u) => u.kind === 26)!,
    b = s.units.find((u) => u.kind === 20)!;
  const initial = createSession(s),
    next = dispatch(initial, { type: 'attack', unitId: a.id, targetId: b.id }),
    back = undo(next);
  assert.deepEqual(back.present, s);
  assert.deepEqual(redo(back).present, next.present);
  assert.deepEqual(
    dispatch(back, { type: 'attack', unitId: a.id, targetId: b.id }).present,
    next.present,
  );
});
test('save round trip preserves history; corrupt and incompatible saves are rejected', () => {
  const initial = createSession(createDemoGame());
  assert.deepEqual(parseSession(JSON.stringify(initial)), initial);
  assert.throws(() => parseSession('{oops'));
  assert.throws(() => parseSession(JSON.stringify({ ...initial, format: 'wrong' })));
  const bad = structuredClone(initial);
  bad.present.units[0].hp = -2;
  assert.throws(() => parseSession(JSON.stringify(bad)));
});
test('visual movement events preserve before and after positions', () => {
  const s = fixture(),
    a = add(s, 1, 1, 2, 3),
    n = applyCommand(s, { type: 'move', unitId: a.id, x: 2, y: 4 });
  const e = n.events.find((e) => e.type === 'move')!;
  assert.deepEqual(e.from, { x: 2, y: 3 });
  assert.deepEqual(e.to, { x: 2, y: 4 });
});

test('attack animation records the same legal detour as the rules', () => {
  const s = fixture(),
    a = add(s, 1, 1, 1, 2),
    b = add(s, 1, 2, 3, 2);
  add(s, 1, 2, 2, 2);
  const n = strike(s, a, b),
    path = n.events.find((e) => e.type === 'attack')?.path;
  assert.ok(path);
  assert.equal(path.length, 5);
  assert.deepEqual(path[0], { x: 1, y: 2 });
  assert.deepEqual(path.at(-1), { x: 3, y: 2 });
  assert.equal(
    path.some((p) => p.x === 2 && p.y === 2),
    false,
  );
});
test('friendly overkill reflects actual HP lost, even if the mirror dies', () => {
  const s = fixture(),
    a = add(s, 1, 1, 2, 4),
    mirror = add(s, 16, 1, 3, 4),
    enemy = add(s, 'grave', 2, 3, 5);
  mirror.hp = 5;
  let n = strike(s, a, mirror);
  assert.equal(n.pending[0].amount, 5);
  assert.deepEqual(parseSession(JSON.stringify(createSession(n))).present, n);
  n = applyCommand(n, { type: 'react', targetId: enemy.id });
  assert.equal(unit(n, enemy.id).hp, 65);
});
test('a completely blocked deployment zone discards the impossible minion instead of deadlocking', () => {
  const s = fixture();
  for (const p of ALL_CELLS)
    if (p.y <= 8 && !(p.x === 5 && p.y === 1)) add(s, 'grave', 1, p.x, p.y);
  card(s, 1);
  const n = applyCommand(s, { type: 'end' });
  assert.equal(n.active, 2);
  assert.equal(n.hands[1].length, 0);
  assert.ok(n.log.some((line) => line.includes('自动弃置')));
});
