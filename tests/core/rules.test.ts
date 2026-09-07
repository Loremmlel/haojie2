import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ALL_CELLS,
  CATALOG,
  applyCommand,
  attackPath,
  canPlace,
  commandError,
  createGame,
  definition,
  getStats,
  template,
  asTarget,
  has,
} from '../../src/engine';
import { add, card, fixture, pass, round, seedFor, strike, unit } from '../helpers';

test('board, pools, staged summons and base values match the source', () => {
  let s = createGame(7);
  assert.equal(ALL_CELLS.length, 117);
  assert.deepEqual(s.bases, { 1: 300, 2: 300 });
  assert.equal(s.phase, 'summon');
  assert.equal(s.hands[1].length, 0);
  s = applyCommand(s, { type: 'summon' });
  s = applyCommand(s, { type: 'summon' });
  assert.equal(s.hands[1].length, 2);
  s = applyCommand(s, { type: 'begin' });
  assert.equal(s.phase, 'play');
  assert.equal(CATALOG.filter((d) => d.tier === 'ultimate').length, 28);
});
test('summon numbers are reproducible and invalid commands are atomic', () => {
  assert.deepEqual(createGame(72), createGame(72));
  const s = fixture(),
    original = structuredClone(s);
  assert.throws(() => applyCommand(s, { type: 'move', unitId: 'unknown', x: 1, y: 1 }));
  assert.deepEqual(s, original);
});
test('minion deployment cannot be deferred, ordinary fatigue expires next own turn', () => {
  let s = fixture();
  const id = card(s, 9);
  assert.match(commandError(s, { type: 'end' })!, /部署/);
  s = applyCommand(s, { type: 'deploy', cardId: id, x: 2, y: 4 });
  const u = s.units[0];
  assert.equal(getStats(s, u).operationsLeft, 0);
  s = round(s);
  assert.equal(getStats(s, unit(s, u.id)).remaining, 2);
});
test('deployment checks all 2x2 cells, bases and row limits', () => {
  const s = fixture(),
    id = card(s, 5);
  for (const p of [
    { x: 9, y: 4 },
    { x: 2, y: 8 },
    { x: 5, y: 1 },
  ])
    assert.ok(commandError(s, { type: 'deploy', cardId: id, ...p }));
  assert.equal(applyCommand(s, { type: 'deploy', cardId: id, x: 2, y: 4 }).units[0].size, 2);
});
test('01 optional charge costs ten HP and allows an operation immediately', () => {
  const s = fixture(),
    id = card(s, 1),
    n = applyCommand(s, { type: 'deploy', cardId: id, x: 2, y: 3, charge: true });
  assert.equal(n.units[0].maxHp, 40);
  assert.equal(getStats(n, n.units[0]).operationsLeft, 1);
});
test('01 all three critical tiers use the documented mutually exclusive probabilities', () => {
  for (const [lo, hi, amount] of [
    [0, 1 / 12, 80],
    [1 / 12, 1 / 3, 40],
    [1 / 3, 1, 20],
  ]) {
    const s = fixture(),
      u = add(s, 1, 1, 2, 3),
      v = add(s, 5, 2, 3, 4);
    s.rng = seedFor(lo, hi);
    assert.equal(unit(strike(s, u, v), v.id).hp, 111 - amount);
  }
});
test('02 ally healing caps at max, death shot belongs to deceased owner', () => {
  let s = fixture();
  const healer = add(s, 2, 1, 2, 3),
    friend = add(s, 1, 1, 3, 3);
  friend.hp = 42;
  s = strike(s, healer, friend);
  assert.equal(unit(s, friend.id).hp, 50);
  s = fixture();
  const a = add(s, 26, 1, 3, 4),
    b = add(s, 2, 2, 3, 5);
  b.hp = 1;
  s = strike(s, a, b);
  assert.equal(s.pending[0].owner, 2);
  assert.ok(commandError(s, { type: 'end' }));
  s = applyCommand(s, { type: 'react', targetId: a.id });
  assert.equal(unit(s, a.id).hp, 35);
});
test('03 no attacks does not remove movement; movement needs a previous charge round', () => {
  let s = fixture();
  const u = add(s, 3, 1, 3, 4);
  assert.equal(getStats(s, u).actions, 0);
  s = applyCommand(s, { type: 'charge', unitId: u.id, mode: 'move' });
  assert.ok(commandError(s, { type: 'move', unitId: u.id, x: 3, y: 5 }));
  s = round(s);
  s = applyCommand(s, { type: 'move', unitId: u.id, x: 3, y: 5 });
  assert.equal(unit(s, u.id).y, 5);
  assert.equal(unit(s, u.id).charge, 0);
});
test('03 guardian protects once, including itself', () => {
  let s = fixture();
  const a = add(s, 26, 1, 3, 4),
    guardian = add(s, 3, 2, 4, 5),
    friend = add(s, 1, 2, 3, 5);
  friend.hp = 1;
  s = strike(s, a, friend);
  assert.equal(unit(s, friend.id).hp, 1);
  assert.equal(unit(s, friend.id).guardUsed, true);
  s = round(s);
  s = strike(s, unit(s, a.id), unit(s, friend.id));
  assert.equal(
    s.units.some((u) => u.id === friend.id),
    false,
  );
});
test('03-prime counts self, both sides and one large unit once in its 3x3', () => {
  const s = fixture(),
    u = add(s, '3p', 1, 4, 4);
  add(s, 5, 2, 5, 4);
  add(s, 'grave', 1, 3, 3);
  assert.equal(getStats(s, u).attack, 25);
  assert.equal(getStats(s, u).range, 3);
});
test('04 cannon charges are explicit; moving never auto-charges or reduces stored charge', () => {
  let s = fixture();
  const u = add(s, 4, 1, 2, 3),
    v = add(s, 5, 2, 2, 7);
  s = applyCommand(s, { type: 'charge', unitId: u.id, mode: 'attack' });
  s = round(s);
  assert.equal(unit(s, u.id).charge, 1);
  s = applyCommand(s, { type: 'move', unitId: u.id, x: 2, y: 4 });
  s = round(s);
  assert.equal(unit(s, u.id).charge, 1);
  s = applyCommand(s, { type: 'charge', unitId: u.id, mode: 'attack' });
  s = round(s);
  s = strike(s, unit(s, u.id), unit(s, v.id));
  assert.equal(unit(s, u.id).charge, 0);
  assert.equal(unit(s, v.id).hp, 11);
});
test('04 at five charge cannon gains exactly one attack range', () => {
  const s = fixture(),
    u = add(s, 4, 1, 2, 3),
    v = add(s, 1, 2, 2, 8);
  u.charge = u.readyCharge = 4;
  assert.ok(commandError(s, { type: 'attack', unitId: u.id, targetId: v.id }));
  u.charge = u.readyCharge = 5;
  assert.equal(commandError(s, { type: 'attack', unitId: u.id, targetId: v.id }), null);
});
test('05 shock ring includes four corners, hits large units once and can hurt allies', () => {
  let s = fixture();
  const u = add(s, 5, 1, 4, 5),
    friend = add(s, 1, 1, 3, 4),
    enemy = add(s, 5, 2, 6, 6);
  s = applyCommand(s, { type: 'skill', unitId: u.id });
  assert.equal(unit(s, friend.id).hp, 30);
  assert.equal(unit(s, enemy.id).hp, 91);
  assert.equal(unit(s, u.id).hp, 111);
  assert.ok(commandError(s, { type: 'move', unitId: u.id, x: 4, y: 6 }));
});
test('06 buff is next-own-turn only and does not apply to ordinary10', () => {
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
test('07 hook has separate target and in-range landing, blocked by gold body', () => {
  let s = fixture();
  const u = add(s, 7, 1, 3, 4),
    v = add(s, 1, 2, 3, 6);
  let n = applyCommand(s, { type: 'skill', unitId: u.id, targetId: v.id, x: 4, y: 4 });
  assert.equal(unit(n, v.id).x, 4);
  v.effects.push({ type: 'immune', owner: 2, from: s.ply, until: s.ply + 2 });
  n = applyCommand(s, { type: 'skill', unitId: u.id, targetId: v.id, x: 4, y: 4 });
  assert.equal(unit(n, v.id).x, 3);
});
test('08 blast damages 2x2 minions once and friendly bases', () => {
  const s = fixture(),
    u = add(s, 5, 1, 2, 3),
    id = card(s, 8);
  assert.equal(unit(applyCommand(s, { type: 'cast', cardId: id, x: 2, y: 3 }), u.id).hp, 91);
  const t = fixture(),
    id2 = card(t, 8);
  assert.equal(applyCommand(t, { type: 'cast', cardId: id2, x: 4, y: 1 }).bases[1], 280);
});
test('09 attack mode has two different targets; it cannot become move mode', () => {
  let s = fixture();
  const u = add(s, 9, 1, 3, 4),
    a = add(s, 5, 2, 3, 6),
    b = add(s, 1, 2, 5, 4);
  s = strike(s, u, a);
  assert.ok(commandError(s, { type: 'move', unitId: u.id, x: 2, y: 4 }));
  assert.ok(commandError(s, { type: 'attack', unitId: u.id, targetId: a.id }));
  s = strike(s, unit(s, u.id), unit(s, b.id));
  assert.equal(unit(s, b.id).hp, 40);
  assert.equal(getStats(s, unit(s, u.id)).operationsLeft, 0);
});
test('moving consumes the operation, not one of multiple attack counts', () => {
  let s = fixture();
  const u = add(s, 9, 1, 3, 4),
    v = add(s, 1, 2, 3, 6);
  s = applyCommand(s, { type: 'move', unitId: u.id, x: 2, y: 4 });
  assert.ok(commandError(s, { type: 'attack', unitId: u.id, targetId: v.id }));
});
test('10 base marks detonate only once at full HP, then require another non10 ally hit', () => {
  let s = fixture();
  const u = add(s, 10, 1, 4, 10),
    a = add(s, 9, 1, 6, 10);
  s = applyCommand(s, { type: 'attack', unitId: u.id, targetId: 'base-2' });
  assert.equal(s.bases[2], 295);
  s = applyCommand(s, { type: 'attack', unitId: u.id, targetId: 'base-2' });
  assert.equal(s.bases[2], 295);
  assert.equal(s.baseEffects[2].length, 1);
  s = applyCommand(s, { type: 'attack', unitId: a.id, targetId: 'base-2' });
  assert.equal(s.bases[2], 280);
  assert.equal(s.baseEffects[2].length, 0);
});
test('11 death adds one summon slot at next owner start', () => {
  let s = fixture();
  const u = add(s, 26, 1, 3, 4),
    v = add(s, 11, 2, 3, 5);
  s = strike(s, u, v);
  assert.equal(s.bonus[2], 1);
  s = pass(s);
  assert.equal(s.summonSlots, 3);
});
test('12 proximity decreases attack count and movement, death leaves a fresh grave', () => {
  let s = fixture();
  const u = add(s, 12, 1, 3, 4),
    v = add(s, 26, 2, 3, 5);
  assert.equal(getStats(s, u).actions, 1);
  assert.equal(getStats(s, u).move, 1);
  s.active = 2;
  s = strike(s, v, u);
  const grave = s.units.find((u) => u.kind === 'grave');
  assert.ok(grave);
  assert.equal(grave.hp, 70);
  assert.notEqual(grave.id, u.id);
});
test('13 requires exactly three straight unobstructed cells', () => {
  const s = fixture(),
    u = add(s, 13, 1, 2, 3);
  assert.ok(commandError(s, { type: 'move', unitId: u.id, x: 4, y: 3 }));
  assert.ok(commandError(s, { type: 'move', unitId: u.id, x: 3, y: 5 }));
  assert.equal(commandError(s, { type: 'move', unitId: u.id, x: 5, y: 3 }), null);
  add(s, 1, 1, 3, 3);
  assert.ok(commandError(s, { type: 'move', unitId: u.id, x: 5, y: 3 }));
});
test('14 sacrifice damage uses current attack and deducts life cap', () => {
  const s = fixture(),
    u = add(s, 14, 1, 3, 4),
    ally = add(s, 26, 1, 2, 4),
    enemy = add(s, 5, 2, 3, 6);
  ally.attackBonus = 5;
  const n = applyCommand(s, { type: 'skill', unitId: u.id, targetId: ally.id, column: 3 });
  assert.equal(unit(n, u.id).maxHp, 20);
  assert.equal(unit(n, enemy.id).hp, 86);
  assert.equal(
    n.units.some((v) => v.id === ally.id),
    false,
  );
});
test('15 upgrading uses the whole operation, lifetime maximum remains three', () => {
  let s = fixture();
  const u = add(s, 15, 1, 3, 4);
  s = applyCommand(s, { type: 'skill', unitId: u.id, mode: 'attack' });
  assert.ok(commandError(s, { type: 'skill', unitId: u.id, mode: 'range' }));
  s = round(s);
  s = applyCommand(s, { type: 'skill', unitId: u.id, mode: 'range' });
  s = round(s);
  s = applyCommand(s, { type: 'skill', unitId: u.id, mode: 'attack' });
  s = round(s);
  assert.ok(commandError(s, { type: 'skill', unitId: u.id, mode: 'range' }));
  assert.equal(getStats(s, unit(s, u.id)).attack, 25);
});
test('16 actual overkill loss becomes the optional reflection amount', () => {
  let s = fixture();
  const u = add(s, 9, 1, 2, 4),
    mirror = add(s, 16, 1, 3, 4),
    enemy = add(s, 5, 2, 3, 5);
  mirror.hp = 5;
  s = strike(s, u, mirror);
  assert.equal(s.pending[0].amount, 5);
  s = applyCommand(s, { type: 'react', targetId: enemy.id });
  assert.equal(unit(s, enemy.id).hp, 106);
});
test('17 gold body stops damage and hostile18 execution', () => {
  let s = fixture();
  const u = add(s, 9, 1, 3, 4),
    v = add(s, 1, 2, 3, 6);
  u.effects.push({ type: 'execute', owner: 1, from: s.ply, until: s.ply + 2 });
  v.effects.push({ type: 'immune', owner: 2, from: s.ply, until: s.ply + 2 });
  s = strike(s, u, v);
  assert.equal(unit(s, v.id).hp, 50);
  assert.equal(
    unit(s, u.id).effects.some((e) => e.type === 'execute'),
    false,
  );
});
test('18 execute is scheduled for next unit turn and cannot execute base', () => {
  let s = fixture();
  const u = add(s, 9, 1, 3, 4),
    v = add(s, 5, 2, 3, 6),
    id = card(s, 18);
  s = applyCommand(s, { type: 'cast', cardId: id, targetId: u.id });
  s = strike(s, u, v);
  assert.equal(unit(s, v.id).hp, 101);
  s = round(s);
  s = strike(s, unit(s, u.id), unit(s, v.id));
  assert.equal(
    s.units.some((a) => a.id === v.id),
    false,
  );
});
test('19 barricade expires at next own turn', () => {
  let s = fixture();
  const u = add(s, 19, 1, 3, 4);
  s = applyCommand(s, { type: 'skill', unitId: u.id, x: 4, y: 4 });
  assert.equal(s.units.filter((v) => v.kind === 'wall').length, 1);
  s = round(s);
  assert.equal(s.units.filter((v) => v.kind === 'wall').length, 0);
});
test('20 death lottery either denies a head or retaliates for30, deterministically', () => {
  for (const deny of [true, false]) {
    const s = fixture(),
      u = add(s, 26, 1, 3, 4),
      v = add(s, 20, 2, 3, 5);
    s.rng = seedFor(deny ? 0 : 0.5, deny ? 0.5 : 1);
    const n = strike(s, u, v);
    assert.equal(n.heads[1], deny ? 6 : 7);
    assert.equal(unit(n, u.id).hp, deny ? 55 : 25);
  }
});
test('21 two explicit charge rounds unlock move-and-attack as one skill', () => {
  let s = fixture();
  const u = add(s, 21, 1, 2, 3),
    v = add(s, 5, 2, 5, 8);
  s = applyCommand(s, { type: 'charge', unitId: u.id, mode: 'skill' });
  s = round(s);
  s = applyCommand(s, { type: 'charge', unitId: u.id, mode: 'skill' });
  s = round(s);
  s = applyCommand(s, { type: 'skill', unitId: u.id, x: 5, y: 6, targetId: v.id });
  assert.equal(unit(s, u.id).hp, 25);
  assert.equal(unit(s, v.id).hp, 81);
  assert.equal(unit(s, u.id).operations, 1);
});
test('22 conversion damages first, blocks on gold body, acts next own turn including lone fighter', () => {
  let s = fixture();
  const u = add(s, 9, 1, 3, 4),
    v = add(s, 23, 2, 3, 6);
  u.effects.push({ type: 'convert', owner: 1, from: s.ply, until: s.ply + 2 });
  s = strike(s, u, v);
  assert.equal(unit(s, v.id).owner, 1);
  assert.equal(getStats(s, unit(s, v.id)).operationsLeft, 0);
  s = round(s);
  assert.equal(getStats(s, unit(s, v.id)).operationsLeft, 1);
});
test('23 rests extra turn and creates an eight-neighbor exclusion zone', () => {
  const s = fixture(),
    u = add(s, 23, 1, 4, 4);
  u.born = 1;
  assert.equal(getStats(s, u).remaining, 6);
  s.turns[1] = 4;
  assert.equal(getStats(s, u).operationsLeft, 0);
  assert.equal(canPlace(s, template(1, 1, 4, { x: 5, y: 5 }), { x: 5, y: 5 }, true), false);
});
test('24 shortest-path frontal hit is capped, side hit is not', () => {
  const s = fixture(),
    u = add(s, 26, 1, 3, 4),
    v = add(s, 24, 2, 3, 6);
  assert.equal(unit(strike(s, u, v), v.id).hp, 40);
  u.x = 1;
  u.y = 6;
  assert.equal(unit(strike(s, u, v), v.id).hp, 30);
});
test('25 sacrifices require two distinct half-health non-clone allies', () => {
  const s = fixture(),
    u = add(s, 1, 1, 2, 4),
    v = add(s, 1, 1, 3, 4),
    id = card(s, 25);
  u.hp = v.hp = 25;
  assert.ok(
    commandError(s, { type: 'cast', cardId: id, mode: 'double', sacrificeIds: [u.id, u.id] }),
  );
  const n = applyCommand(s, {
    type: 'cast',
    cardId: id,
    mode: 'double',
    sacrificeIds: [u.id, v.id],
  });
  assert.equal(n.units.length, 0);
  assert.equal(n.hands[1].length, 2);
});
test('26 growth cycles cap/heal then attack then range', () => {
  let s = fixture();
  const u = add(s, 26, 1, 3, 4);
  for (let i = 0; i < 4; i++) {
    unit(s, u.id).operations = 0;
    const v = add(s, 'grave', 2, 3, 5);
    v.hp = 1;
    s = strike(s, unit(s, u.id), v);
  }
  assert.equal(unit(s, u.id).maxHp, 65);
  assert.equal(getStats(s, unit(s, u.id)).attack, 25);
  assert.equal(getStats(s, unit(s, u.id)).range, 5);
});
test('path detours avoid enemy and include exact visual path snapshots', () => {
  const s = fixture(),
    u = add(s, 26, 1, 1, 2),
    v = add(s, 1, 2, 3, 2);
  add(s, 1, 2, 2, 2);
  assert.equal(attackPath(s, u, asTarget(v), 2), null);
  const n = strike(s, u, v),
    p = n.events.find((e) => e.type === 'attack')!.path!;
  assert.equal(p.length, 5);
  assert.equal(
    p.some((p) => p.x === 2 && p.y === 2),
    false,
  );
});
test('storage includes draw turn and expires at exact next-own boundary', () => {
  let s = fixture();
  const id = card(s, 8);
  s = round(round(round(s)));
  assert.ok(s.hands[1].some((c) => c.id === id));
  s = round(s);
  assert.equal(
    s.hands[1].some((c) => c.id === id),
    false,
  );
});
test('extra deployment control is sampled once at turn start', () => {
  let s = fixture();
  add(s, 1, 1, 1, 10);
  add(s, 1, 1, 3, 10);
  assert.equal(s.deployRows[1].includes(10), false);
  s = round(s);
  assert.ok(s.deployRows[1].includes(10));
  s.units.pop();
  assert.ok(s.deployRows[1].includes(10));
  s = round(s);
  assert.equal(s.deployRows[1].includes(10), false);
});
test('no-place minions are discarded instead of deadlocking turn end', () => {
  const s = fixture();
  for (const p of ALL_CELLS)
    if (p.y <= 8 && !(p.x === 5 && p.y === 1)) add(s, 'grave', 1, p.x, p.y);
  card(s, 1);
  const n = applyCommand(s, { type: 'end' });
  assert.equal(n.active, 2);
  assert.ok(n.log.some((l) => l.includes('自动弃置')));
});
test('base loss ends game and rejects later combat', () => {
  const s = fixture(),
    u = add(s, 9, 1, 5, 10);
  s.bases[2] = 10;
  const n = applyCommand(s, { type: 'attack', unitId: u.id, targetId: 'base-2' });
  assert.equal(n.winner, 1);
  assert.ok(commandError(n, { type: 'end' }));
});
