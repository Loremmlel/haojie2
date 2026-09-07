import test from 'node:test';
import assert from 'node:assert/strict';
import {
  applyCommand,
  asTarget,
  attackPath,
  canPlace,
  commandError,
  createGame,
  createSession,
  definition,
  getStats,
  has,
  parseSession,
  template,
  allegiance,
} from '../../src/engine';
import { addEffect, draw, resetUnit } from '../../src/engine/state';
import { kill, damage, resolution } from '../../src/engine/combat';
import { add, card, fixture, pass, round, seedFor, strike, unit } from '../helpers';

test('head spending replaces an unrevealed summon, never pays after seeing its result', () => {
  let s = createGame(7);
  s.heads[1] = 2;
  s = applyCommand(s, { type: 'summon', ultimate: true });
  assert.equal(s.heads[1], 0);
  assert.equal(s.summonSlots, 1);
  assert.ok(s.hands[1].every((c) => definition(c.kind).tier !== 'normal'));
  assert.ok(commandError(s, { type: 'summon', ultimate: true }));
  s = applyCommand(s, { type: 'summon' });
  assert.equal(s.summonSlots, 0);
  assert.ok(commandError(s, { type: 'summon' }));
});
test('spell kills retain owner attribution and grant a head', () => {
  const s = fixture(),
    v = add(s, 1, 2, 3, 5),
    id = card(s, 8);
  v.hp = 20;
  const n = applyCommand(s, { type: 'cast', cardId: id, x: 3, y: 5 });
  assert.equal(n.heads[1], 7);
});
test('U01 innate charge and mutually exclusive double/100/normal attack rolls', () => {
  for (const [low, high, amount] of [
    [0, 0.2, 100],
    [0.2, 0.2 + 1 / 3, 60],
    [0.2 + 1 / 3, 1, 30],
  ]) {
    const s = fixture(),
      u = add(s, 'u1', 1, 3, 4),
      v = add(s, 5, 2, 3, 6);
    s.rng = seedFor(low, high);
    assert.equal(unit(strike(s, u, v), v.id).hp, 111 - amount);
  }
  const s = fixture(),
    id = card(s, 'u1'),
    n = applyCommand(s, { type: 'deploy', cardId: id, x: 2, y: 4 });
  assert.equal(getStats(n, n.units[0]).operationsLeft, 1);
});
test('U02 attack charge adds15 per layer, survives movement, resets on hit', () => {
  let s = fixture();
  const u = add(s, 'u2', 1, 3, 4),
    v = add(s, 5, 2, 3, 7);
  s = applyCommand(s, { type: 'charge', unitId: u.id, mode: 'attack' });
  s = round(s);
  s = applyCommand(s, { type: 'move', unitId: u.id, x: 3, y: 5 });
  s = round(s);
  assert.equal(getStats(s, unit(s, u.id)).attack, 30);
  s = strike(s, unit(s, u.id), unit(s, v.id));
  assert.equal(unit(s, v.id).hp, 81);
  assert.equal(unit(s, u.id).charge, 0);
});
test('U03 counterspell consumes the spell but does not apply its effect', () => {
  const s = fixture(),
    u = add(s, 'u3', 2, 8, 12),
    target = add(s, 1, 1, 3, 4),
    id = card(s, 17);
  s.rng = seedFor(0, 1 / 3);
  const n = applyCommand(s, { type: 'cast', cardId: id, targetId: target.id });
  assert.equal(unit(n, target.id).effects.length, 0);
  assert.equal(n.hands[1].length, 0);
  assert.ok(n.events.some((e) => e.text === '法术反制'));
});
test('U04 removes original passives and stuns without destroying basic stats or equipment', () => {
  let s = fixture();
  const u = add(s, 'u4', 1, 3, 4),
    v = add(s, 11, 2, 3, 6);
  s = strike(s, u, v);
  assert.ok(unit(s, v.id).silenced);
  assert.ok(has(s, unit(s, v.id), 'stun'));
  assert.equal(getStats(s, unit(s, v.id)).attack, 20);
  kill(s, unit(s, v.id), { owner: 1, unit: u, kind: 'attack' });
  assert.equal(s.bonus[2], 0);
});
test('U05 only explicitly listed mages can equip a frost staff', () => {
  const s = fixture(),
    mage = add(s, 'u6', 1, 3, 4),
    notMage = add(s, 'u3', 1, 4, 4),
    id = card(s, 'u5');
  assert.ok(commandError(s, { type: 'equip', cardId: id, targetId: notMage.id }));
  const n = applyCommand(s, { type: 'equip', cardId: id, targetId: mage.id });
  assert.equal(getStats(n, unit(n, mage.id)).attack, 10);
  assert.equal(getStats(n, unit(n, mage.id)).range, 6);
});
test('frozen units are neutral blockers, attackable by both sides and cannot act', () => {
  let s = fixture();
  const mage = add(s, 'u6', 1, 2, 4),
    v = add(s, 1, 2, 3, 4),
    friend = add(s, 26, 2, 4, 4),
    id = card(s, 'u5');
  s = applyCommand(s, { type: 'equip', cardId: id, targetId: mage.id });
  s = strike(s, unit(s, mage.id), unit(s, v.id));
  assert.equal(allegiance(s, unit(s, v.id)), 0);
  assert.equal(getStats(s, unit(s, v.id)).operationsLeft, 0);
  assert.equal(attackPath(s, unit(s, friend.id), asTarget(unit(s, mage.id)), 2), null);
  s.active = 2;
  s = strike(s, unit(s, friend.id), unit(s, v.id));
  assert.equal(unit(s, v.id).hp, 20);
});
test('U06 unarmed hit applies6-turn burn, staff replaces it with freeze and base10', () => {
  let s = fixture();
  const mage = add(s, 'u6', 1, 2, 4),
    v = add(s, 5, 2, 3, 5);
  s = strike(s, mage, v);
  assert.equal(unit(s, v.id).hp, 111);
  assert.ok(has(s, unit(s, v.id), 'burn'));
  s = pass(s);
  assert.equal(unit(s, v.id).hp, 106);
});
test('U06 charged cross hits center40 and arms20 exactly once per target', () => {
  const s = fixture(),
    u = add(s, 'u6', 1, 4, 4),
    center = add(s, 5, 2, 4, 6),
    arm = add(s, 1, 2, 6, 6);
  u.charge = u.readyCharge = 1;
  u.chargeType = 'skill';
  const n = applyCommand(s, { type: 'skill', unitId: u.id, x: 4, y: 6 });
  assert.equal(unit(n, center.id).hp, 71);
  assert.equal(unit(n, arm.id).hp, 30);
  assert.equal(unit(n, u.id).hp, 25);
  assert.ok(unit(n, u.id).onceUsed);
});
test('U07 giant transformation is free, permanent 2x2, costs the source and requires space', () => {
  let s = fixture();
  const u = add(s, 'u7', 1, 2, 3),
    v = add(s, 1, 1, 6, 5);
  u.operations = 1;
  s = applyCommand(s, { type: 'skill', unitId: u.id, targetId: v.id });
  assert.equal(unit(s, u.id).operations, 1);
  assert.equal(getStats(s, unit(s, u.id)).attack, 10);
  assert.equal(unit(s, u.id).hp, 80);
  assert.equal(unit(s, v.id).size, 2);
  assert.equal(unit(s, v.id).hp, 55);
  assert.ok(commandError(s, { type: 'skill', unitId: u.id, targetId: v.id }));
});
test('U08 basic lifesteal and crit growth are derived from current kills', () => {
  let s = fixture();
  const u = add(s, 'u8', 1, 3, 4),
    v = add(s, 5, 2, 3, 6);
  u.hp = 30;
  s.rng = seedFor(0.8, 1);
  s = strike(s, u, v);
  assert.equal(unit(s, u.id).hp, 32);
  assert.equal(unit(s, v.id).hp, 101);
  u.kills = 4;
  u.operations = 0;
  u.mode = 'none';
  u.shots = 0;
  v.hp = 1;
  const n = strike(fixtureWith(u, v), u, v);
  assert.equal(unit(n, u.id).kills, 5);
  assert.equal(getStats(n, unit(n, u.id)).range, 5);
});
function fixtureWith(...units: any[]) {
  const s = fixture();
  s.units = structuredClone(units);
  return s;
}
test('U09 flame storm hits now and again next own start at then-current positions', () => {
  let s = fixture();
  const u = add(s, 5, 2, 3, 7),
    id = card(s, 'u9');
  s = applyCommand(s, { type: 'cast', cardId: id, mode: 'row', row: 7 });
  assert.equal(unit(s, u.id).hp, 91);
  s = round(s);
  assert.equal(unit(s, u.id).hp, 71);
  assert.equal(s.hazards.length, 0);
});
test('U10 rage increments only for actual positive damage, including DOT', () => {
  const s = fixture(),
    u = add(s, 'u10', 2, 3, 5),
    ctx = resolution();
  damage(s, asTarget(u), 5, { owner: 1, kind: 'status' }, ctx);
  assert.equal(getStats(s, u).attack, 20);
  addEffect(s, u, 'immune', 2, 0, 2);
  damage(s, asTarget(u), 15, { owner: 1, kind: 'attack' }, ctx);
  assert.equal(getStats(s, u).attack, 20);
});
test('U11 kingslayer adds cap not instant heal, first attack drains100% once per turn', () => {
  let s = fixture();
  const u = add(s, 9, 1, 3, 4),
    a = add(s, 5, 2, 3, 6),
    b = add(s, 1, 2, 5, 4),
    id = card(s, 'u11');
  u.hp = 20;
  s = applyCommand(s, { type: 'equip', cardId: id, targetId: u.id });
  assert.equal(unit(s, u.id).hp, 20);
  assert.equal(unit(s, u.id).maxHp, 70);
  s = strike(s, unit(s, u.id), unit(s, a.id));
  assert.equal(unit(s, u.id).hp, 35);
  s = strike(s, unit(s, u.id), unit(s, b.id));
  assert.equal(unit(s, u.id).hp, 35);
});
test('U12 SZF collision does30 and offers free bounce without spending another movement', () => {
  let s = fixture();
  const u = add(s, 'u12', 1, 2, 4),
    v = add(s, 5, 2, 3, 4);
  u.charge = u.readyCharge = 1;
  u.chargeType = 'move';
  s = applyCommand(s, { type: 'move', unitId: u.id, x: 3, y: 4 });
  assert.equal(unit(s, v.id).hp, 81);
  assert.equal(s.pending[0].kind, 'bounce');
  assert.equal(unit(s, u.id).moves, 4);
  s = applyCommand(s, { type: 'react', x: 3, y: 3 });
  assert.equal(unit(s, u.id).moves, 4);
  assert.equal(unit(s, u.id).y, 3);
  assert.equal(s.pending.length, 0);
});
test('U12 bounce can chain through enemies, including bases', () => {
  let s = fixture();
  const u = add(s, 'u12', 1, 4, 12);
  u.charge = u.readyCharge = 1;
  u.chargeType = 'move';
  s = applyCommand(s, { type: 'move', unitId: u.id, x: 5, y: 12 });
  s = applyCommand(s, { type: 'move', unitId: u.id, x: 5, y: 13 });
  assert.equal(s.bases[2], 270);
  s = applyCommand(s, { type: 'react', x: 4, y: 13 });
  assert.equal(s.pending.length, 0);
  assert.equal(unit(s, u.id).moves, 3);
});
test('U12-prime cannot finish inside enemy and locks other operations during transit', () => {
  let s = fixture();
  const u = add(s, 'u12p', 1, 2, 4),
    v = add(s, 5, 2, 3, 4),
    friend = add(s, 9, 1, 7, 4);
  u.charge = u.readyCharge = 1;
  u.chargeType = 'move';
  s = applyCommand(s, { type: 'move', unitId: u.id, x: 3, y: 4 });
  assert.equal(unit(s, v.id).hp, 81);
  assert.ok(commandError(s, { type: 'end' }));
  assert.ok(commandError(s, { type: 'move', unitId: friend.id, x: 8, y: 4 }));
  s = applyCommand(s, { type: 'move', unitId: u.id, x: 3, y: 3 });
  s = applyCommand(s, { type: 'finish-mode', unitId: u.id });
  assert.equal(unit(s, u.id).mode, 'none');
});
test('U13 reroll consumes its own per-turn allowance and keeps summon pool', () => {
  let s = fixture();
  const u = add(s, 'u13', 1, 2, 4),
    id = card(s, 'u9');
  s = applyCommand(s, { type: 'reroll', unitId: u.id, cardId: id });
  assert.ok(s.hands[1].every((c) => definition(c.kind).tier !== 'normal'));
  assert.ok(commandError(s, { type: 'reroll', unitId: u.id, cardId: s.hands[1][0].id }));
});
test('U13 first-five-turn self-reroll is optional and cannot endlessly reroll itself', () => {
  const s = fixture(),
    id = card(s, 'u13'),
    n = applyCommand(s, { type: 'reroll', cardId: id });
  assert.ok(n.hands[1].every((c) => c.rerolled));
  s.turns[1] = 6;
  assert.ok(commandError(s, { type: 'reroll', cardId: id }));
});
test('U14 siphon allows base healing, is free and disconnects immediately out of range', () => {
  let s = fixture();
  const u = add(s, 'u14', 1, 5, 4),
    victim = add(s, 5, 2, 5, 7);
  s.bases[1] = 250;
  s = applyCommand(s, { type: 'skill', unitId: u.id, targetId: victim.id, secondId: 'base-1' });
  assert.equal(unit(s, u.id).operations, 0);
  assert.equal(s.siphons.length, 1);
  s = pass(s);
  assert.equal(unit(s, victim.id).hp, 91);
  assert.equal(s.bases[1], 270);
  unit(s, victim.id).y = 11;
  s.phase = 'play';
  s.summonSlots = 0;
  const friend = add(s, 1, 2, 8, 11);
  s = applyCommand(s, { type: 'move', unitId: friend.id, x: 8, y: 12 });
  assert.equal(s.siphons.length, 0);
});
test('U15 immunity tower spends15 max HP once per spell-target packet', () => {
  const s = fixture(),
    tower = add(s, 'u15', 2, 6, 7),
    victim = add(s, 1, 2, 4, 6),
    id = card(s, 8),
    n = applyCommand(s, { type: 'cast', cardId: id, x: 4, y: 6 });
  assert.equal(unit(n, victim.id).hp, 50);
  assert.equal(unit(n, tower.id).maxHp, 45);
});
test('U16 boots grant a complete attack operation after movement', () => {
  let s = fixture();
  const u = add(s, 9, 1, 3, 4),
    a = add(s, 5, 2, 3, 6),
    b = add(s, 1, 2, 5, 4),
    id = card(s, 'u16');
  s = applyCommand(s, { type: 'equip', cardId: id, targetId: u.id });
  s = applyCommand(s, { type: 'move', unitId: u.id, x: 2, y: 4 });
  s = strike(s, unit(s, u.id), unit(s, a.id));
  s = strike(s, unit(s, u.id), unit(s, b.id));
  assert.equal(unit(s, u.id).operations, 1);
  assert.equal(unit(s, u.id).bonusAttacks, 0);
  assert.ok(commandError(s, { type: 'move', unitId: u.id, x: 1, y: 4 }));
});
test('U17 advances exactly one unit, enabling fresh summon plus next-turn execute immediately', () => {
  let s = fixture();
  const id = card(s, 9),
    execute = card(s, 18),
    horn = card(s, 'u17'),
    enemy = add(s, 5, 2, 3, 6);
  s = applyCommand(s, { type: 'deploy', cardId: id, x: 3, y: 4 });
  const u = s.units.find((u) => u.kind === 9)!;
  s = applyCommand(s, { type: 'cast', cardId: execute, targetId: u.id });
  s = applyCommand(s, { type: 'cast', cardId: horn, targetId: u.id });
  assert.equal(s.ply, 5);
  assert.equal(s.turns[1], 3);
  assert.equal(getStats(s, unit(s, u.id)).remaining, 2);
  s = strike(s, unit(s, u.id), unit(s, enemy.id));
  assert.equal(
    s.units.some((u) => u.id === enemy.id),
    false,
  );
  assert.equal(definition('u17').spell, 8);
});
test('U18 ignores attacks <=10, retaliates after real damage, does not loop forever', () => {
  let s = fixture();
  const u = add(s, 9, 1, 3, 4),
    king = add(s, 'u18', 2, 3, 6);
  s = strike(s, u, king);
  assert.equal(unit(s, king.id).hp, 50);
  s = fixture();
  const a = add(s, 'u18', 1, 3, 4),
    b = add(s, 'u18', 2, 3, 6);
  s = strike(s, a, b);
  assert.ok(s.events.length < 30);
  assert.equal(unit(s, a.id).hp, 35);
  assert.equal(unit(s, b.id).hp, 35);
});
test('U19 revives a recent allied original with no equipment/buffs and spends20 cap', () => {
  let s = fixture();
  const mage = add(s, 'u19', 1, 3, 4),
    v = add(s, 1, 1, 5, 4);
  v.equipment = ['u11'];
  kill(s, v, { owner: 2, kind: 'spell' });
  s = round(s);
  const death = s.deaths[0];
  s = applyCommand(s, { type: 'skill', unitId: mage.id, deathId: death.id, x: 4, y: 4 });
  const revived = s.units.find((u) => u.kind === 1)!;
  assert.equal(revived.maxHp, 50);
  assert.deepEqual(revived.equipment, []);
  assert.equal(unit(s, mage.id).maxHp, 5);
  assert.ok(s.deaths[0].revived);
});
test('U20 knockback moves two squares, pushes one follower one square, and boundary inflicts30', () => {
  let s = fixture();
  const u = add(s, 'u20', 1, 3, 4),
    v = add(s, 1, 2, 3, 6);
  s = strike(s, u, v);
  assert.equal(unit(s, v.id).y, 8);
  s = fixture();
  const a = add(s, 'u20', 1, 3, 4),
    b = add(s, 1, 2, 3, 6),
    f = add(s, 1, 2, 3, 7);
  s = strike(s, a, b);
  assert.equal(unit(s, b.id).y, 7);
  assert.equal(unit(s, f.id).y, 8);
  s = fixture();
  const c = add(s, 'u20', 1, 3, 10),
    d = add(s, 1, 2, 3, 12);
  s = strike(s, c, d);
  assert.equal(unit(s, d.id).hp, 15);
  assert.equal(unit(s, d.id).y, 12);
});
test('U21 heals25 on ally hit and all nearby allies on death', () => {
  let s = fixture();
  const u = add(s, 'u21', 1, 3, 4),
    v = add(s, 1, 1, 4, 4);
  v.hp = 10;
  s = strike(s, u, v);
  assert.equal(unit(s, v.id).hp, 35);
  kill(s, unit(s, u.id), { owner: 2, kind: 'spell' });
  assert.equal(unit(s, v.id).hp, 50);
});
test('U22 hut queues a legal spawn, spends10 cap and does not duplicate the dead unit identity', () => {
  let s = fixture();
  const hut = add(s, 'u22', 1, 3, 4),
    v = add(s, 1, 1, 4, 4);
  kill(s, v, { owner: 2, kind: 'spell' });
  assert.equal(s.pending[0].kind, 'hut-spawn');
  s = applyCommand(s, { type: 'react', x: 4, y: 4 });
  const summoned = s.units.find((v) => v.kind === 20)!;
  assert.ok(summoned);
  assert.equal(unit(s, hut.id).maxHp, 40);
  assert.notEqual(summoned.id, v.id);
});
test('U23 after a kill, next turn can replace attacking with any-distance front hook', () => {
  let s = fixture();
  const u = add(s, 'u23', 1, 3, 4),
    victim = add(s, 1, 2, 3, 6),
    far = add(s, 1, 2, 8, 12);
  victim.hp = 10;
  s = strike(s, u, victim);
  assert.ok(commandError(s, { type: 'skill', unitId: u.id, targetId: far.id }));
  s = round(s);
  s = applyCommand(s, { type: 'skill', unitId: u.id, targetId: far.id });
  assert.deepEqual([unit(s, far.id).x, unit(s, far.id).y], [3, 5]);
  assert.ok(commandError(s, { type: 'attack', unitId: u.id, targetId: far.id }));
});
test('U24 marks its11x11, freezes next own start and does10 per end', () => {
  let s = fixture();
  const u = add(s, 'u24', 1, 4, 4),
    v = add(s, 1, 2, 4, 8);
  s = applyCommand(s, { type: 'skill', unitId: u.id, x: 4, y: 8 });
  assert.equal(has(s, unit(s, v.id), 'freeze'), false);
  s = round(s);
  assert.ok(has(s, unit(s, v.id), 'freeze'));
  s = pass(s);
  assert.equal(unit(s, v.id).hp, 40);
});
test('U25 eight clones stack and only last death grants one head and killer upgrade', () => {
  let s = fixture();
  const killer = add(s, 26, 1, 3, 4);
  for (let i = 0; i < 8; i++) {
    const v = add(s, 'u25', 2, 3, 6);
    v.group = 'eight';
  }
  for (let i = 0; i < 8; i++) {
    unit(s, killer.id).operations = 0;
    const victim = s.units.find((v) => v.kind === 'u25')!;
    s = strike(s, unit(s, killer.id), victim);
    assert.equal(s.heads[1], i === 7 ? 7 : 6);
  }
  assert.equal(unit(s, killer.id).kills, 1);
});
test('U25 targeted attacks cannot select under-stack unit, AoE hits all stack members', () => {
  const s = fixture(),
    u = add(s, 26, 1, 3, 4),
    a = add(s, 'u25', 2, 3, 6),
    b = add(s, 'u25', 2, 3, 6);
  a.group = b.group = 'g';
  assert.ok(commandError(s, { type: 'attack', unitId: u.id, targetId: b.id }));
  const id = card(s, 8),
    n = applyCommand(s, { type: 'cast', cardId: id, x: 3, y: 6 });
  assert.equal(n.units.filter((u) => u.kind === 'u25').length, 0);
  assert.equal(n.heads[1], 7);
});
test('U26 inner fire tracks current HP instead of snapshotting it', () => {
  let s = fixture();
  const u = add(s, 1, 1, 3, 4),
    id = card(s, 'u26');
  s = applyCommand(s, { type: 'cast', cardId: id, targetId: u.id });
  assert.equal(getStats(s, unit(s, u.id)).attack, 50);
  damage(s, asTarget(unit(s, u.id)), 7, { owner: 2, kind: 'status' });
  assert.equal(getStats(s, unit(s, u.id)).attack, 43);
});
test('U27 miner deploys in hostile rows, has two operations only on the next turn and base damage10', () => {
  let s = fixture();
  const id = card(s, 'u27');
  s = applyCommand(s, { type: 'deploy', cardId: id, x: 5, y: 11 });
  const u = s.units[0];
  s = round(s);
  s = applyCommand(s, { type: 'move', unitId: u.id, x: 4, y: 11 });
  s = applyCommand(s, { type: 'attack', unitId: u.id, targetId: 'base-2' });
  assert.equal(s.bases[2], 290);
  assert.equal(unit(s, u.id).operations, 2);
  s = round(s);
  assert.equal(getStats(s, unit(s, u.id)).operationLimit, 1);
});
test('U28 heart rejects mages and newborn charge units, grants HP and freeze immunity', () => {
  let s = fixture();
  const id = card(s, 'u28'),
    mage = add(s, 'u6', 1, 3, 4),
    charge = card(s, 'u1');
  assert.ok(commandError(s, { type: 'equip', cardId: id, targetId: mage.id }));
  s = applyCommand(s, { type: 'deploy', cardId: charge, x: 4, y: 4 });
  const u = s.units.find((v) => v.kind === 'u1')!;
  assert.ok(commandError(s, { type: 'equip', cardId: id, targetId: u.id }));
  s = round(s);
  s = applyCommand(s, { type: 'equip', cardId: id, targetId: u.id });
  assert.equal(unit(s, u.id).maxHp, 55);
});
test('U28 penetration hits enemies behind blockers along the selected cardinal line and burns', () => {
  let s = fixture();
  const u = add(s, 26, 1, 3, 3),
    a = add(s, 1, 2, 3, 4),
    b = add(s, 1, 2, 3, 6),
    id = card(s, 'u28');
  s = applyCommand(s, { type: 'equip', cardId: id, targetId: u.id });
  s = strike(s, unit(s, u.id), unit(s, b.id));
  assert.equal(unit(s, a.id).hp, 30);
  assert.equal(unit(s, b.id).hp, 30);
  assert.ok(has(s, unit(s, a.id), 'burn'));
  assert.ok(has(s, unit(s, b.id), 'burn'));
});
test('three unspent hearts craft a deployable firelord, without adding undefined Shantie content', () => {
  const s = fixture(),
    ids = [card(s, 'u28'), card(s, 'u28'), card(s, 'u28')],
    n = applyCommand(s, { type: 'craft', cardIds: ids });
  assert.equal(n.hands[1].length, 1);
  assert.equal(n.hands[1][0].kind, 'firelord');
  assert.ok(commandError(n, { type: 'end' }));
  assert.throws(() => definition('shantie' as any));
});
test('firelord targets highest HP and penetrates target stack, with10 splash', () => {
  const s = fixture(),
    lord = add(s, 'firelord', 1, 3, 4),
    a = add(s, 'u25', 2, 3, 6),
    b = add(s, 'u25', 2, 3, 6),
    splash = add(s, 1, 2, 4, 6);
  a.group = b.group = 'g';
  a.hp = a.maxHp = 100;
  b.hp = b.maxHp = 100;
  const n = pass(s);
  assert.equal(unit(n, a.id).hp, 20);
  assert.equal(unit(n, b.id).hp, 20);
  assert.equal(unit(n, splash.id).hp, 40);
  assert.ok(commandError(s, { type: 'attack', unitId: lord.id, targetId: a.id }));
});
test('every ultimate number can be reached from the 28-number pool', () => {
  const kinds = new Set();
  const s = fixture();
  for (let n = 0; n < 4000; n++) {
    s.hands[1] = [];
    for (const c of draw(s, 1, 1, true)) kinds.add(c.kind);
  }
  for (let n = 1; n <= 28; n++) assert.ok(kinds.has(`u${n}`), `U${n} must be reachable`);
  assert.ok(kinds.has('u12p'));
});
