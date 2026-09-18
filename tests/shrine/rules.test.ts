import test from 'node:test';
import assert from 'node:assert/strict';
import {
  applyCommand,
  createGame,
  definition,
  template,
  getStats,
  canPlace,
  asTarget,
  commandError,
  unitActions,
  createSession,
  parseSession,
  validState,
  type GameState,
  type Kind,
  type Landmark,
  type Player,
} from '../../src/engine';
import { addUnit, addEffect, resetUnit } from '../../src/engine/state';
import { damage, heal, kill, resolution } from '../../src/engine/combat';
import {
  captureClockFrame,
  demolish,
  rebuildLandmarks,
  syncBanners,
  installEquipment,
} from '../../src/engine/shrines';
import { add, card, fixture, pass, round, strike, unit } from '../helpers';
function land(s: GameState, kind: Kind = 's1', p: Player = 1, x = 3, y = 6) {
  return addUnit(s, kind, p, { x, y }) as Landmark;
}

test('3.0 shrine draft has three distinct offers per side, atomic commitments, reveal then sequential optional setup', () => {
  let s = createGame(90, 'shrine');
  assert.equal(s.ply, 0);
  assert.equal(s.phase, 'shrine-draft');
  for (const p of [1, 2] as Player[]) assert.equal(new Set(s.shrineDraft!.offers[p]).size, 3);
  const snapshot = structuredClone(s);
  assert.throws(() => applyCommand(s, { type: 'summon' }));
  assert.deepEqual(s, snapshot);
  s = applyCommand(s, { type: 'choose-shrine', player: 2, shrineKind: 's13' });
  assert.equal(s.shrineDraft!.revealed, false);
  assert.equal(s.hands[2].length, 0);
  assert.throws(() => applyCommand(s, { type: 'choose-shrine', player: 2, shrineKind: 's13' }));
  s = applyCommand(s, { type: 'choose-shrine', player: 1, shrineKind: 's8' });
  assert.equal(s.phase, 'shrine-setup');
  assert.equal(s.active, 1);
  assert.equal(s.hands[1][0].kind, 's8');
  assert.equal(s.hands[2][0].kind, 's13');
  s = applyCommand(s, { type: 'finish-shrine-setup' });
  assert.equal(s.active, 2);
  s = applyCommand(s, { type: 'activate-aura', cardId: s.hands[2][0].id });
  s = applyCommand(s, { type: 'finish-shrine-setup' });
  assert.equal(s.ply, 1);
  assert.equal(s.summonSlots, 2);
  assert.equal(s.hands[1][0].kind, 's8');
  assert.ok(validState(s));
  assert.deepEqual(parseSession(JSON.stringify(createSession(s))).present, s);
});

test('3.0 free ultimate, separately paid extras, and classic summon fees do not mix', () => {
  let s = createGame(90, 'shrine');
  for (const p of [1, 2] as Player[])
    s = applyCommand(s, {
      type: 'choose-shrine',
      player: p,
      shrineKind: s.shrineDraft!.offers[p][0],
    });
  s = applyCommand(applyCommand(s, { type: 'finish-shrine-setup' }), {
    type: 'finish-shrine-setup',
  });
  s.heads[1] = 5;
  s = applyCommand(s, { type: 'summon' });
  assert.equal(s.heads[1], 5);
  assert.equal(s.summonSlots, 1);
  assert.equal(s.hands[1].at(-1)!.summonPool, 'ultimate');
  s = applyCommand(s, { type: 'extra-summon', ultimate: false });
  assert.equal(s.heads[1], 3);
  assert.equal(s.summonSlots, 1);
  assert.equal(s.hands[1].at(-1)!.summonPool, 'normal');
  s = applyCommand(s, { type: 'extra-summon' });
  assert.equal(s.heads[1], 0);
  assert.equal(s.summonSlots, 1);
  const old = structuredClone(s);
  assert.throws(() => applyCommand(s, { type: 'extra-summon' }));
  assert.deepEqual(s, old);
  s = applyCommand(applyCommand(s, { type: 'summon' }), { type: 'begin' });
  assert.ok(commandError(s, { type: 'extra-summon' }));
  const classic = createGame(90);
  assert.ok(commandError(classic, { type: 'summon', ultimate: true }));
});

test('3.0 landmark coordinates are column-first, own one occupant, and distinguish live/dormant enemy deployment', () => {
  const s = fixture();
  for (const y of [8, 9, 10])
    assert.ok(canPlace(s, template('s8', 1, 0, { x: 7, y }), { x: 7, y }, true));
  for (const p of [
    { x: 8, y: 7 },
    { x: 9, y: 7 },
    { x: 6, y: 7 },
  ])
    assert.equal(canPlace(s, template('s8', 1, 0, p), p, true), false);
  const l = land(s),
    friendly = template(1, 1, 0, l),
    enemy = template(1, 2, 0, l);
  assert.ok(canPlace(s, friendly, l, true));
  assert.equal(canPlace(s, enemy, l, true), false);
  assert.ok(canPlace(s, enemy, l, false));
  const resident = add(s, 1, 1, l.x, l.y);
  assert.equal(canPlace(s, template('u25', 1, 0, l), l, true), false);
  s.units = [];
  demolish(s, l);
  assert.ok(canPlace(s, enemy, l, true));
  assert.equal(l.rebuildTicks, 0);
  assert.equal(s.deaths.length, 0);
  assert.deepEqual(s.heads, { 1: 6, 2: 0 });
});

test('3.0 attacks hit the occupant first, while damage spells pierce both layers without duplicate hits', () => {
  let s = fixture();
  const l = land(s, 's1', 2),
    v = add(s, 1, 2, 3, 6),
    u = add(s, 9, 1, 3, 4);
  assert.ok(commandError(s, { type: 'attack', unitId: u.id, targetId: l.id }));
  s = strike(s, u, v);
  assert.equal(s.landmarks![0].hp, 20);
  assert.equal(unit(s, v.id).hp, v.maxHp - 10);
  const bomb = card(s, 8);
  s = applyCommand(s, { type: 'cast', cardId: bomb, x: 3, y: 6 });
  assert.equal(s.landmarks![0].hp, 0);
  assert.equal(unit(s, v.id).hp, v.maxHp - 30);
  assert.equal(s.heads[1], 6);
  assert.ok(validState(s));
});

test('3.0 zero-range landmarks can only attack a hostile occupant and rebuild starts with next own turn', () => {
  let s = fixture();
  const l = land(s),
    v = add(s, 1, 2, 3, 6),
    away = add(s, 1, 2, 3, 7);
  assert.ok(commandError(s, { type: 'attack', unitId: l.id, targetId: away.id }));
  s = applyCommand(s, { type: 'attack', unitId: l.id, targetId: v.id });
  assert.equal(unit(s, v.id).hp, 10);
  const live = s.landmarks![0];
  demolish(s, live);
  rebuildLandmarks(s);
  assert.equal(live.rebuildTicks, 0, 'same ply never counts');
  for (let i = 0; i < 5; i++) {
    s.ply += 2;
    rebuildLandmarks(s);
  }
  assert.equal(live.hp, 0, 'enemy blocks reconstruction after countdown');
  const blocker = unit(s, v.id);
  blocker.owner = 1;
  addEffect(s, blocker, 'freeze', 2, 0, 2);
  s.ply += 2; // Renew freeze to test neutral occupancy, not already-expired freeze.
  blocker.effects = [];
  addEffect(s, blocker, 'freeze', 2, 0, 2);
  rebuildLandmarks(s);
  assert.equal(live.hp, 0);
  blocker.effects = [];
  s.ply += 2;
  rebuildLandmarks(s);
  assert.equal(live.hp, 20);
  assert.equal(live.dormantSince, undefined);
});

test('3.0 Jinye applies horn on deployment; an already charging follower gets two complete operations', () => {
  for (const [kind, charge, limit] of [
    [1, false, 1],
    [1, true, 2],
    ['u1', false, 2],
  ] as const) {
    let s = fixture();
    land(s);
    const id = card(s, kind);
    s = applyCommand(s, { type: 'deploy', cardId: id, x: 3, y: 6, charge });
    const u = s.units[0];
    assert.equal(u.chargedOnDeploy, true);
    assert.equal(u.offset, 2);
    assert.equal(getStats(s, u).sleeping, false);
    assert.equal(getStats(s, u).operationLimit, limit);
    assert.equal(u.hp, definition(kind).health - (charge ? 10 : 0));
  }
});

test('3.0 flag removal retracts maximum health without damaging current health and excludes forbidden attack recipients', () => {
  const s = fixture(),
    v = add(s, 1, 1, 2, 3),
    yy = add(s, 's7', 1, 2, 4),
    cat = add(s, 10, 1, 2, 5);
  const l = land(s, 's8', 1, 7, 9);
  assert.equal(v.hp, 60);
  assert.equal(v.maxHp, 60);
  assert.equal(l.maxHp, 40);
  assert.equal(getStats(s, v).attack, definition(1).attack + 10);
  for (const u of [yy, cat]) assert.equal(getStats(s, u).attack, definition(u.kind).attack);
  demolish(s, l);
  assert.equal(v.hp, 60);
  assert.equal(v.maxHp, 50);
  assert.equal(getStats(s, v).attack, definition(1).attack);
  assert.ok(validState(s));
  assert.deepEqual(parseSession(JSON.stringify(createSession(s))).present, s);
});

test('3.0 CX consumes independent 1/3 and 1/4 boundaries, cannot convert shrines or undamaged enemies', () => {
  for (const convert of [true, false])
    for (const extra of [true, false]) {
      const s = fixture(),
        u = add(s, 's3', 1, 3, 4),
        v = add(s, 1, 2, 3, 5),
        rolls = [convert ? 0.1 : 0.8, extra ? 0.1 : 0.8],
        cuts: number[][] = [];
      const n = applyCommand(s, { type: 'attack', unitId: u.id, targetId: v.id }, (b) => {
        cuts.push([...b]);
        return rolls.shift()!;
      });
      assert.equal(unit(n, v.id).owner, convert ? 1 : 2);
      assert.equal(n.summonSlots, extra ? 1 : 0);
      assert.deepEqual(cuts, [
        [0, 1 / 3, 1],
        [0, 1 / 4, 1],
      ]);
    }
  for (const immune of [false, true]) {
    const s = fixture(),
      u = add(s, 's3', 1, 3, 4),
      v = add(s, immune ? 1 : 's12', 2, 3, 5);
    if (immune) addEffect(s, v, 'immune', 2, 0, 2);
    const n = applyCommand(s, { type: 'attack', unitId: u.id, targetId: v.id }, () => 0.1);
    assert.equal(unit(n, v.id).owner, 2);
  }
});

test('3.0 Fusion and reflected shrine expose explicit healing/damage; friendly kill counts for attacker; fusion end heals', () => {
  for (const kind of ['s4', 's6'] as Kind[]) {
    let s = fixture();
    const u = add(s, kind, 1, 3, 4),
      v = add(s, 1, 1, 3, 5);
    v.hp = 5;
    const n = applyCommand(s, { type: 'attack', unitId: u.id, targetId: v.id, mode: 'heal' });
    assert.equal(unit(n, v.id).hp, 5 + definition(kind).attack);
    s = applyCommand(s, { type: 'attack', unitId: u.id, targetId: v.id, mode: 'damage' });
    assert.equal(
      s.units.some((x) => x.id === v.id),
      false,
    );
    assert.equal(s.heads[1], 7);
    assert.equal(s.heads[2], 0);
  }
  const s = fixture(),
    u = add(s, 's4', 1, 3, 4),
    v = add(s, 1, 1, 3, 5),
    away = add(s, 1, 1, 9, 12);
  u.hp = 10;
  v.hp = 5;
  away.hp = 5;
  const n = pass(s);
  assert.equal(unit(n, u.id).hp, 60);
  assert.equal(unit(n, v.id).hp, 50);
  assert.equal(unit(n, away.id).hp, 5);
});

test('3.0 reflection records actual damage across two global plies, excludes older and overkill damage', () => {
  const s = fixture(),
    u = add(s, 's6', 1, 3, 4),
    v = add(s, 5, 2, 3, 6);
  damage(s, asTarget(u), 15, { owner: 2, kind: 'attack' });
  s.ply++;
  damage(s, asTarget(u), 20, { owner: 2, kind: 'attack' });
  s.active = 1;
  const n = strike(s, u, v);
  assert.equal(unit(n, v.id).hp, 111 - 10 - 15 - 20);
  s.ply += 2;
  resetUnit(s, u);
  const old = strike(s, u, v);
  assert.equal(unit(old, v.id).hp, 101);
});

test('3.0 strong seizure transfers equipment identity and genuinely exposes acquired active skills', () => {
  let s = fixture();
  const u = add(s, 's5', 1, 3, 4),
    v = add(s, 6, 2, 3, 5);
  installEquipment(v, 's2', 'unique-iron');
  v.hp = 5;
  s = strike(s, u, v);
  const stolen = unit(s, u.id);
  assert.ok(stolen.traits?.includes(6));
  assert.deepEqual(stolen.equipment, ['s2']);
  assert.equal(stolen.equipmentIds?.s2, 'unique-iron');
  assert.equal(s.hands[2].length, 0);
  s = round(s);
  const action = unitActions(s, unit(s, u.id)).find(
    (a) => a.command.type === 'skill' && a.command.ability === 6,
  );
  assert.ok(action);
  assert.equal(commandError(s, action.command), null);
  assert.ok(validState(applyCommand(s, action.command)));
});

test('3.0 YYF and normal10 reject weapons and allied unit attack buffs in every rules mode', () => {
  for (const kind of [10, 's7'] as Kind[]) {
    const s = fixture(),
      u = add(s, kind, 1, 3, 4),
      booster = add(s, 6, 1, 3, 5),
      id = card(s, 's2');
    assert.ok(commandError(s, { type: 'equip', cardId: id, targetId: u.id }));
    const n = applyCommand(s, { type: 'skill', unitId: booster.id });
    assert.equal(getStats(n, unit(n, u.id)).attack, definition(kind).attack);
  }
});

test('3.0 Jade suicide uses selected ordinal and 5-point ceiling, gives no suicide head', () => {
  const s = fixture();
  s.auras = { 1: [{ kind: 's9', parity: 'odd' }], 2: [] };
  const u = add(s, 9, 1, 3, 4),
    v = add(s, 5, 2, 3, 6),
    even = add(s, 2, 1, 2, 4);
  u.hp = 12;
  assert.ok(commandError(s, { type: 'shatter', unitId: even.id, targetId: v.id }));
  const n = applyCommand(s, { type: 'shatter', unitId: u.id, targetId: v.id });
  assert.equal(
    n.units.some((v) => v.id === u.id),
    false,
  );
  assert.equal(unit(n, v.id).hp, 96);
  assert.deepEqual(n.heads, s.heads);
});

test('3.0 Clock restores previous-own-start life, position, operations and once-use state, rejects collision/shrine atomically', () => {
  let s = fixture();
  s.auras = { 1: [{ kind: 's10' }], 2: [] };
  const u = add(s, 'u6', 1, 3, 4);
  captureClockFrame(s);
  s.ply += 2;
  s.turns[1]++;
  captureClockFrame(s);
  u.hp = 5;
  u.onceUsed = true;
  u.operations = 1;
  u.x = 4;
  const blocker = add(s, 1, 1, 3, 4);
  assert.ok(commandError(s, { type: 'clock', targetId: u.id }));
  assert.equal(s.auras[1][0].usedPly, undefined);
  s.units = s.units.filter((v) => v.id !== blocker.id);
  s = applyCommand(s, { type: 'clock', targetId: u.id });
  const back = unit(s, u.id);
  assert.equal(back.hp, 45);
  assert.equal(back.x, 3);
  assert.equal(back.onceUsed, false);
  assert.equal(back.operations, 0);
  assert.ok(commandError(s, { type: 'clock', targetId: u.id }));
  assert.ok(validState(s));
  const fresh = fixture();
  fresh.auras = { 1: [{ kind: 's10' }], 2: [] };
  const shrine = add(fresh, 's6', 2, 3, 5);
  assert.ok(commandError(fresh, { type: 'clock', targetId: shrine.id }));
  const enemy = add(fresh, 1, 2, 4, 5);
  enemy.attackBonus = 50;
  installEquipment(enemy, 's2', 'iron');
  enemy.hp = 1;
  const n = applyCommand(fresh, { type: 'clock', targetId: enemy.id });
  assert.equal(unit(n, enemy.id).hp, 50);
  assert.equal(unit(n, enemy.id).attackBonus, 0);
  assert.deepEqual(unit(n, enemy.id).equipment, []);
});

test('3.0 Clock cannot duplicate a returned physical shrine weapon', () => {
  const s = fixture();
  s.auras = { 1: [{ kind: 's10' }], 2: [] };
  const u = add(s, 1, 1, 3, 4);
  installEquipment(u, 's2', 'iron');
  captureClockFrame(s);
  s.ply += 2;
  s.turns[1]++;
  captureClockFrame(s);
  u.equipment = [];
  u.equipmentIds = {};
  s.hands[1].push({ id: 'iron', kind: 's2', drawnAt: 3, summonedPly: s.ply });
  assert.ok(commandError(s, { type: 'clock', targetId: u.id }));
  assert.equal(s.auras[1][0].usedPly, undefined);
});

test('3.0 strong kill and mana fountain modify correct damage packets and block healing without buffing fixed heals', () => {
  const s = fixture();
  s.auras = { 1: [{ kind: 's11' }], 2: [] };
  const fountain = add(s, 's14', 1, 3, 4),
    v = add(s, 5, 2, 3, 6),
    friend = add(s, 1, 1, 4, 4);
  friend.hp = 5;
  fountain.attackBonus = 99;
  const bomb = card(s, 8);
  let n = applyCommand(s, { type: 'cast', cardId: bomb, x: 3, y: 6 });
  assert.equal(unit(n, v.id).hp, 71, '20+15+5');
  heal(n, unit(n, v.id), 50, resolution());
  assert.equal(unit(n, v.id).hp, 71);
  n = applyCommand(n, { type: 'attack', unitId: fountain.id, targetId: friend.id });
  assert.equal(unit(n, friend.id).hp, 25);
  assert.equal(getStats(n, unit(n, fountain.id)).attack, -20);
});

test('3.0 first strike refreshes a complete action with exact 3/5 split', () => {
  for (const [roll, left] of [
    [0.59, 1],
    [0.6, 0],
  ]) {
    const s = fixture(),
      u = add(s, 's12', 1, 3, 4),
      v = add(s, 5, 2, 3, 5);
    const n = applyCommand(s, { type: 'attack', unitId: u.id, targetId: v.id }, (b) => {
      assert.deepEqual(b, [0, 3 / 5, 1]);
      return roll;
    });
    assert.equal(getStats(n, unit(n, u.id)).operationsLeft, left);
    assert.equal(unit(n, u.id).mode, 'none');
  }
});

test('3.0 Old K chooses two full outcome groups from 3/4 and never changes paid extra summons', () => {
  for (const [roll, count] of [
    [0.1, 4],
    [0.9, 3],
  ]) {
    const s = fixture();
    s.mode = 'shrine';
    s.phase = 'summon';
    s.summonSlots = 2;
    s.regularSummons = 2;
    s.auras = { 1: [{ kind: 's13' }], 2: [] };
    let n = applyCommand(s, { type: 'summon' }, () => roll);
    assert.equal(n.summonOffer!.groups.length, count);
    assert.equal(n.summonSlots, 0);
    assert.equal(n.hands[1].length, 0);
    assert.ok(commandError(n, { type: 'begin' }));
    assert.ok(commandError(n, { type: 'choose-summons', offerIndices: [0, 0] }));
    const retained = n.summonOffer!.groups.slice(0, 2).flat();
    n = applyCommand(n, { type: 'choose-summons', offerIndices: [0, 1] });
    assert.deepEqual(n.hands[1], retained);
    n = applyCommand(n, { type: 'extra-summon' });
    assert.equal(n.summonOffer, undefined);
    assert.equal(n.summonSlots, 0);
    assert.equal(n.heads[1], 3);
  }
});

test('3.0 iron returns on death; blade needs a kill while held and uses pre-hit missing life', () => {
  let s = fixture();
  const u = add(s, 1, 1, 3, 4),
    id = card(s, 's2');
  s = applyCommand(s, { type: 'equip', cardId: id, targetId: u.id });
  assert.equal(unit(s, u.id).maxHp, 70);
  assert.equal(unit(s, u.id).hp, 70);
  kill(s, unit(s, u.id), { owner: 2, kind: 'attack' });
  assert.equal(s.hands[1][0].id, id);
  for (const qualified of [false, true]) {
    let s = fixture();
    const u = add(s, 9, 1, 3, 4),
      v = add(s, 1, 2, 3, 5),
      id = card(s, 's15');
    v.hp = 10;
    s = applyCommand(s, { type: 'equip', cardId: id, targetId: u.id });
    assert.equal(unit(s, u.id).maxHp, definition(9).health - 5);
    if (qualified) s = strike(s, unit(s, u.id), unit(s, v.id));
    kill(s, unit(s, u.id), { owner: 2, kind: 'attack' });
    assert.equal(
      s.hands[1].some((c) => c.id === id),
      qualified,
    );
  }
  const t = fixture(),
    a = add(t, 9, 1, 3, 4),
    b = add(t, 5, 2, 3, 5);
  b.hp = 80;
  installEquipment(a, 's15');
  const n = strike(t, a, b);
  assert.equal(unit(n, b.id).hp, 19, '10+20 plus 111-80');
});

test('3.0 walnut staff is mage-only, adds range, and -15 attack expires at next own start', () => {
  let s = fixture();
  const mage = add(s, 's14', 1, 3, 4),
    friend = add(s, 1, 1, 3, 5),
    id = card(s, 's16');
  friend.hp = 10;
  assert.ok(commandError(s, { type: 'equip', cardId: id, targetId: friend.id }));
  s = applyCommand(s, { type: 'equip', cardId: id, targetId: mage.id });
  assert.equal(getStats(s, unit(s, mage.id)).range, 6);
  s = strike(s, unit(s, mage.id), unit(s, friend.id));
  assert.equal(unit(s, friend.id).hp, 30);
  assert.equal(getStats(s, unit(s, friend.id)).attack, 5);
  s = round(s);
  assert.equal(getStats(s, unit(s, friend.id)).attack, definition(1).attack);
});

test('3.0 three on-board U13 synthesize persistent Laoqian, choose correct pool once/turn without free fees', () => {
  let s = fixture();
  const ids = [2, 3, 4].map((x) => add(s, 'u13', 1, x, 3).id);
  s.phase = 'synthesis';
  s.summonSlots = 2;
  s = applyCommand(s, { type: 'synthesize', recipeId: 'laoqian', materialIds: ids });
  assert.equal(s.units.length, 0);
  assert.equal(s.auras![1][0].kind, 'laoqian');
  assert.equal(s.deaths.length, 0);
  assert.equal(s.heads[1], 6);
  const old = structuredClone(s);
  assert.throws(() => applyCommand(s, { type: 'summon', chosenKind: 'u1' }));
  assert.deepEqual(s, old);
  s = applyCommand(s, { type: 'summon', ultimate: true, chosenKind: 'u25' });
  assert.equal(s.heads[1], 4);
  assert.equal(s.hands[1].length, 8);
  assert.ok(s.hands[1].every((c) => c.kind === 'u25'));
  assert.ok(commandError(s, { type: 'summon', chosenKind: 1 }));
  assert.ok(validState(s));
});
