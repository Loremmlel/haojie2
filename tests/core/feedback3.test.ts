import test from 'node:test';
import assert from 'node:assert/strict';
import {
  applyCommand,
  commandError,
  getStats,
  has,
  asTarget,
  attackPath,
  expansionAnchors,
  unitActions,
  cardActions,
  createSession,
  dispatch,
  undo,
  redo,
  parseSession,
  validState,
  type Command,
  type Kind,
  type GameState,
} from '../../src/engine';
import { addEffect, addUnit } from '../../src/engine/state';
import { installEquipment } from '../../src/engine/shrines';
import { startIntent, canChoose, advanceIntent, commandFor } from '../../src/ui/game/selection';
import { add, card, fixture, pass, round, seedFor, unit } from '../helpers';

function rejected(s: GameState, c: Command) {
  const before = structuredClone(s);
  assert.ok(commandError(s, c), JSON.stringify(c));
  assert.throws(() => applyCommand(s, c));
  assert.deepEqual(s, before, 'invalid commands must not pay costs, consume RNG or change history');
}
function readyRunner(s: GameState, kind: 'u12' | 'u12p', x = 2, y = 5) {
  const u = add(s, kind, 1, x, y);
  u.charge = u.readyCharge = 1;
  u.chargeType = 'move';
  return u;
}

for (const [x, y] of [
  [5, 6],
  [6, 6],
  [5, 7],
  [6, 7],
])
  test(`feedback3 giant: enemy can expand toward anchor (${x},${y}), no action cost even off-turn`, () => {
    const s = fixture(),
      bw = add(s, 'u7', 1, 2, 3),
      enemy = add(s, 1, 2, 6, 7);
    s.active = 2;
    s.phase = 'summon';
    s.summonSlots = 2;
    bw.operations = 1;
    assert.equal(expansionAnchors(s, enemy).length, 4);
    const c: Command = { type: 'skill', unitId: bw.id, targetId: enemy.id, x, y };
    const session = dispatch(createSession(s), c),
      n = session.present;
    assert.deepEqual([unit(n, enemy.id).x, unit(n, enemy.id).y, unit(n, enemy.id).size], [x, y, 2]);
    assert.equal(unit(n, enemy.id).hp, 55);
    assert.equal(unit(n, bw.id).hp, 80);
    assert.equal(unit(n, bw.id).operations, 1);
    assert.equal(n.summonSlots, 2);
    assert.ok(validState(n));
    assert.deepEqual(parseSession(JSON.stringify(session)).present, n);
    assert.deepEqual(undo(session).present, s);
    assert.deepEqual(redo(undo(session)).present, n);
    rejected(n, c);
  });

test('feedback3 giant: preview exposes only legal full footprints and rejects stacked/landmark/base/out-of-bounds growth', () => {
  const s = fixture(),
    bw = add(s, 'u7', 1, 1, 3),
    enemy = add(s, 1, 2, 6, 7);
  add(s, 1, 1, 5, 6);
  addUnit(s, 's1', 2, { x: 7, y: 7 });
  const c: Command = { type: 'skill', unitId: bw.id, targetId: enemy.id };
  assert.deepEqual(expansionAnchors(s, enemy), [{ x: 5, y: 7 }]);
  rejected(s, { ...c, x: 5, y: 6 });
  rejected(s, { ...c, x: 6, y: 7 });
  const a = unitActions(s, bw).find((a) => a.id === 'giant')!;
  let intent = startIntent(a);
  assert.ok(canChoose(s, intent, enemy));
  intent = advanceIntent(intent, enemy, s);
  assert.ok(canChoose(s, intent, { x: 5, y: 7 }));
  assert.equal(canChoose(s, intent, { x: 6, y: 7 }), false);
  assert.deepEqual(commandFor(s, intent, { x: 5, y: 7 }), { ...c, x: 5, y: 7 });
  const blocked = add(s, 1, 1, 5, 8);
  assert.equal(canChoose(s, startIntent(a), enemy), false);
  blocked.x = 3;
  blocked.y = 1;
  const nearBase = add(s, 1, 1, 4, 1);
  rejected(s, { ...c, targetId: nearBase.id, x: 4, y: 1 });
  rejected(s, { ...c, x: 9, y: 13 });
});

test('feedback3 base: friendly paths cross own base while identical opposing paths cannot', () => {
  const s = fixture(),
    attacker = add(s, 26, 1, 4, 1),
    enemy = add(s, 1, 2, 6, 1);
  assert.ok(attackPath(s, attacker, asTarget(enemy), 2));
  attacker.owner = 2;
  enemy.owner = 1;
  assert.equal(attackPath(s, attacker, asTarget(enemy), 2), null);
  assert.ok(attackPath(s, attacker, { id: 'base-1', owner: 1, x: 5, y: 1 }, 1));
});

test('feedback3 frozen counterspell and reroll passives stay enabled while active commands remain unavailable', () => {
  const s = fixture(),
    counter = add(s, 'u3', 2, 8, 10),
    victim = add(s, 1, 2, 3, 7);
  addEffect(s, counter, 'freeze', 1, 0, 2);
  s.rng = seedFor(0, 1 / 3);
  const n = applyCommand(s, { type: 'cast', cardId: card(s, 8), x: 3, y: 7 });
  assert.equal(unit(n, victim.id).hp, 50);
  assert.ok(n.events.some((e) => e.text === '法术反制'));
  n.active = 2;
  rejected(n, { type: 'move', unitId: counter.id, x: 8, y: 9 });
  const m = fixture(),
    mage = add(m, 'u13', 1, 3, 4);
  addEffect(m, mage, 'freeze', 2, 0, 2);
  card(m, 8);
  const reroll = cardActions(m, m.hands[1][0]).find(
    (a) => a.command.type === 'reroll' && a.command.unitId === mage.id,
  )!;
  assert.ok(reroll);
  assert.equal(commandError(m, reroll.command), null);
});

for (const kind of [2, 'u21', 'sage', 's4', 's6', 's14'] as Kind[])
  test(`feedback3 healer ${kind}: can heal itself through explicit UI command`, () => {
    const s = fixture(),
      u = add(s, kind, 1, 3, 5);
    u.hp = 1;
    const action = unitActions(s, u).find((a) => a.id === 'self-heal');
    assert.ok(action);
    const n = applyCommand(s, action.command);
    assert.ok(unit(n, u.id).hp > 1);
    assert.ok(unit(n, u.id).operations + unit(n, u.id).shots > 0);
  });

test('feedback3 healer: self-heal works underneath another legal stacked piece with Heart equipped', () => {
  const s = fixture(),
    healer = add(s, 'u25', 1, 4, 5),
    top = add(s, 'u25', 1, 4, 5);
  healer.traits = [2];
  healer.group = top.group = 'healer-stack';
  installEquipment(healer, 'u28');
  healer.hp = 1;
  const n = applyCommand(s, {
    type: 'attack',
    unitId: healer.id,
    targetId: healer.id,
    mode: 'heal',
  });
  assert.ok(unit(n, healer.id).hp > 1);
  assert.equal(unit(n, top.id).hp, top.hp);
});

for (const [x, y, expected] of [
  [3, 5, 80],
  [2, 5, 40],
  [2, 4, 20],
])
  test(`feedback3 bomb: 2x2 coverage at (${x},${y}) deals ${expected}, once per cell not per identity`, () => {
    const s = fixture(),
      target = add(s, 5, 2, 3, 5);
    const n = applyCommand(s, { type: 'cast', cardId: card(s, 8), x, y });
    assert.equal(unit(n, target.id).hp, 111 - expected);
  });

test('feedback3 spell: full hostile stack and landmark are hit, friendly stack and base remain safe', () => {
  const s = fixture(),
    a = add(s, 'u25', 2, 3, 7),
    b = add(s, 'u25', 2, 3, 7);
  a.group = b.group = 'enemy-stack';
  a.hp = a.maxHp = b.hp = b.maxHp = 50;
  const land = addUnit(s, 's1', 2, { x: 3, y: 7 }),
    friend = add(s, 1, 1, 4, 7);
  let n = applyCommand(s, { type: 'cast', cardId: card(s, 8), x: 3, y: 7 });
  assert.equal(unit(n, a.id).hp, 30);
  assert.equal(unit(n, b.id).hp, 30);
  assert.equal(n.landmarks!.find((u) => u.id === land.id)!.hp, land.hp - 20);
  assert.equal(unit(n, friend.id).hp, 50);
  n = applyCommand(n, { type: 'cast', cardId: card(n, 8), x: 5, y: 1 });
  assert.equal(n.bases[1], 300);
});

test('feedback3 spell: reflection goes to casting base and spell kills preserve head attribution', () => {
  const s = fixture(),
    reflect = add(s, 'slayer', 2, 3, 6),
    victim = add(s, 1, 2, 4, 6);
  victim.hp = 20;
  const n = applyCommand(s, { type: 'cast', cardId: card(s, 8), x: 3, y: 6 });
  assert.equal(n.bases[1], 290);
  assert.equal(n.heads[1], 7);
  assert.equal(
    n.units.some((u) => u.id === victim.id),
    false,
  );
  assert.equal(unit(n, reflect.id).hp, reflect.hp - 20);
});

test('feedback3 storm: both immediate and delayed packets spare friends and hit both big cells', () => {
  const s = fixture(),
    big = add(s, 5, 2, 3, 6),
    friend = add(s, 1, 1, 6, 6);
  let n = applyCommand(s, { type: 'cast', cardId: card(s, 'u9'), row: 6, mode: 'row' });
  assert.equal(unit(n, big.id).hp, 71);
  assert.equal(unit(n, friend.id).hp, 50);
  n = round(n);
  assert.equal(unit(n, big.id).hp, 31);
  assert.equal(unit(n, friend.id).hp, 50);
});

test('feedback3 large body: steps use small cells, validate full footprint and charge operation only once', () => {
  const s = fixture(),
    big = add(s, 'u4', 1, 3, 5);
  big.size = 2;
  rejected(s, { type: 'move', unitId: big.id, x: 5, y: 5 });
  let n = applyCommand(s, { type: 'move', unitId: big.id, x: 4, y: 5 });
  assert.equal(unit(n, big.id).operations, 0);
  assert.equal(unit(n, big.id).moves, 1);
  n = applyCommand(n, { type: 'move', unitId: big.id, x: 5, y: 5 });
  assert.equal(unit(n, big.id).operations, 1);
  assert.equal(unit(n, big.id).mode, 'none');
  add(s, 1, 2, 5, 6);
  rejected(s, { type: 'move', unitId: big.id, x: 4, y: 5 });
  s.units.pop();
  addUnit(s, 's1', 1, { x: 5, y: 6 });
  rejected(s, { type: 'move', unitId: big.id, x: 4, y: 5 });
});

for (const kind of ['u12p', 'u12'] as const)
  test(`feedback3 runner ${kind}: traverses equipment, friendly stack and occupied landmark; cannot stop inside`, () => {
    let s = fixture();
    const runner = readyRunner(s, kind, 2, 7),
      a = add(s, 'u25', 1, 3, 7),
      b = add(s, 'u25', 1, 3, 7);
    a.group = b.group = 'pass-stack';
    installEquipment(a, 'u28');
    addUnit(s, 's1', 1, { x: 4, y: 7 });
    const resident = add(s, 1, 1, 4, 7);
    s = applyCommand(s, { type: 'move', unitId: runner.id, x: 3, y: 7 });
    assert.ok(validState(s));
    assert.deepEqual(parseSession(JSON.stringify(createSession(s))).present, s);
    rejected(s, { type: 'finish-mode', unitId: runner.id });
    rejected(s, { type: 'end' });
    s = applyCommand(s, { type: 'move', unitId: runner.id, x: 4, y: 7 });
    assert.equal(unit(s, resident.id).hp, resident.hp);
    s = applyCommand(s, { type: 'move', unitId: runner.id, x: 5, y: 7 });
    s = applyCommand(s, { type: 'finish-mode', unitId: runner.id });
    assert.equal(unit(s, runner.id).operations, 1);
    assert.ok(validState(s));
  });

test('feedback3 Little BW: crosses both bases but cannot stop there or consume last step on occupied cell', () => {
  for (const y of [1, 13]) {
    let s = fixture();
    const runner = readyRunner(s, 'u12p', 4, y);
    s = applyCommand(s, { type: 'move', unitId: runner.id, x: 5, y });
    assert.ok(validState(s));
    rejected(s, { type: 'finish-mode', unitId: runner.id });
    s = applyCommand(s, { type: 'move', unitId: runner.id, x: 6, y });
    assert.equal(commandError(s, { type: 'finish-mode', unitId: runner.id }), null);
  }
  let s = fixture();
  const r = readyRunner(s, 'u12p', 2, 5),
    blocker = add(s, 1, 2, 7, 5);
  for (let x = 3; x <= 6; x++) s = applyCommand(s, { type: 'move', unitId: r.id, x, y: 5 });
  rejected(s, { type: 'move', unitId: r.id, x: 7, y: 5 });
  assert.equal(unit(s, blocker.id).hp, 50);
});

test('feedback3 soul mage: 40HP, self/base endpoints, operation-consuming once per actual turn, persists while frozen', () => {
  const s = fixture(),
    mage = add(s, 'u14', 1, 5, 4);
  assert.equal(mage.hp, 40);
  const c: Command = { type: 'skill', unitId: mage.id, targetId: mage.id, secondId: 'base-1' };
  let n = applyCommand(s, c);
  assert.equal(unit(n, mage.id).operations, 1);
  unit(n, mage.id).operations = 0;
  rejected(n, c);
  addEffect(n, unit(n, mage.id), 'freeze', 2, 0, 2);
  n = pass(n);
  assert.equal(n.siphons.length, 1);
  assert.equal(n.bases[1], 300);
});

test('feedback3 soul mage: connection disappears when an endpoint leaves range', () => {
  const s = fixture(),
    mage = add(s, 'u14', 1, 2, 5),
    target = add(s, 1, 1, 7, 5);
  let n = applyCommand(s, {
    type: 'skill',
    unitId: mage.id,
    targetId: mage.id,
    secondId: target.id,
  });
  n = applyCommand(n, { type: 'move', unitId: target.id, x: 8, y: 5 });
  assert.equal(n.siphons.length, 0);
});

test('feedback3 equipment: replacement retains one weapon and never accumulates Heart health', () => {
  let s = fixture();
  const carrier = add(s, 1, 1, 3, 4);
  for (const weapon of ['u28', 'u28', 'u16', 'u28'] as Kind[]) {
    s = applyCommand(s, { type: 'equip', cardId: card(s, weapon), targetId: carrier.id });
    assert.deepEqual(unit(s, carrier.id).equipment, [weapon]);
    assert.equal(unit(s, carrier.id).maxHp, 50 + (weapon === 'u28' ? 5 : 0));
  }
});

test('feedback3 Heart: bent selected path hits only enemies on it, adds burn, never extends to unselected cells', () => {
  const s = fixture(),
    carrier = add(s, 26, 1, 2, 5),
    first = add(s, 1, 2, 3, 5),
    last = add(s, 1, 2, 4, 6),
    outside = add(s, 1, 2, 5, 6),
    friend = add(s, 1, 1, 3, 6);
  installEquipment(carrier, 'u28');
  const path = [
    { x: 2, y: 5 },
    { x: 3, y: 5 },
    { x: 3, y: 6 },
    { x: 4, y: 6 },
  ];
  const original = structuredClone(s);
  let i = startIntent(unitActions(s, carrier).find((a) => a.id === 'attack-path')!);
  for (const p of path) {
    assert.ok(canChoose(s, i, p));
    assert.equal(commandFor(s, i, p), null);
    i = advanceIntent(i, p, s);
  }
  assert.deepEqual(s, original);
  assert.ok(i.kind === 'select');
  const n = applyCommand(s, i.draft);
  for (const v of [first, last]) {
    assert.equal(unit(n, v.id).hp, 30);
    assert.ok(has(n, unit(n, v.id), 'burn'));
  }
  assert.equal(unit(n, friend.id).hp, 50);
  assert.equal(unit(n, outside.id).hp, 50);
  assert.equal(unit(n, carrier.id).shots, 0);
  assert.equal(unit(n, carrier.id).operations, 1);
  assert.deepEqual(n.events.filter((e) => e.type === 'attack').at(-1)!.path, path);
});

test('feedback3 Heart: multiple covered cells of one large unit are one normal hit, unlike AOE', () => {
  const s = fixture(),
    carrier = add(s, 26, 1, 2, 5),
    big = add(s, 5, 2, 3, 5);
  installEquipment(carrier, 'u28');
  const n = applyCommand(s, {
    type: 'attack',
    unitId: carrier.id,
    path: [
      { x: 2, y: 5 },
      { x: 3, y: 5 },
      { x: 4, y: 5 },
    ],
  });
  assert.equal(unit(n, big.id).hp, 91);
  assert.equal(unit(n, big.id).effects.filter((e) => e.type === 'burn').length, 1);
});

test('feedback3 Heart: untrusted paths cannot teleport, loop, overshoot, start elsewhere or cross enemy base', () => {
  const s = fixture(),
    carrier = add(s, 26, 1, 4, 12),
    enemy = add(s, 1, 2, 6, 13);
  installEquipment(carrier, 'u28');
  for (const path of [
    [],
    [
      { x: 4, y: 12 },
      { x: 6, y: 13 },
    ],
    [
      { x: 3, y: 12 },
      { x: 4, y: 12 },
    ],
    [
      { x: 4, y: 12 },
      { x: 4, y: 13 },
      { x: 4, y: 12 },
    ],
    [
      { x: 4, y: 12 },
      { x: 4, y: 13 },
      { x: 5, y: 13 },
      { x: 6, y: 13 },
    ],
    [
      { x: 4, y: 12 },
      { x: 4.5, y: 12 },
    ],
    [
      { x: 4, y: 12 },
      { x: 4, y: 11 },
      { x: 4, y: 10 },
      { x: 4, y: 9 },
      { x: 4, y: 8 },
      { x: 4, y: 7 },
    ],
  ])
    rejected(s, { type: 'attack', unitId: carrier.id, targetId: enemy.id, path });
});

test('feedback3 human control: only own free giant skill can interrupt an AI turn', async () => {
  const { humanCommandAllowed } = await import('../../src/match/history');
  const { Arena } = await import('../../src/match/arena');
  const s = fixture(),
    own = add(s, 'u7', 1, 2, 3),
    enemy = add(s, 'u7', 2, 7, 8),
    target = add(s, 1, 2, 5, 7);
  s.active = 2;
  const session = createSession(s, { mode: 'ai', human: 1, difficulty: 'easy' });
  const c: Command = { type: 'skill', unitId: own.id, targetId: target.id, x: 4, y: 6 };
  assert.ok(humanCommandAllowed(session, c));
  assert.equal(humanCommandAllowed(session, { type: 'move', unitId: own.id, x: 3, y: 3 }), false);
  assert.equal(humanCommandAllowed(session, { ...c, unitId: enemy.id }), false);
  const arena = new Arena(session),
    entry = arena.play(c);
  assert.equal(entry.owner, 1);
  assert.equal(entry.actor, 'human');
  assert.equal(unit(arena.session.present, target.id).size, 2);
  session.present.active = 1;
  assert.equal(humanCommandAllowed(session, { ...c, unitId: enemy.id }), false);
});

test('feedback3 horn cannot refresh the soul mage actual-turn skill allowance', () => {
  const s = fixture(),
    mage = add(s, 'u14', 1, 3, 5),
    target = add(s, 1, 2, 4, 5);
  const c: Command = { type: 'skill', unitId: mage.id, targetId: target.id, secondId: mage.id };
  let n = applyCommand(s, c);
  n = applyCommand(n, { type: 'cast', cardId: card(n, 'u17'), targetId: mage.id });
  assert.equal(unit(n, mage.id).operations, 0);
  rejected(n, c);
});
