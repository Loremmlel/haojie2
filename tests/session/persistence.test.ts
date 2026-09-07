import test from 'node:test';
import assert from 'node:assert/strict';
import {
  applyCommand,
  createGame,
  createDemoGame,
  createSession,
  dispatch,
  undo,
  redo,
  parseSession,
  commandError,
  getStats,
  definition,
  ALL_CELLS,
  isStored,
} from '../../src/engine';
import { add, card, fixture, round, seedFor, strike, unit } from '../helpers';
import { draw, addEffect } from '../../src/engine/state';
import { damage, resolution, kill } from '../../src/engine/combat';
import type { Command, GameState } from '../../src/engine';
const roundtrip = (s: GameState) =>
  assert.deepEqual(parseSession(JSON.stringify(createSession(s))).present, s);

test('v2 saves roundtrip opening, demonstration and multi-operation history', () => {
  for (const s of [createGame(7), createDemoGame()]) roundtrip(s);
  let session = createSession(createGame(7));
  for (const c of [{ type: 'summon' }, { type: 'summon' }, { type: 'begin' }] as Command[])
    session = dispatch(session, c);
  const copied = parseSession(JSON.stringify(session));
  assert.deepEqual(copied, session);
  assert.deepEqual(redo(undo(copied)).present, copied.present);
});
test('undoing a critical hit restores both random numbers and heads', () => {
  const s = fixture(),
    u = add(s, 1, 1, 3, 4),
    v = add(s, 20, 2, 3, 5);
  s.rng = seedFor(0.2, 0.3);
  const c: Command = { type: 'attack', unitId: u.id, targetId: v.id },
    start = createSession(s),
    next = dispatch(start, c);
  assert.deepEqual(undo(next).present, s);
  assert.deepEqual(dispatch(undo(next), c).present, next.present);
  assert.equal(dispatch(undo(next), c).future.length, 0);
  roundtrip(next.present);
});
test('undo of terminal base damage reopens the same match', () => {
  const s = fixture(),
    u = add(s, 9, 1, 5, 10);
  s.bases[2] = 10;
  const start = createSession(s),
    end = dispatch(start, { type: 'attack', unitId: u.id, targetId: 'base-2' });
  assert.equal(end.present.winner, 1);
  roundtrip(end.present);
  assert.equal(undo(end).present.winner, undefined);
});
test('undo does not reroll an ultimate summon or the secondary SZF transformation', () => {
  const s = createGame(12345);
  s.heads[1] = 4;
  const start = createSession(s),
    cmd: Command = { type: 'summon', ultimate: true };
  const next = dispatch(start, cmd);
  assert.deepEqual(dispatch(undo(next), cmd).present, next.present);
});
test('clone stacks, frozen neutrality, equipment and links all survive serialization', () => {
  const s = fixture(),
    mage = add(s, 'u14', 1, 3, 4),
    a = add(s, 'u25', 2, 4, 5),
    b = add(s, 'u25', 2, 4, 5);
  a.group = b.group = 'one-batch';
  a.equipment = ['u16'];
  addEffect(s, a, 'freeze', 1, 0, 4, 5);
  s.siphons.push({ id: 'link-test', owner: 1, sourceId: mage.id, fromId: a.id, toId: 'base-1' });
  s.hazards.push({ id: 'fire-test', owner: 2, axis: 'row', line: 4, due: s.ply + 2 });
  s.iceMarks.push({ id: 'ice-test', owner: 1, sourceId: mage.id, x: 4, y: 6, due: s.ply + 2 });
  roundtrip(s);
});
test('a saved SZF collision resumes its compulsory bounce without losing move budget', () => {
  let s = fixture();
  const u = add(s, 'u12', 1, 3, 4),
    v = add(s, 1, 2, 3, 5);
  u.charge = u.readyCharge = 1;
  s = applyCommand(s, { type: 'move', unitId: u.id, x: 3, y: 5 });
  roundtrip(s);
  let session = dispatch(createSession(s), { type: 'react', x: 4, y: 5 });
  assert.equal(unit(session.present, u.id).moves, 4);
  session = undo(session);
  assert.equal(session.present.pending[0].kind, 'bounce');
  assert.deepEqual(parseSession(JSON.stringify(session)), session);
});
test('a saved small-BW transit disallows unrelated actions until it reaches empty ground', () => {
  let s = fixture();
  const u = add(s, 'u12p', 1, 3, 4),
    v = add(s, 1, 2, 3, 5),
    id = card(s, 17);
  u.charge = u.readyCharge = 1;
  s = applyCommand(s, { type: 'move', unitId: u.id, x: 3, y: 5 });
  roundtrip(s);
  assert.ok(commandError(s, { type: 'cast', cardId: id, targetId: u.id }));
  s = applyCommand(s, { type: 'move', unitId: u.id, x: 4, y: 5 });
  assert.equal(commandError(s, { type: 'cast', cardId: id, targetId: u.id }), null);
  roundtrip(s);
});
test('end-of-turn death choices serialize before switching players', () => {
  let s = fixture();
  const u = add(s, 2, 1, 3, 4);
  u.hp = 5;
  addEffect(s, u, 'burn', 2, 0, 12, 5);
  s = applyCommand(s, { type: 'end' });
  assert.equal(s.active, 1);
  assert.equal(s.summonSlots, -1);
  roundtrip(s);
  s = applyCommand(s, { type: 'react' });
  assert.equal(s.active, 2);
  assert.equal(s.phase, 'summon');
  roundtrip(s);
});
test('old saves and malformed numeric, graph and event fields fail without mutation', () => {
  const base = createSession(createDemoGame());
  assert.throws(
    () => parseSession(JSON.stringify({ ...base, format: 'haojie2-session-v1' })),
    /旧版/,
  );
  assert.throws(() => parseSession('{oops'));
  for (const change of [
    (s: any) => (s.present.units[0].hp = -1),
    (s: any) => (s.present.units[0].offset = 1),
    (s: any) => (s.present.units[0].size = 9),
    (s: any) => (s.present.hazards = [{ axis: 'row', line: 200 }]),
    (s: any) => (s.present.events = [null]),
    (s: any) => (s.present.units[0].equipment = ['missing']),
    (s: any) => (s.present.units[0].deployedAt = 'now'),
  ]) {
    const bad = structuredClone(base);
    change(bad);
    assert.throws(() => parseSession(JSON.stringify(bad)));
  }
  assert.deepEqual(base.present, createDemoGame());
});
test('history retains the most recent 60 commands with reversible random state', () => {
  let session = createSession(createGame(7));
  // Setting up currency is test-only; summon/begin are still real engine commands.
  for (let n = 0; n < 24; n++) {
    for (let i = 0; i < 2; i++) session = dispatch(session, { type: 'summon' });
    session = dispatch(session, { type: 'begin' });
    const clean = structuredClone(session.present);
    clean.hands[clean.active] = [];
    session = { ...session, present: clean };
    session = dispatch(session, { type: 'end' });
  }
  assert.equal(session.past.length, 60);
  assert.deepEqual(parseSession(JSON.stringify(session)), session);
});
test('gold body also rejects hook, silence, firestorm and hostile inner-fire', () => {
  for (const k of [7, 'u4', 'u9', 'u26'] as const) {
    let s = fixture();
    const v = add(s, 1, 2, 3, 6);
    addEffect(s, v, 'immune', 2, 0, 4);
    if (k === 7) {
      const u = add(s, k, 1, 3, 4);
      s = applyCommand(s, { type: 'skill', unitId: u.id, targetId: v.id, x: 4, y: 4 });
      assert.equal(unit(s, v.id).y, 6);
    } else if (k === 'u4') {
      const u = add(s, k, 1, 3, 4);
      s = strike(s, u, v);
      assert.equal(unit(s, v.id).hp, 50);
      assert.equal(unit(s, v.id).silenced, false);
    } else {
      const id = card(s, k);
      s = applyCommand(
        s,
        k === 'u9'
          ? { type: 'cast', cardId: id, mode: 'row', row: 6 }
          : { type: 'cast', cardId: id, targetId: v.id },
      );
      assert.equal(unit(s, v.id).hp, 50);
      assert.equal(
        unit(s, v.id).effects.some((e) => e.type === 'inner-fire'),
        false,
      );
    }
    roundtrip(s);
  }
});
test('enemy AoE skills select only the exposed clone, while spells select the full stack', () => {
  for (const kind of [5, 'u6'] as const) {
    const s = fixture(),
      u = add(s, kind, 1, 3, 4),
      a = add(s, 'u25', 2, 5, 5),
      b = add(s, 'u25', 2, 5, 5);
    a.group = b.group = 'batch';
    u.charge = u.readyCharge = 1;
    u.chargeType = 'skill';
    const n = applyCommand(s, { type: 'skill', unitId: u.id, x: 5, y: 5 });
    assert.equal(
      n.units.some((v) => v.id === a.id),
      false,
    );
    assert.ok(n.units.some((v) => v.id === b.id));
    assert.equal(n.heads[1], 6);
    roundtrip(n);
  }
});
test('ordinary non-charge minion can equip a heart next turn; haste does not bypass charge deployment ban', () => {
  let s = fixture();
  const summon = card(s, 1),
    heart = card(s, 'u28');
  s = applyCommand(s, { type: 'deploy', cardId: summon, x: 3, y: 4 });
  const u = s.units[0];
  s = round(s);
  s = applyCommand(s, { type: 'equip', cardId: heart, targetId: u.id });
  assert.deepEqual(unit(s, u.id).equipment, ['u28']);
  s = fixture();
  const id = card(s, 1),
    h = card(s, 'u28'),
    horn = card(s, 'u17');
  s = applyCommand(s, { type: 'deploy', cardId: id, x: 3, y: 4, charge: true });
  const charge = s.units[0];
  assert.ok(commandError(s, { type: 'equip', cardId: h, targetId: charge.id }));
  s = applyCommand(s, { type: 'cast', cardId: horn, targetId: charge.id });
  assert.ok(commandError(s, { type: 'equip', cardId: h, targetId: charge.id }));
  s = round(s);
  assert.equal(commandError(s, { type: 'equip', cardId: h, targetId: charge.id }), null);
});
test('a solitary clone can be enlarged, but a stack cannot be enlarged through an occupied footprint', () => {
  const s = fixture(),
    u = add(s, 'u7', 1, 1, 3),
    v = add(s, 'u25', 1, 4, 4);
  v.group = 'batch';
  const n = applyCommand(s, { type: 'skill', unitId: u.id, targetId: v.id });
  assert.equal(unit(n, v.id).size, 2);
  roundtrip(n);
  add(s, 'u25', 1, 4, 4);
  assert.ok(commandError(s, { type: 'skill', unitId: u.id, targetId: v.id }));
});
test('lowering a huts maximum to zero removes it and does not leave a zero-health live unit', () => {
  const s = fixture(),
    hut = add(s, 'u22', 1, 3, 4),
    v = add(s, 1, 1, 3, 5);
  hut.hp = hut.maxHp = 10;
  kill(s, v, { owner: 2, kind: 'spell' });
  const n = applyCommand(s, { type: 'react', x: 3, y: 5 });
  assert.equal(
    n.units.some((u) => u.id === hut.id),
    false,
  );
  assert.ok(n.units.some((u) => u.kind === 20));
  roundtrip(n);
});
