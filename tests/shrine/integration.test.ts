import test from 'node:test';
import assert from 'node:assert/strict';
import {
  applyCommand,
  createGame,
  createSession,
  dispatch,
  undo,
  redo,
  validState,
  parseSession,
  unitActions,
  getStats,
  type Player,
  type Kind,
} from '../../src/engine';
import { observe } from '../../src/ai/observation';
import { decide } from '../../src/ai/planning/search';
import { chargeFor } from '../../src/engine/core/traits';
import { add, card, fixture, round, unit } from '../helpers';

test('3.0 hidden commitment and Jade parity do not alter opponent observation, fingerprint input or AI choice', () => {
  const original = createGame(90, 'shrine');
  original.shrineDraft!.offers[1] = ['s9', 's8', 's2'];
  for (const humanFirst of [true, false]) {
    const side: Player = humanFirst ? 1 : 2,
      ai: Player = side === 1 ? 2 : 1;
    original.shrineDraft!.offers[side] = ['s9', 's8', 's2'];
    const states = [
      { kind: 's9' as Kind, parity: 'odd' as const },
      { kind: 's9' as Kind, parity: 'even' as const },
      { kind: 's8' as Kind, parity: 'odd' as const },
    ].map((choice) =>
      applyCommand(original, {
        type: 'choose-shrine',
        player: side,
        shrineKind: choice.kind,
        parity: choice.parity,
      }),
    );
    const observations = states.map((s) => observe(s, ai));
    assert.deepEqual(observations[0], observations[1]);
    assert.deepEqual(observations[0], observations[2]);
    assert.equal('rng' in observations[0], false);
    assert.equal('log' in observations[0], false);
    assert.equal(observations[0].shrineDraft!.choices[side], undefined);
    for (const difficulty of ['easy', 'medium', 'hard'] as const) {
      const moves = observations.map(
        (o) => decide(o, ai, difficulty, { mode: 'work', simulations: 240 }).command,
      );
      assert.deepEqual(moves[0], moves[1]);
      assert.deepEqual(moves[0], moves[2]);
      assert.ok(moves[0]);
      assert.ok(validState(applyCommand(states[0], moves[0]!)));
    }
  }
});

test('3.0 drafting persists and undo/redo restores secret commitment, reveal and PRNG exactly', () => {
  const s = createGame(90, 'shrine'),
    session = createSession(s);
  const committed = dispatch(session, { type: 'choose-shrine', player: 1, shrineKind: 's8' });
  assert.deepEqual(parseSession(JSON.stringify(committed)), committed);
  const revealed = dispatch(committed, { type: 'choose-shrine', player: 2, shrineKind: 's13' });
  assert.deepEqual(undo(revealed).present, committed.present);
  assert.deepEqual(redo(undo(revealed)).present, revealed.present);
  const broken = structuredClone(revealed);
  broken.present.shrineDraft!.choices[1] = { kind: 'laoqian' };
  assert.throws(() => parseSession(JSON.stringify(broken)), /损坏/);
  const legacy = createSession(createGame(7));
  assert.deepEqual(parseSession(JSON.stringify(legacy)), legacy);
});

test('3.0 inherited charge resources are independent from each other and native identity', () => {
  let s = fixture();
  const u = add(s, 's5', 1, 3, 4);
  u.traits = ['u2', 'u6'];
  for (const ability of ['u2', 'u6'] as Kind[]) {
    const action = unitActions(s, unit(s, u.id)).find(
      (a) => a.command.type === 'charge' && a.command.ability === ability,
    )!;
    assert.ok(action);
    s = applyCommand(s, action.command);
    s = round(s);
  }
  const v = unit(s, u.id);
  assert.equal(v.kind, 's5');
  assert.equal(v.charge, 0);
  assert.equal(chargeFor(v, 'u2').charge, 1);
  assert.equal(chargeFor(v, 'u6').charge, 1);
  assert.equal(getStats(s, v).attack, 35);
  assert.ok(validState(s));
  const action = unitActions(s, v).find(
    (a) => a.command.type === 'skill' && a.command.ability === 'u6',
  )!;
  s = applyCommand(s, { ...action.command, x: 3, y: 6 });
  assert.ok(unit(s, u.id).abilityUsage?.u6?.once);
  assert.equal(chargeFor(unit(s, u.id), 'u2').charge, 1);
});

test('3.0 Laoqian self-selection also covers rerolls and reforge without crossing pools', () => {
  let s = fixture();
  s.auras = { 1: [{ kind: 'laoqian' }], 2: [] };
  const mage = add(s, 'u13', 1, 3, 4),
    id = card(s, 25);
  s = applyCommand(s, { type: 'cast', cardId: id, mode: 'single', chosenKind: 1 });
  assert.equal(s.hands[1][0].kind, 1);
  assert.equal(s.auras![1][0].usedPly, s.ply);
  assert.equal(s.summonSlots, 0);
  s = round(s);
  const drawn = card(s, 'u1');
  s.hands[1].find((c) => c.id === drawn)!.summonPool = 'ultimate';
  s = applyCommand(s, { type: 'reroll', cardId: drawn, unitId: mage.id, chosenKind: 'u21' });
  assert.equal(s.hands[1].at(-1)!.kind, 'u21');
  assert.ok(validState(s));
});

test('3.0 inherited fractional movement has a selectable charge, not an unpayable movement lock', () => {
  let s = fixture();
  const u = add(s, 's5', 1, 3, 4);
  u.traits = [3];
  const action = unitActions(s, u).find(
    (a) => a.command.type === 'charge' && a.command.ability === 3,
  );
  assert.ok(action);
  s = applyCommand(s, action.command);
  s = round(s);
  s = applyCommand(s, { type: 'move', unitId: u.id, x: 3, y: 5 });
  assert.equal(unit(s, u.id).y, 5);
  assert.equal(chargeFor(unit(s, u.id), 3).charge, 0);
});
