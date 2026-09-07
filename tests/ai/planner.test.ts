import test from 'node:test';
import assert from 'node:assert/strict';
import { add, card, fixture } from '../helpers';
import {
  applyCommand,
  CATALOG,
  createGame,
  createSession,
  dispatch,
  isLegal,
  parseSession,
  validState,
} from '../../src/engine';
import { decide } from '../../src/ai/search';
import { observe, imagined, decisionOwner } from '../../src/ai/observation';
import { distribution } from '../../src/ai/simulate';
import { candidateGroups } from '../../src/ai/candidates';
import { rewindMatch, matchSettings } from '../../src/match/history';
import { writeStoredSession } from '../../src/ui/session/storage';
import type { Difficulty } from '../../src/ai/types';
const limits = { simulations: 2200, milliseconds: 100000 };
const plan = (s: ReturnType<typeof fixture>, d: Difficulty = 'medium') =>
  decide(observe(s), decisionOwner(s), d, limits);

test('observation whitelist strips true RNG, seed, logs and unknown future secret fields', () => {
  const s = fixture();
  Object.assign(s, { futureSecret: 'not for AI' });
  s.log.push('a secret');
  const o = observe(s);
  assert.ok(!('rng' in o) && !('seed' in o) && !('log' in o) && !('futureSecret' in o));
  const twin = structuredClone(s);
  twin.rng = 918202;
  twin.seed = 8181;
  twin.events = [];
  twin.log = [];
  add(s, 9, 1, 5, 10);
  add(twin, 9, 1, 5, 10);
  assert.deepEqual(plan(s, 'easy'), plan(twin, 'easy'));
  assert.equal(imagined(o).rng, 1);
});
test('planning and simulation never mutate the authoritative board, RNG or history', () => {
  const s = fixture();
  add(s, 1, 1, 3, 4);
  add(s, 20, 2, 3, 5);
  const before = structuredClone(s);
  plan(s);
  assert.deepEqual(s, before);
});
test('critical-hit enumeration evaluates three results, not average damage as a guaranteed kill', () => {
  const s = fixture(),
    a = add(s, 1, 1, 3, 4),
    b = add(s, 1, 2, 3, 5);
  b.hp = 30;
  const result = distribution(s, { type: 'attack', unitId: a.id, targetId: b.id });
  assert.equal(result.sampled, false);
  const killChance = result.outcomes
    .filter((o) => !o.state.units.some((u) => u.id === b.id))
    .reduce((n, o) => n + o.weight, 0);
  assert.ok(Math.abs(killChance - 1 / 3) < 1e-9);
  assert.ok(Math.abs(result.outcomes.reduce((n, o) => n + o.weight, 0) - 1) < 1e-9);
});
test('random counterspell and critical retaliation compose independent exact probabilities', () => {
  const s = fixture(),
    a = add(s, 1, 1, 3, 4),
    b = add(s, 20, 2, 3, 5);
  const result = distribution(s, { type: 'attack', unitId: a.id, targetId: b.id }, 20);
  assert.equal(result.sampled, false);
  assert.ok(
    Math.abs(
      result.outcomes
        .filter((o) => o.state.heads[1] === s.heads[1] + 1)
        .reduce((n, o) => n + o.weight, 0) - 0.5,
    ) < 1e-9,
  );
});
test('summon distribution uses independent imaginary samples, not the next real draw', () => {
  const s = createGame(77),
    before = structuredClone(s);
  const a = distribution(imagined(observe(s)), { type: 'summon' }, 12, 4);
  s.rng = 50001;
  const b = distribution(imagined(observe(s)), { type: 'summon' }, 12, 4);
  assert.deepEqual(a, b);
  assert.equal(a.sampled, true);
  assert.ok(a.outcomes.length >= 2);
  assert.deepEqual(before.units, s.units);
});
test('all difficulty levels choose an available certain base kill', () => {
  for (const d of ['easy', 'medium', 'hard'] as Difficulty[]) {
    const s = fixture(),
      a = add(s, 9, 1, 5, 10);
    s.bases[2] = 10;
    const result = plan(s, d);
    assert.ok(result.command);
    const next = applyCommand(s, result.command);
    assert.equal(next.winner, 1, `${d}: ${JSON.stringify(result.command)}`);
  }
});
test('beam planning finds execution + horn + attack from a freshly deployed unit', () => {
  const s = fixture(),
    a = add(s, 9, 1, 4, 4),
    b = add(s, 5, 2, 4, 6);
  a.born = s.turns[1];
  a.deployedAt = s.ply;
  card(s, 18);
  card(s, 'u17');
  const result = plan(s, 'medium');
  assert.ok(result.plan.length >= 3, JSON.stringify(result));
  let next = s;
  for (const step of result.plan) next = applyCommand(next, step.command);
  assert.equal(
    next.units.some((u) => u.id === b.id),
    false,
  );
});
test('hard level actually expands opponent decisions instead of labelling a wider beam adversarial', () => {
  const s = fixture();
  add(s, 9, 1, 3, 5);
  add(s, 26, 1, 6, 5);
  add(s, 26, 2, 3, 7);
  add(s, 9, 2, 6, 8);
  const result = decide(observe(s), 1, 'hard', { simulations: 7000, milliseconds: 100000 });
  assert.ok(result.stats.replies > 0, JSON.stringify(result.stats));
  assert.ok(result.command && isLegal(s, result.command));
});
test('all summonable unit families have executable AI candidates under their rules', () => {
  for (const d of CATALOG.filter((d) => d.spell === undefined && d.weapon === undefined)) {
    const s = fixture(),
      u = add(s, d.id, 1, 4, 4);
    add(s, 1, 2, 4, 7);
    add(s, 1, 1, 2, 4);
    u.born = 1;
    u.charge = u.readyCharge = d.id === 4 || d.id === 21 ? 2 : 1;
    u.chargeType = d.move % 1 ? 'move' : d.id === 21 || d.id === 'u6' ? 'skill' : 'attack';
    const groups = candidateGroups(imagined(observe(s)), 'medium');
    assert.ok(
      (d.actions === 0 && d.move === 0) ||
        groups.flatMap((g) => g.commands).some((c) => c.unitId === u.id && isLegal(s, c)),
      String(d.id),
    );
  }
});
test('all active skill descriptions produce a legal candidate, including multistage targeting', () => {
  for (const kind of [
    5,
    6,
    7,
    14,
    15,
    19,
    21,
    'u6',
    'u7',
    'u14',
    'u19',
    'u21',
    'u23',
    'u24',
  ] as const) {
    const s = fixture(),
      u = add(s, kind, 1, 4, 4);
    add(s, 1, 2, 4, 6);
    const friend = add(s, 26, 1, 2, 4);
    friend.hp = 20;
    u.charge = u.readyCharge = 2;
    u.chargeType = 'skill';
    u.hookReadyAt = s.ply;
    u.hookExpiresAt = s.ply + 2;
    s.deaths.push({ id: 'dead-check', kind: 1, owner: 1, ply: s.ply - 2, revived: false });
    const commands = candidateGroups(imagined(observe(s)), 'hard')
      .flatMap((g) => g.commands)
      .filter((c) => c.type === 'skill' && c.unitId === u.id);
    assert.ok(
      commands.some((c) => isLegal(s, c)),
      String(kind),
    );
  }
});
test('all stored cards, including weapons, have AI targeting candidates', () => {
  for (const d of CATALOG.filter((d) => d.spell !== undefined || d.weapon !== undefined)) {
    const s = fixture();
    add(s, 19, 1, 2, 3);
    add(s, 26, 1, 4, 4);
    add(s, 1, 2, 4, 5);
    const id = card(s, d.id);
    assert.ok(
      candidateGroups(imagined(observe(s)), 'hard')
        .flatMap((g) => g.commands)
        .some((c) => c.cardId === id && isLegal(s, c)),
      d.name,
    );
  }
});
test('reaction ownership is independent of the active turn and AI does not steal a human reaction', () => {
  const s = fixture(),
    u = add(s, 2, 2, 3, 5);
  s.units = [];
  s.pending.push({ kind: 'death-shot', owner: 2, source: u, amount: 20 });
  add(s, 1, 1, 3, 4);
  assert.equal(decisionOwner(s), 2);
  assert.equal(decide(observe(s), 1, 'hard').command, null);
  const r = decide(observe(s), 2, 'easy', limits);
  assert.equal(r.command?.type, 'react');
  assert.ok(isLegal(s, r.command!));
});
test('forced bounce and small-BW transit keep all legal escape candidates', () => {
  for (const kind of ['u12', 'u12p'] as const) {
    const s = fixture(),
      a = add(s, kind, 1, 3, 4);
    add(s, 5, 2, 3, 5);
    a.charge = a.readyCharge = 1;
    a.chargeType = 'move';
    const next = applyCommand(s, { type: 'move', unitId: a.id, x: 3, y: 5 });
    const result = plan(next, 'easy');
    assert.ok(result.command && isLegal(next, result.command));
    assert.ok(result.command.type === (kind === 'u12' ? 'react' : 'move'));
  }
});
test('bounded greedy self-play finishes real turns with summons, deployments and no illegal commands', () => {
  for (const seed of [7, 42, 20260907]) {
    let s = createGame(seed),
      count = 0;
    while (s.ply < 9 && !s.winner && count++ < 220) {
      const decision = decide(observe(s), decisionOwner(s), 'easy', {
        simulations: 160,
        milliseconds: 100000,
      });
      assert.ok(decision.command, `no command: ${seed} ply${s.ply}`);
      s = applyCommand(s, decision.command);
      assert.ok(validState(s));
    }
    assert.ok(s.ply >= 9 || s.winner, `seed ${seed} did not progress; ply${s.ply}`);
  }
});
test('AI settings roundtrip; old v2 files stay local; quota fallback does not lose the opponent', () => {
  const s = createSession(fixture(), { mode: 'ai', human: 2, difficulty: 'hard' });
  assert.deepEqual(parseSession(JSON.stringify(s)), s);
  assert.equal(matchSettings(parseSession(JSON.stringify(createSession(fixture())))).mode, 'local');
  const corrupt = { ...s, match: { mode: 'ai', human: 5, difficulty: 'extra' } };
  assert.throws(() => parseSession(JSON.stringify(corrupt)));
  let stored = '';
  writeStoredSession(
    {
      getItem: () => null,
      setItem: (_k, v) => {
        if (!stored) {
          stored = 'retry';
          throw new Error('quota');
        }
        stored = v;
      },
    },
    'test',
    s,
  );
  assert.deepEqual(JSON.parse(stored).match, s.match);
});
test('human undo rewinds the entire computer response and redo restores it without rerolling', () => {
  const state = fixture();
  add(state, 9, 2, 5, 10);
  let s = createSession(state, { mode: 'ai', human: 1, difficulty: 'medium' });
  const before = s.present;
  s = dispatch(s, { type: 'end' });
  s = dispatch(s, { type: 'summon' });
  s = dispatch(s, { type: 'summon' });
  s = dispatch(s, { type: 'begin' });
  const after = s.present,
    back = rewindMatch(s);
  assert.deepEqual(back.present, before);
  assert.deepEqual(rewindMatch(back, true).present, after);
});

test('long AI turns preserve a human decision anchor beyond the 60 atomic undo limit', () => {
  const state = fixture();
  state.bonus[2] = 78;
  let s = createSession(state, { mode: 'ai', human: 1, difficulty: 'medium' });
  s = dispatch(s, { type: 'end' });
  for (let i = 0; i < 65; i++) s = dispatch(s, { type: 'summon' });
  s = parseSession(JSON.stringify(s));
  const beforeUndo = s.present;
  const back = rewindMatch(s);
  assert.deepEqual(back.present, state);
  const forward = rewindMatch(back, true);
  assert.deepEqual(forward.present, beforeUndo);
  assert.deepEqual(rewindMatch(forward).present, state);
});
