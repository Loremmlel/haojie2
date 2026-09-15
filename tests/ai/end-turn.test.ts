import test from 'node:test';
import assert from 'node:assert/strict';
import position from '../fixtures/ai/end-turn-20260915.json';
import { add, fixture } from '../helpers';
import { applyCommand, commandError, createSession } from '../../src/engine';
import { getStats } from '../../src/engine/state';
import { Arena } from '../../src/match/arena';
import { decide } from '../../src/ai/search';
import { cachedDecision } from '../../src/ai/plan-cache';
import { fingerprint, imagined, observe } from '../../src/ai/observation';
import { evaluate } from '../../src/ai/evaluate';
import type { Command, GameState } from '../../src/engine';
import type { Observation } from '../../src/ai/types';

// Public observation before command 229 of the 315-command September 15 CLI match.
// No real seed/RNG, history, oracle move or future draw is part of the fixture.
const beforeDeploy = () => imagined(position as Observation);
const beforeEnd = () =>
  applyCommand(beforeDeploy(), {
    type: 'deploy',
    cardId: 'c424',
    charge: false,
    x: 5,
    y: 12,
  });
const arenaFor = (s: GameState, difficulty: 'easy' | 'medium' | 'hard' = 'hard') =>
  new Arena({
    ...createSession(s),
    match: { mode: 'ai', human: s.active === 1 ? 2 : 1, difficulty },
  });

function finishTurn(arena: Arena, max = 20) {
  const entries = [];
  const ply = arena.session.present.ply;
  while (arena.computerTurn && arena.session.present.ply === ply && entries.length < max)
    entries.push(arena.step({ trace: true })!);
  assert.ok(
    arena.session.present.ply > ply || arena.session.present.winner,
    'must actually finish, not loop',
  );
  return entries;
}

test('the recorded deployment no longer carries a cached END past profitable remaining attacks', () => {
  const initial = beforeDeploy();
  assert.equal(fingerprint(initial), '151ngyz:14340');
  const arena = arenaFor(initial);
  const entries = finishTurn(arena);
  assert.ok(entries.some((e) => e.command.type === 'deploy'));
  assert.ok(entries.some((e) => e.command.type === 'attack'));
  assert.equal(
    arena.session.present.units.some((u) => u.id === 'u305'),
    false,
    'finish the protected clone',
  );
  assert.equal(entries.at(-1)!.command.type, 'end');
  assert.notEqual(entries.at(-1)!.decision!.stats.cached, true, 'END must be freshly decided');
});

test('a 40-simulation fresh search compares attacks and waiting at the same turn boundary', () => {
  const s = beforeEnd(),
    observation = observe(s);
  const result = decide(observation, 2, 'hard', { simulations: 40, trace: true });
  assert.equal(result.command?.type, 'attack');
  assert.ok(result.stats.endTurnImproved);
  assert.ok(result.stats.endTurnChecks! > 0);
  assert.ok(
    result.stats.simulations <= 40,
    'recheck uses the original budget, not a second search',
  );
  const noRandom = () => {
    throw new Error('this recorded attack sequence needs no dice');
  };
  let next = s;
  for (const step of result.plan) {
    assert.equal(step.before, fingerprint(next));
    next = applyCommand(next, step.command, noRandom);
  }
  const wait = applyCommand(s, { type: 'end' }, noRandom);
  const attackThenEnd = applyCommand(next, { type: 'end' }, noRandom);
  assert.equal(
    attackThenEnd.units.some((u) => u.id === 'u305'),
    false,
  );
  assert.ok(evaluate(attackThenEnd, 2) > evaluate(wait, 2));
  assert.deepEqual(new Set(result.trace!.alternatives.map((a) => a.stage)), new Set(['end-turn']));
  assert.deepEqual(observe(s), observation);
});

test('shared cache keeps verified tactical steps but treats END as a fresh decision', () => {
  const s = beforeEnd(),
    before = fingerprint(s);
  const attack: Command = { type: 'attack', unitId: 'u322', targetId: 'u305' };
  const plan = [{ before, command: attack }];
  assert.deepEqual(cachedDecision(observe(s), plan)?.command, attack);
  assert.equal(cachedDecision(observe(s), [{ before, command: { type: 'end' } }]), null);
  assert.equal(cachedDecision(observe(applyCommand(s, attack)), plan), null);
  assert.deepEqual(plan, [{ before, command: attack }]);
});

test('all levels collect reachable damage, healing and charge value without needing a move', () => {
  for (const difficulty of ['easy', 'medium', 'hard'] as const) {
    const attack = beforeEnd();
    const a = arenaFor(attack, difficulty);
    const entries = finishTurn(a);
    assert.ok(
      entries.some((e) => e.command.type === 'attack'),
      difficulty,
    );

    const heal = fixture(),
      nurse = add(heal, 2, 1, 4, 4),
      ally = add(heal, 5, 1, 4, 5);
    ally.hp = 30;
    ally.operations = 1;
    const h = decide(observe(heal), 1, difficulty, { simulations: 800 });
    const healed = applyCommand(heal, h.command!);
    assert.equal(healed.units.find((u) => u.id === ally.id)!.hp, 50);
    assert.equal(h.command!.unitId, nurse.id);

    const charge = fixture(),
      gun = add(charge, 4, 1, 5, 7);
    add(charge, 5, 2, 5, 10);
    const c = decide(observe(charge), 1, difficulty, { simulations: 800 });
    assert.equal(c.command?.type, 'charge');
    assert.equal(applyCommand(charge, c.command!).units.find((u) => u.id === gun.id)!.charge, 1);
  }
});

test('an actionable guardian can hold its firing-lane cover instead of being forced to move', () => {
  for (const difficulty of ['medium', 'hard'] as const) {
    const s = fixture(),
      guardian = add(s, 3, 1, 5, 3),
      gun = add(s, 4, 2, 5, 5);
    s.heads[1] = 0;
    s.bases[1] = 40;
    guardian.charge = guardian.readyCharge = 1;
    guardian.chargeType = 'move';
    gun.charge = gun.readyCharge = 2;
    gun.chargeType = 'attack';
    assert.ok(getStats(s, guardian).operationsLeft > 0);
    assert.equal(commandError(s, { type: 'move', unitId: guardian.id, x: 4, y: 3 }), null);
    const result = decide(observe(s), 1, difficulty, { simulations: 800 });
    assert.equal(result.command?.type, 'end');
  }
});

test('retaliation deaths, rage feeding and immune hits are not mistaken for free income', () => {
  for (const kind of ['retaliation', 'rage', 'immune'] as const) {
    const s = fixture(),
      attacker = add(s, kind === 'rage' ? 23 : kind === 'immune' ? 9 : 26, 1, 4, 4);
    s.heads[1] = 0;
    attacker.mode = 'attack';
    if (kind === 'retaliation') attacker.hp = 5;
    const target = add(s, kind === 'rage' ? 'u10' : 'u18', 2, 4, 6);
    assert.equal(
      commandError(s, { type: 'attack', unitId: attacker.id, targetId: target.id }),
      null,
    );
    const arena = arenaFor(s);
    const entries = finishTurn(arena);
    assert.ok(
      entries.every((e) => e.command.type !== 'attack'),
      kind,
    );
    assert.equal(arena.session.present.units.find((u) => u.id === attacker.id)!.hp, attacker.hp);
    assert.equal(arena.session.present.units.find((u) => u.id === target.id)!.attackBonus, 0);
  }
});

test('end rechecks remain public-only, deterministic in work mode', (t) => {
  const s = beforeEnd(),
    snapshot = structuredClone(s),
    observation = observe(s);
  const expected = decide(observation, 2, 'hard', { simulations: 40, trace: true });
  let clock = 0;
  t.mock.method(performance, 'now', () => (clock += 10000));
  s.seed = 98231;
  s.rng = 3321123;
  assert.deepEqual(decide(observe(s), 2, 'hard', { simulations: 40, trace: true }), expected);
  assert.deepEqual(observation, observe(snapshot));
  assert.equal('rng' in position, false);
  assert.equal('seed' in position, false);
});
