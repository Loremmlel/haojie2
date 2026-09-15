import test from 'node:test';
import assert from 'node:assert/strict';
import { applyCommand, asTarget, createGame, isLegal } from '../../src/engine';
import { candidateGroups, commandPriority } from '../../src/ai/candidates';
import { analyzePayload, attackPressure, hitPackets } from '../../src/ai/threats';
import { distribution } from '../../src/ai/simulate';
import { imagined, observe } from '../../src/ai/observation';
import { add, card, fixture } from '../helpers';
import type { Difficulty } from '../../src/ai/types';

test('feedback AI: all levels enumerate legal flank/knockback directions and both explicit charge deployments', () => {
  for (const difficulty of ['easy', 'medium', 'hard'] as Difficulty[]) {
    const s = fixture(),
      a = add(s, 26, 1, 4, 4),
      b = add(s, 24, 2, 4, 6);
    const id = card(s, 1);
    const commands = candidateGroups(imagined(observe(s)), difficulty).flatMap((g) => g.commands);
    const attacks = commands.filter(
      (c) => c.type === 'attack' && c.unitId === a.id && c.targetId === b.id,
    );
    assert.deepEqual(new Set(attacks.map((c) => c.direction)), new Set(['down', 'left', 'right']));
    assert.ok(attacks.every((c) => isLegal(s, c)));
    assert.ok(
      commandPriority(s, attacks.find((c) => c.direction === 'left')!) >
        commandPriority(s, attacks.find((c) => c.direction === 'down')!),
    );
    const deployments = commands.filter((c) => c.type === 'deploy' && c.cardId === id);
    assert.deepEqual(new Set(deployments.map((c) => c.charge)), new Set([false, true]));
    assert.equal(new Set(deployments.map((c) => JSON.stringify(c))).size, deployments.length);
  }
});

test('feedback AI: same-kind sacrifice and immediately spendable bonus summons are present in every difficulty', () => {
  const s = fixture(),
    a = add(s, 14, 1, 3, 4),
    b = add(s, 14, 1, 4, 4);
  for (const difficulty of ['easy', 'medium', 'hard'] as Difficulty[]) {
    const commands = candidateGroups(imagined(observe(s)), difficulty).flatMap((g) => g.commands);
    const special = commands.find(
      (c) => c.type === 'skill' && c.mode === 'summon' && c.unitId === a.id && c.targetId === b.id,
    )!;
    assert.ok(special);
    assert.ok(isLegal(s, special));
    const after = applyCommand(s, special);
    const draws = candidateGroups(imagined(observe(after)), difficulty)
      .flatMap((g) => g.commands)
      .filter((c) => c.type === 'summon');
    assert.deepEqual(new Set(draws.map((c) => c.ultimate)), new Set([false, true]));
    assert.ok(draws.every((c) => isLegal(after, c)));
  }
});

test('feedback AI: charged damage is one shot, uncharged follow-up needs its own range, and conversion excludes giant', () => {
  const s = fixture(),
    a = add(s, 15, 1, 3, 4),
    b = add(s, 'grave', 2, 3, 6);
  a.charge = a.readyCharge = 4;
  assert.equal(attackPressure(s, a, asTarget(b)), 30); // 25 + 5, not 25 * 2.
  const far = structuredClone(s);
  far.units[1].y = 9;
  assert.equal(attackPressure(far, far.units[0], asTarget(far.units[1])), 25);
  const empty = fixture(),
    charger = add(empty, 15, 1, 3, 4);
  assert.ok(
    candidateGroups(imagined(observe(empty)), 'medium')
      .flatMap((g) => g.commands)
      .some((c) => c.type === 'charge' && c.unitId === charger.id && isLegal(empty, c)),
  );
  const giant = add(s, 5, 2, 5, 4);
  a.effects.push({ type: 'convert', owner: 1, from: s.ply, until: s.ply + 2 });
  const analysis = analyzePayload(s, a, 'convert', () => 100);
  assert.equal(analysis.targets.find((t) => t.id === giant.id)?.probability, 0);
});

test('feedback AI: gold lottery is exactly 20/80 conditional on normal slot 17; damage immunity is an independent 50/50 branch', () => {
  const draw = distribution(createGame(1), { type: 'summon' }, 50);
  assert.equal(draw.sampled, false);
  const probability = (kind: number | string) =>
    draw.outcomes
      .filter((o) => o.state.hands[1][0].kind === kind)
      .reduce((n, o) => n + o.weight, 0);
  assert.ok(Math.abs(probability(17) - 0.2 / 26) < 1e-12);
  assert.ok(Math.abs(probability('17p') - 0.8 / 26) < 1e-12);
  const s = fixture(),
    a = add(s, 26, 1, 3, 4),
    b = add(s, '17p', 2, 3, 6);
  const result = distribution(s, { type: 'attack', unitId: a.id, targetId: b.id });
  assert.equal(result.sampled, false);
  assert.deepEqual(result.outcomes.map((o) => o.weight).sort(), [0.5, 0.5]);
  assert.deepEqual(
    result.outcomes.map((o) => o.state.units.find((u) => u.id === b.id)!.hp).sort((a, b) => a - b),
    [5, 25],
  );
  assert.deepEqual(hitPackets(s, a, asTarget(b)), [
    { damage: 0, probability: 0.5 },
    { damage: 20, probability: 0.5 },
  ]);
});
