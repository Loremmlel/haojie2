import test from 'node:test';
import assert from 'node:assert/strict';
import { add, fixture } from '../helpers';
import { applyCommand, commandError } from '../../src/engine';
import { asTarget } from '../../src/engine/state';
import { attackProfile } from '../../src/engine/attack-profile';
import { decide } from '../../src/ai/search';
import { observe } from '../../src/ai/observation';
import { allocateBudget, emptyBudget } from '../../src/ai/budget';
import { distribution } from '../../src/ai/simulate';
import { hitPackets, incoming, payloadOpportunity } from '../../src/ai/threats';
import { materialValue } from '../../src/ai/evaluate';
import type { GameState, Kind } from '../../src/engine';
import type { Decision } from '../../src/ai/types';

function focusPosition() {
  const s = fixture();
  const a = add(s, 26, 1, 4, 5),
    b = add(s, 26, 1, 4, 6);
  const fragile = add(s, 26, 2, 5, 7);
  fragile.hp = 30;
  const tank = add(s, 5, 2, 6, 6);
  return { s, a, b, fragile, tank };
}
function execute(s: GameState, d: Decision) {
  for (const step of d.plan) s = applyCommand(s, step.command);
  return s;
}

test('production work budget and decisions are independent of elapsed wall time', (t) => {
  const { s } = focusPosition(),
    observation = observe(s);
  const spent = { ...emptyBudget(), ply: s.ply, nodes: 120, commands: 1 };
  const fast = allocateBudget(s, 'hard', { ...spent, ms: 1 });
  const slow = allocateBudget(s, 'hard', { ...spent, ms: 1_000_000 });
  assert.deepEqual(fast, slow);
  const ordinary = decide(observation, 1, 'hard', fast);
  let clock = 0;
  t.mock.method(performance, 'now', () => (clock += 50_000));
  assert.deepEqual(decide(observation, 1, 'hard', slow), ordinary);
  assert.deepEqual(observe(s), observation);
});
test('timed mode is explicit and returns a legal interrupted fallback', (t) => {
  const { s } = focusPosition();
  let clock = 0;
  t.mock.method(performance, 'now', () => (clock += 50));
  const result = decide(observe(s), 1, 'hard', {
    mode: 'timed',
    simulations: 800,
    milliseconds: 10,
  });
  assert.ok(result.command);
  assert.equal(commandError(s, result.command), null);
  assert.equal(result.stats.mode, 'timed');
  assert.equal(result.stats.stopReason, 'time');
  assert.equal(result.stats.exhausted, true);
});
test('short work budgets publish paired completed reply rounds, retaining an earlier round', () => {
  const { s } = focusPosition();
  const results = [50, 60, 70, 90, 140].map((simulations) =>
    decide(observe(s), 1, 'hard', { simulations, trace: true }),
  );
  let adopted = false,
    partialRetained = false;
  for (const r of results) {
    const stages = new Set(r.trace!.alternatives.map((a) => a.stage));
    assert.equal(stages.size, 1, 'never compare partial reply and static scores');
    assert.ok(r.stats.replyCandidates === 0 || r.stats.replyCandidates === 2);
    if (r.stats.replyCandidates) {
      adopted = true;
      assert.ok(r.stats.replySamples! >= 1);
      assert.equal(r.trace!.alternatives.length, r.stats.replyCandidates);
      if (r.stats.replySamples === 1) partialRetained = true;
    }
  }
  assert.ok(adopted, 'a real production-sized work budget must evaluate replies');
  assert.ok(partialRetained, 'a complete first round must survive an interrupted later round');
});
test('begin does not cache an unchecked entire play phase', () => {
  const { s } = focusPosition();
  s.phase = 'summon';
  const begin = decide(observe(s), 1, 'hard');
  assert.deepEqual(
    begin.plan.map((p) => p.command),
    [{ type: 'begin' }],
  );
  const play = applyCommand(s, begin.command!);
  const next = decide(observe(play), 1, 'hard', allocateBudget(play, 'hard', emptyBudget()));
  assert.equal(next.stats.replyCandidates, 2);
});
test('medium and hard finish a jointly killable shooter instead of chipping the 111-HP tank', () => {
  for (const level of ['medium', 'hard'] as const) {
    const { s, fragile, tank } = focusPosition();
    const result = decide(observe(s), 1, level, allocateBudget(s, level, emptyBudget()));
    const next = execute(s, result);
    assert.ok(!next.units.some((u) => u.id === fragile.id), JSON.stringify(result.plan));
    assert.equal(next.units.find((u) => u.id === tank.id)?.hp, 111);
  }
});
test('a 5-HP runner escapes a surviving threat, or heals before firing when healing saves it', () => {
  for (const healer of [false, true]) {
    const s = fixture(),
      runner = add(s, 20, 1, 4, 5);
    runner.hp = 5;
    add(s, 5, 2, 4, 7);
    if (healer) add(s, 2, 1, 3, 5);
    assert.ok(incoming(s, runner) >= runner.hp);
    const result = decide(observe(s), 1, 'hard', allocateBudget(s, 'hard', emptyBudget()));
    const next = execute(s, result),
      survivor = next.units.find((u) => u.id === runner.id)!;
    assert.ok(survivor);
    assert.ok(incoming(next, survivor) < survivor.hp, JSON.stringify(result.plan));
    if (healer) assert.ok(survivor.hp > 5);
  }
});
test('a low-cost interceptor prevents a charged ally from being converted', () => {
  const s = fixture(),
    core = add(s, 'u2', 1, 5, 5);
  core.charge = core.readyCharge = 5;
  core.operations = 1;
  const screen = add(s, 26, 1, 4, 6),
    carrier = add(s, 9, 2, 5, 9);
  carrier.hp = carrier.maxHp = 100;
  carrier.effects.push({ type: 'convert', owner: 2, from: 0, until: 100 });
  const before = payloadOpportunity(s, carrier, 'convert', (u) => materialValue(s, u));
  const result = decide(observe(s), 1, 'hard', allocateBudget(s, 'hard', emptyBudget()));
  assert.equal(result.command?.type, 'move');
  assert.equal(result.command?.unitId, screen.id);
  const next = applyCommand(s, result.command!),
    enemy = next.units.find((u) => u.id === carrier.id)!;
  assert.ok(payloadOpportunity(next, enemy, 'convert', (u) => materialValue(next, u)) < before);
});
test('rule attack profiles, AI packets and exact engine chance trees agree', () => {
  for (const kind of [1, 'u1', 'u8'] as Kind[])
    for (const kills of [0, 2, 4, 7])
      for (const silenced of [false, true]) {
        const s = fixture(),
          a = add(s, kind, 1, 4, 5),
          target = add(s, 26, 2, 4, 6);
        a.kills = kills;
        a.silenced = silenced;
        target.hp = target.maxHp = 1000;
        const packets = hitPackets(s, a, asTarget(target));
        assert.ok(Math.abs(packets.reduce((n, p) => n + p.probability, 0) - 1) < 1e-10);
        const exact = distribution(s, { type: 'attack', unitId: a.id, targetId: target.id }, 64);
        assert.equal(exact.sampled, false);
        const expected = new Map<number, number>();
        for (const p of packets)
          if (p.probability > 0)
            expected.set(p.damage, (expected.get(p.damage) ?? 0) + p.probability);
        const actual = new Map<number, number>();
        for (const o of exact.outcomes) {
          const damage = 1000 - o.state.units.find((u) => u.id === target.id)!.hp;
          actual.set(damage, (actual.get(damage) ?? 0) + o.weight);
        }
        assert.equal(actual.size, expected.size);
        for (const [damage, p] of expected)
          assert.ok(
            Math.abs(p - actual.get(damage)!) < 1e-10,
            `${kind}/${kills}/${silenced}/${damage}`,
          );
        assert.ok(
          attackProfile(kind, 20, kills, silenced, false).packets.every((p) => p.probability >= 0),
        );
      }
});
