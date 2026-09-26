import test from 'node:test';
import assert from 'node:assert/strict';
import { fixture, add } from '../helpers';
import { observe } from '../../src/ai/observation';
import { decodeCommand, type PolicyEvaluator } from '../../src/ai/training/decoder';
import { COMMANDS } from '../../src/ai/training/encoding/schema';
import { inspectTrainingCommand } from '../../src/ai/training/queries';
import { beamDecode } from '../../scripts/training/improvement/beam';
import { policySearch, halvingVisits } from '../../scripts/training/improvement/search';
import { softExamples } from '../../scripts/training/improvement/labels';
import { positions } from '../../scripts/training/search/positions';
import { reference } from '../../scripts/training/search/reference';

test('完整命令概率会惩罚分散的参数质量；束命令合法且取消后不交付旧结果', async () => {
  const s = fixture();
  add(s, 1, 1, 5, 5);
  const o = observe(s),
    before = structuredClone(o);
  const evaluator = (): PolicyEvaluator => {
    let calls = 0;
    return async (input) => ({
      logits: input.candidates.map((row) =>
        calls ? 0 : row[COMMANDS.indexOf('move')] ? 1 : row[COMMANDS.indexOf('end')] ? 0 : -50,
      ),
      value: (calls++, 0),
    });
  };
  const greedy = await decodeCommand(o, 1, evaluator());
  const beam = await beamDecode(o, 1, evaluator());
  assert.equal(greedy.command?.type, 'move');
  assert.equal(beam.command?.type, 'end');
  assert.ok(beam.candidates.length > 1);
  assert.ok(beam.beam.retainedMass > 0 && beam.beam.retainedMass <= 1 + 1e-9);
  for (const c of beam.candidates)
    assert.notEqual(inspectTrainingCommand(o, 1, c.command).status, 'invalid');
  assert.deepEqual(o, before);
  const cancelled = new AbortController();
  const stopped = await beamDecode(
    o,
    1,
    async (input) => {
      cancelled.abort();
      return { logits: input.candidates.map(() => 0), value: 0 };
    },
    { signal: cancelled.signal },
  );
  assert.equal(stopped.status, 'paused');
  assert.equal(stopped.reason, 'cancelled');
  assert.equal(stopped.command, undefined);
});

test('有限候选随机树使用真实机会抽样、根视角回报和同方连续行动，Gumbel共用相同终局口径', async () => {
  const evaluate: PolicyEvaluator = async (input) => ({
    logits: input.candidates.map((r) =>
      r[COMMANDS.indexOf('attack')] ? 6 : r[COMMANDS.indexOf('react')] ? 6 : 0,
    ),
    value: -1,
  });
  for (const mode of ['mcts', 'gumbel'] as const)
    for (const p of positions().filter(
      (p) =>
        p.x === 5 &&
        p.actor === 1 &&
        ['immediate-win', 'same-player-two-attacks', 'chance-win'].includes(p.family),
    )) {
      const before = structuredClone(p.observation);
      const oracle = reference(p.observation, 2);
      const result = await policySearch(p.observation, p.actor, evaluate, {
        mode,
        simulations: 128,
        horizon: 2,
        sampleSeed: 42,
      });
      assert.equal(result.status, 'command');
      assert.equal(result.search.stats.simulations, 128);
      assert.ok(result.search.stats.transitions >= 128 && result.search.stats.transitions <= 256);
      const value = oracle.rootValues.find(
        (r) => JSON.stringify(r.command) === JSON.stringify(result.command),
      )?.value;
      assert.equal(value, oracle.best, `${mode}:${p.family}`);
      assert.ok(result.search.stats.sameActor > 0);
      if (p.family === 'chance-win') assert.ok(result.search.stats.chanceOutcomes > 2);
      assert.deepEqual(p.observation, before);
    }
  const visits = halvingVisits(8, 16);
  assert.equal(visits.length, 16);
  assert.equal(visits.filter((v) => v === 0).length, 8);
});

test('候选软质量按当前前缀归一，不给未到达的参数凭空分配监督', () => {
  const s = fixture(),
    unit = add(s, 1, 1, 5, 5);
  const o = observe(s);
  const command = { type: 'move' as const, unitId: unit.id, x: 5, y: 6 };
  const examples = softExamples(o, 1, command, [
    { command, probability: 0.8 },
    { command: { type: 'end' }, probability: 0.2 },
  ]);
  assert.equal(examples[0].policy[examples[0].selected], 0.8);
  assert.equal(examples[1].policy[examples[1].selected], 1);
  for (const row of examples) assert.ok(Math.abs(row.policy.reduce((a, b) => a + b, 0) - 1) < 1e-9);
});

test('未校准阶段使整次价值比较退回基线，价值异步返回后的取消不允许落子', async () => {
  const s = fixture();
  add(s, 1, 1, 5, 5);
  const o = observe(s);
  const evaluate: PolicyEvaluator = async (input) => ({
    logits: input.candidates.map((r) => (r[COMMANDS.indexOf('end')] ? 8 : -8)),
    value: -1,
  });
  let calls = 0;
  const uncovered = await policySearch(o, 1, evaluate, {
    simulations: 1,
    valuePhases: ['play'],
    leafValue: async () => {
      calls++;
      return 0.5;
    },
  });
  assert.match(uncovered.search.fallback!, /uncovered-value-phase/);
  assert.equal(uncovered.command?.type, 'end');
  assert.equal(calls, 0);
  const controller = new AbortController();
  const cancelled = await policySearch(o, 1, evaluate, {
    signal: controller.signal,
    simulations: 1,
    valuePhases: ['play', 'synthesis', 'summon'],
    leafValue: async (_, root) => {
      assert.equal(root, 1);
      controller.abort();
      return 0.5;
    },
  });
  assert.equal(cancelled.reason, 'cancelled');
  assert.equal(cancelled.command, undefined);
});
