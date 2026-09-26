import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { TinyPolicy, randomStream } from '../../scripts/training/economics/policy';
import { sampleCommand, emptyMetrics } from '../../scripts/training/economics/sample';
import { TrainingActionTree } from '../../src/ai/training/action-tree';
import { encodeDecision } from '../../src/ai/training/encoding/decision';
import { observe, fingerprint } from '../../src/ai/observation';
import { TrainingEnvironment } from '../../src/match/training';
import { withRecordOutput } from '../../scripts/training/records/io';
import { recordHeader, readTrainingRecords } from '../../scripts/training/records/replay';
import { fixture, add } from '../helpers';

test('独立小网络对候选换序等变；只读观察与随机流可重复', () => {
  const env = new TrainingEnvironment({ seed: 90, rules: 'shrine' });
  const observation = env.observation(1);
  const original = structuredClone(observation);
  const input = encodeDecision(observation, 1, new TrainingActionTree(observation, 1).node());
  const policy = new TinyPolicy(73129);
  const logits = policy.logits(input);
  assert.ok(logits.every(Number.isFinite));
  assert.ok(new Set(logits).size > 1);
  assert.deepEqual(
    policy.logits({
      ...input,
      candidates: [...input.candidates].reverse(),
      sources: [...input.sources].reverse(),
      targets: [...input.targets].reverse(),
    }),
    [...logits].reverse(),
  );
  const first = sampleCommand(observation, 1, policy, randomStream(421), emptyMetrics());
  assert.deepEqual(
    first,
    sampleCommand(observation, 1, new TinyPolicy(73129), randomStream(421), emptyMetrics()),
  );
  assert.deepEqual(observation, original);
  assert.ok(first.command);
  env.step(1, first.command);
});

test('回合外巨大化与不介入都可采样；不把无动作当作强制结束', () => {
  const state = fixture();
  add(state, 'u7', 2, 8, 8);
  add(state, 1, 1, 3, 5);
  const observation = observe(state, 2);
  let passed = false,
    acted = false;
  for (let seed = 1; seed <= 100 && !(passed && acted); seed++) {
    const result = sampleCommand(
      observation,
      2,
      new TinyPolicy(73129),
      randomStream(seed * 8191),
      emptyMetrics(),
      true,
    );
    if (result.passed) passed = true;
    if (result.command) {
      assert.equal(result.command.type, 'skill');
      TrainingEnvironment.fromState(state).step(2, result.command);
      acted = true;
    }
  }
  assert.ok(passed && acted);
});

test('神经轨迹重放按真实权限接受另一方先暗选；截断没有胜负标签', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'haojie-economics-'));
  try {
    const path = join(directory, 'record.jsonl.gz');
    const env = new TrainingEnvironment({ seed: 90, rules: 'shrine', maxCommands: 1 });
    await withRecordOutput(path, async (emit) => {
      await emit({
        type: 'game',
        ...recordHeader(env),
        game: 0,
        source: 'neural',
        seed: 90,
        rules: 'shrine',
      });
      const o = env.observation(2);
      const command = {
        type: 'choose-shrine',
        shrineKind: o.shrineDraft!.offers[2][0],
        parity: 'odd',
      } as const;
      env.step(2, command);
      await emit({
        type: 'decision',
        game: 0,
        index: 0,
        actor: 2,
        before: fingerprint(o),
        after: fingerprint(env.observation()),
        command,
      });
      await emit({
        type: 'outcome',
        game: 0,
        ...env.status(),
        after: fingerprint(env.observation()),
      });
    });
    const rows = [];
    for await (const row of readTrainingRecords(path)) rows.push(row);
    assert.equal(rows[1].observation.shrineDraft.choices[1], undefined);
    assert.equal(rows.at(-1).truncated, true);
    assert.equal(rows.at(-1).returns, null);
    const illegalPath = join(directory, 'illegal-neural.jsonl.gz');
    const normal = new TrainingEnvironment({ seed: 19 });
    await withRecordOutput(illegalPath, async (emit) => {
      await emit({
        type: 'game',
        ...recordHeader(normal),
        game: 0,
        source: 'neural',
        seed: 19,
        rules: 'classic',
      });
      await emit({
        type: 'decision',
        game: 0,
        index: 0,
        actor: 2,
        before: fingerprint(normal.observation(2)),
        after: 'invalid',
        command: { type: 'summon' },
      });
    });
    await assert.rejects(async () => {
      for await (const row of readTrainingRecords(illegalPath)) void row;
    }, /席位/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
