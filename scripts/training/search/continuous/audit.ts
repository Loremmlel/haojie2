import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { readTrainingRecords } from '../../records/replay';
import { searchExamples } from '../bootstrap/encoding';
import { fingerprint, hash } from '../../../../src/ai/observation';

// 逐局重放实际命令与访问候选；回执、模型席位、调用量和结束原因必须一致。
const result = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const digest = (path: string) => createHash('sha256').update(readFileSync(path)).digest('hex');
assert.equal(digest(result.output), result.sha256);
let count = 0,
  roots = 0,
  primaryCalls = 0,
  opponentCalls = 0;
let header: any, outcome: any;
for await (const row of readTrainingRecords(result.output)) {
  if (row.type === 'game') header = row;
  if (row.type === 'outcome') outcome = row;
  if (row.type !== 'sample') continue;
  count++;
  assert.equal(
    row.searchSeed,
    hash(`bootstrap-experiment:2026092851:${fingerprint(row.observation)}`),
  );
  const stats = row.searchStats;
  assert.ok(
    Object.values(stats).every((x) => typeof x === 'number' && Number.isFinite(x) && x >= 0),
  );
  assert.ok(stats.teacherCalls <= 17 && stats.teacherWork <= stats.teacherCalls * 256);
  assert.ok(stats.searchTransitions + stats.rolloutTransitions <= 32);
  const calls = row.valueStats?.calls ?? 0;
  assert.ok(Number.isInteger(calls) && calls >= 0 && calls <= 16);
  if (row.valueStats)
    assert.ok(
      [row.valueStats.min, row.valueStats.max].every(
        (v) => Number.isFinite(v) && Math.abs(v) <= 0.25,
      ),
    );
  if (result.kind === 'selfplay' || row.actor === result.primary) primaryCalls += calls;
  else opponentCalls += calls;
  if (row.policyMode === 'search') {
    assert.equal(stats.searchSimulations, 16);
    const examples = searchExamples(row.observation, row.actor, row.command, row.searchPolicy);
    assert.equal(examples.filter((e) => e.step === 0).length, 1);
    roots++;
  } else assert.equal(row.searchPolicy, null);
}
assert.ok(header && outcome && !outcome.interrupted && !outcome.error);
assert.equal(header.seed, result.seed);
assert.equal(header.experimentKind, result.kind);
assert.equal(header.primaryPlayer, result.primary);
assert.equal(count, result.commands);
for (const key of ['terminated', 'truncated', 'winner', 'returns'])
  assert.deepEqual(outcome[key], result[key]);
if (result.model) {
  assert.equal(header.valueModelSha256, result.model.checkpoint_sha256);
  assert.equal(digest(result.checkpoint), result.model.checkpoint_sha256);
  assert.equal(primaryCalls, result.inference.calls);
}
if (result.opponentModel) {
  assert.equal(header.opponentModelSha256, result.opponentModel.checkpoint_sha256);
  assert.equal(digest(result.opponentCheckpoint), result.opponentModel.checkpoint_sha256);
  assert.equal(opponentCalls, result.opponentInference.calls);
} else assert.equal(opponentCalls, 0);
writeFileSync(
  process.argv[3],
  JSON.stringify(
    {
      passed: true,
      count,
      roots,
      primaryCalls,
      opponentCalls,
      resultSha256: digest(process.argv[2]),
    },
    null,
    2,
  ),
  { flag: 'wx' },
);
