import test from 'node:test';
import assert from 'node:assert/strict';
import { appendFile, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { encodeTeacherFile } from '../../scripts/training/encode';
import { TrainingEnvironment } from '../../src/match/training';
import { recordHeader } from '../../scripts/training/records/replay';
import { fingerprint } from '../../src/ai/observation';
import { runNeuralMatches } from '../../scripts/training/neural-match';
import { withRecordOutput, readRecordLines } from '../../scripts/training/records/io';
import { readCorrections } from '../../scripts/training/corrections/records';
import { relabel } from '../../scripts/training/corrections/relabel';

test('纠错重放学生公开局面，反事实标签不继承胜负，拒绝缺尾、重复和原轨迹漂移', async () => {
  const folder = await mkdtemp(join(tmpdir(), 'haojie-correction-'));
  const collect = async (stream: AsyncIterable<any>) => {
    const rows = [];
    for await (const row of stream) rows.push(row);
    return rows;
  };
  try {
    const source = join(folder, 'student.jsonl.gz'),
      labels = join(folder, 'labels.jsonl.gz');
    await withRecordOutput(source, (emit) =>
      runNeuralMatches(
        { games: 2, seed: 127, maxCommands: 16, simulations: 40 },
        async (input) => ({ logits: input.candidates.map((_, i) => -i), value: 0 }),
        emit,
      ),
    );
    const result = await relabel(source, labels, 3, 40);
    assert.ok(result.labels > 0 && result.labels <= 6);
    const raw = await collect(readRecordLines(labels));
    assert.ok(raw.every((r) => !('observation' in r) && !('returns' in r)));
    const restored = await collect(readCorrections(labels));
    assert.equal(restored.filter((r) => r.type === 'sample').length, result.labels);
    assert.ok(
      restored
        .filter((r) => r.type === 'outcome')
        .every((r) => r.returns === null && !r.terminated),
    );
    const encoded = await collect(encodeTeacherFile(labels));
    assert.ok(encoded.some((r) => r.type === 'example'));
    assert.ok(
      encoded
        .filter((r) => r.type === 'game')
        .every((r) => r.group.endsWith(':127') && r.difficulty === 'hard'),
    );
    const broken = join(folder, 'broken.jsonl');
    for (const rows of [raw.slice(0, -1), [...raw.slice(0, -1), raw[1], raw.at(-1)]]) {
      await writeFile(broken, rows.map((r) => JSON.stringify(r)).join('\n'));
      await assert.rejects(collect(readCorrections(broken)));
    }
    raw[0].source.sha256 = 'wrong';
    await writeFile(broken, raw.map((r) => JSON.stringify(r)).join('\n'));
    await assert.rejects(collect(readCorrections(broken)), /指纹改变/);
  } finally {
    await rm(folder, { recursive: true, force: true });
  }
});

test('教师转换保留中断局的策略样本，标签与宿主种子不进入网络输入', async () => {
  const folder = await mkdtemp(join(tmpdir(), 'haojie-encoding-'));
  try {
    const path = join(folder, 'teacher.jsonl');
    const env = new TrainingEnvironment({ seed: 19 });
    const before = fingerprint(env.observation());
    env.step(1, { type: 'summon' });
    await writeFile(
      path,
      [
        {
          type: 'game',
          game: 0,
          ...recordHeader(env),
          source: 'teacher',
          rules: 'classic',
          seed: 19,
          budget: 40,
        },
        {
          type: 'sample',
          game: 0,
          index: 0,
          actor: 1,
          before,
          after: fingerprint(env.observation()),
          command: { type: 'summon' },
        },
      ]
        .map((r) => JSON.stringify(r))
        .join('\n'),
      'utf8',
    );
    const rows = [];
    for await (const row of encodeTeacherFile(path)) rows.push(row);
    const ending = rows.at(-1)!;
    assert.equal(ending.type, 'outcome');
    assert.ok('interrupted' in ending && ending.interrupted);
    assert.ok('returns' in ending && ending.returns === null);
    const sample = rows.find((r) => r.type === 'example')!;
    assert.ok('input' in sample && sample.input);
    for (const forbidden of ['seed', 'rng', 'policy', 'value', 'selected', 'observation'])
      assert.equal(forbidden in sample.input, false);
    await appendFile(
      path,
      '\n' +
        JSON.stringify({
          type: 'outcome',
          game: 0,
          ...env.status(),
          after: fingerprint(env.observation()),
          interrupted: 'cancelled',
          returns: null,
          winner: null,
        }),
    );
    const explicit = [];
    for await (const row of encodeTeacherFile(path)) explicit.push(row);
    const outcome = explicit.at(-1)!;
    assert.ok('interrupted' in outcome && outcome.interrupted);
    assert.ok('interruptionReason' in outcome && outcome.interruptionReason === 'cancelled');
    assert.ok('returns' in outcome && outcome.returns === null);
  } finally {
    await rm(folder, { recursive: true, force: true });
  }
});
