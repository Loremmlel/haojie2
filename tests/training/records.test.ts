import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { recordHeader, readTrainingRecords } from '../../scripts/training/records/replay';
import { withRecordOutput } from '../../scripts/training/records/io';
import { TrainingEnvironment } from '../../src/match/training';
import { fingerprint } from '../../src/ai/observation';

// 使用固定正式命令构造审核夹具，不重新引入旧教师采样器。
function fixtureRecords(rules: 'classic' | 'shrine', seed: number) {
  const env = new TrainingEnvironment({ seed, rules, maxCommands: 1 });
  const rows: any[] = [
    { type: 'game', game: 0, source: 'teacher', ...recordHeader(env), rules, seed },
  ];
  const observation = env.observation(1);
  const command =
    rules === 'classic'
      ? { type: 'summon' as const }
      : {
          type: 'choose-shrine' as const,
          player: 1 as const,
          shrineKind: observation.shrineDraft!.offers[1][0],
          parity: 'even' as const,
        };
  env.step(1, command);
  rows.push({
    type: 'sample',
    game: 0,
    index: 0,
    actor: 1,
    command,
    before: fingerprint(observation),
    after: fingerprint(env.observation()),
  });
  rows.push({ type: 'outcome', game: 0, ...env.status(), after: fingerprint(env.observation()) });
  return rows;
}

async function collect(rows: AsyncIterable<any>) {
  const result = [];
  for await (const row of rows) result.push(row);
  return result;
}

test('压缩和普通增量轨迹重建相同操作者观察，神龛暗选仍脱敏', async () => {
  const folder = await mkdtemp(join(tmpdir(), 'haojie-records-'));
  try {
    for (const rules of ['classic', 'shrine'] as const) {
      const rows = fixtureRecords(rules, 90);
      assert.ok(rows.every((row) => !('observation' in row)));
      const paths = ['.jsonl', '.jsonl.gz'].map((suffix) => join(folder, rules + suffix));
      for (const path of paths)
        await withRecordOutput(path, async (emit) => {
          for (const row of rows) await emit(row);
        });
      const restored = await collect(readTrainingRecords(paths[1]));
      assert.deepEqual(restored, await collect(readTrainingRecords(paths[0])));
      let env: TrainingEnvironment;
      for (const row of restored) {
        if (row.type === 'game')
          env = new TrainingEnvironment({ rules, seed: row.seed, ...row.limits });
        if (row.type === 'sample') {
          assert.deepEqual(row.observation, env!.observation(row.actor));
          assert.equal('seed' in row.observation, false);
          assert.equal('rng' in row.observation, false);
          if (row.observation.shrineDraft && !row.observation.shrineDraft.revealed)
            assert.equal(row.observation.shrineDraft.choices[3 - row.actor], undefined);
          env!.step(row.actor, row.command);
        }
      }
      assert.deepEqual([...(await readFile(paths[1])).subarray(0, 2)], [0x1f, 0x8b]);
    }
  } finally {
    await rm(folder, { recursive: true, force: true });
  }
});

test('旧格式、顺序、规则版本、操作者、前后指纹和伪造结果均拒绝，缺失结束行不给价值标签', async () => {
  const folder = await mkdtemp(join(tmpdir(), 'haojie-invalid-records-'));
  try {
    const rows = fixtureRecords('classic', 19);
    const path = join(folder, 'test.jsonl');
    const mutations: ((copy: any[]) => void)[] = [
      (r) => {
        delete r[0].format;
      },
      (r) => {
        r[0].recordRuleset = 'old';
      },
      (r) => {
        r[0].limits.maxCommands = 0;
      },
      (r) => {
        r[1].observation = {};
      },
      (r) => {
        r[1].index++;
      },
      (r) => {
        r[1].actor = 2;
      },
      (r) => {
        r[1].before = 'bad';
      },
      (r) => {
        r[1].after = 'bad';
      },
      (r) => {
        r.at(-1).returns = { 1: 1, 2: -1 };
      },
      (r) => {
        r.at(-1).truncated = false;
      },
    ];
    for (const mutate of mutations) {
      const copy = structuredClone(rows);
      mutate(copy);
      await writeFile(path, copy.map((r) => JSON.stringify(r)).join('\n'));
      await assert.rejects(collect(readTrainingRecords(path)));
    }
    await writeFile(
      path,
      rows
        .slice(0, -1)
        .map((r) => JSON.stringify(r))
        .join('\n'),
    );
    const last = (await collect(readTrainingRecords(path))).at(-1);
    assert.equal(last.interrupted, 'missing-outcome');
    assert.equal(last.truncated, false);
    assert.equal(last.returns, null);
  } finally {
    await rm(folder, { recursive: true, force: true });
  }
});

test('压缩截断和坏JSON不会被当成正常中断；生成失败仍封好gzip且不覆盖旧文件', async () => {
  const folder = await mkdtemp(join(tmpdir(), 'haojie-record-io-'));
  try {
    const path = join(folder, 'teacher.jsonl.gz');
    await assert.rejects(
      withRecordOutput(path, async (emit) => {
        for (const row of fixtureRecords('classic', 19)) {
          if (row.type === 'outcome') throw new Error('生成器失败');
          await emit(row);
        }
      }),
      /生成器失败/,
    );
    assert.equal((await collect(readTrainingRecords(path))).at(-1).interrupted, 'missing-outcome');
    await assert.rejects(
      withRecordOutput(path, async () => {}),
      /EEXIST/,
    );
    const data = await readFile(path);
    await writeFile(path, data.subarray(0, data.length - 6));
    await assert.rejects(collect(readTrainingRecords(path)));
    const malformed = join(folder, 'bad.jsonl');
    await writeFile(malformed, '{"type":');
    await assert.rejects(collect(readTrainingRecords(malformed)));
  } finally {
    await rm(folder, { recursive: true, force: true });
  }
});
