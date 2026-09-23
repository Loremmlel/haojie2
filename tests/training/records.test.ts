import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { runSelfPlay } from '../../scripts/training/self-play';
import { readTrainingRecords } from '../../scripts/training/records/replay';
import { withRecordOutput } from '../../scripts/training/records/io';
import { encodeTeacherFile } from '../../scripts/training/encode';
import { TrainingEnvironment } from '../../src/match/training';

async function collect(rows: AsyncIterable<any>) {
  const result = [];
  for await (const row of rows) result.push(row);
  return result;
}

test('压缩和普通增量轨迹重建相同操作者观察与训练输入，神龛暗选仍脱敏', async () => {
  const folder = await mkdtemp(join(tmpdir(), 'haojie-records-'));
  try {
    for (const rules of ['classic', 'shrine'] as const) {
      const rows: any[] = [];
      await runSelfPlay(
        { games: 2, seed: 90, rules, maxCommands: 8, simulations: 40 },
        async (row) => {
          rows.push(row);
        },
      );
      assert.ok(rows.every((row) => !('observation' in row)));
      const paths = ['.jsonl', '.jsonl.gz'].map((suffix) => join(folder, rules + suffix));
      for (const path of paths)
        await withRecordOutput(path, async (emit) => {
          for (const row of rows) await emit(row);
        });
      const restored = await collect(readTrainingRecords(paths[1]));
      assert.deepEqual(restored, await collect(readTrainingRecords(paths[0])));
      assert.deepEqual(
        await collect(encodeTeacherFile(paths[1])),
        await collect(encodeTeacherFile(paths[0])),
      );
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
      assert.ok((await readFile(paths[1])).length < (await readFile(paths[0])).length / 2);
    }
  } finally {
    await rm(folder, { recursive: true, force: true });
  }
});

test('旧格式、顺序、规则版本、操作者、前后指纹和伪造结果均拒绝，缺失结束行不给价值标签', async () => {
  const folder = await mkdtemp(join(tmpdir(), 'haojie-invalid-records-'));
  try {
    const rows: any[] = [];
    await runSelfPlay({ games: 1, seed: 19, maxCommands: 3, simulations: 40 }, async (row) => {
      rows.push(row);
    });
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
        await runSelfPlay(
          { games: 1, seed: 19, maxCommands: 3, simulations: 40 },
          async (row: any) => {
            if (row.type === 'outcome') throw new Error('生成器失败');
            await emit(row);
          },
        );
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
