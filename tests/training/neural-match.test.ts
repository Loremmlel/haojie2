import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { runNeuralMatches } from '../../scripts/training/neural-match';
import { summarizeMatches } from '../../scripts/training/neural-report';
import { PythonPolicy } from '../../scripts/training/python-policy';
import { COMMANDS } from '../../src/ai/training/encoding/schema';

test('CLI交换模型席位且保留显式截断；报告不把截断或推理中断计为胜负', async () => {
  const rows: any[] = [];
  await runNeuralMatches(
    { games: 2, seed: 987, maxCommands: 4, simulations: 40 },
    async (input) => {
      assert.deepEqual(
        Object.keys(input).sort(),
        [
          'entities',
          'kinds',
          'entity_mask',
          'globals',
          'candidates',
          'sources',
          'targets',
          'candidate_mask',
        ].sort(),
      );
      return { logits: input.candidates.map((c) => c[COMMANDS.indexOf('summon')]), value: 0 };
    },
    async (row) => {
      rows.push(row);
    },
  );
  assert.deepEqual(
    rows.filter((r) => r.type === 'game').map((r) => [r.seed, r.networkPlayer]),
    [
      [987, 1],
      [987, 2],
    ],
  );
  assert.equal(
    rows.filter((r) => r.type === 'outcome' && r.truncated && r.returns === null).length,
    2,
  );
  const folder = await mkdtemp(join(tmpdir(), 'haojie-network-'));
  try {
    const path = join(folder, 'games.jsonl');
    await writeFile(path, rows.map((r) => JSON.stringify(r)).join('\n'));
    const summary = await summarizeMatches(path);
    assert.equal(summary.truncatedGames, 2);
    assert.equal(summary.networkWins + summary.teacherWins + summary.draws, 0);
    assert.equal(summary.rejectedCommands, 0);
    assert.equal(summary.network.decisions + summary.teacher.decisions, 8);
    assert.ok((await readFile(path, 'utf8')).length > 0);
  } finally {
    await rm(folder, { recursive: true });
  }

  const paused: any[] = [];
  let commandsBeforeFailure = 0;
  await runNeuralMatches(
    { games: 1, simulations: 40 },
    async () => {
      commandsBeforeFailure = paused.filter((r) => r.type === 'decision').length;
      throw new Error('设备故障');
    },
    async (row) => {
      paused.push(row);
    },
  );
  assert.equal(paused.at(-1).interrupted, 'inference-error');
  assert.equal(paused.at(-1).commands, commandsBeforeFailure);
  assert.equal(paused.at(-1).returns, null);
});

test('宿主取消后不提交网络旧结果；启动失败的子进程及时返回错误', async () => {
  const cancellation = new AbortController(),
    rows: any[] = [];
  let commandsBeforeCancel = 0;
  await runNeuralMatches(
    { games: 1, signal: cancellation.signal },
    async (input) => {
      commandsBeforeCancel = rows.filter((r) => r.type === 'decision').length;
      cancellation.abort();
      return { logits: input.candidates.map(() => 0), value: 0 };
    },
    async (row) => {
      rows.push(row);
    },
  );
  assert.equal(rows.at(-1).interrupted, 'cancelled');
  assert.equal(rows.at(-1).commands, commandsBeforeCancel);
  await assert.rejects(
    PythonPolicy.start({
      python: join(tmpdir(), 'nonexistent-haojie-python.exe'),
      checkpoint: 'unused',
      device: 'cpu',
      precision: 'fp32',
      threads: 1,
      timeoutMs: 1000,
    }),
    /ENOENT/,
  );
});
