import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { TrainingService } from '../../scripts/training/protocol';
import { runSelfPlay } from '../../scripts/training/self-play';

test('常驻批量接口隔离环境、逐项报告错误，参数错误不会先落子或替换环境', () => {
  const service = new TrainingService();
  const run = (input: unknown) => service.handle(input) as any;
  run({ op: 'reset', env: 'a', options: { seed: 7 } });
  run({ op: 'reset', env: 'b', options: { seed: 9 } });
  const a = run({ op: 'observe', env: 'a' }).result;
  const batch = run({
    op: 'batch',
    requests: [
      { op: 'step', env: 'a', actor: 2, command: { type: 'summon' } },
      { op: 'step', env: 'b', actor: 1, command: { type: 'summon' }, observe: false },
    ],
  });
  assert.equal(batch.results[0].ok, false);
  assert.equal(batch.results[1].result.commands, 1);
  assert.equal('observation' in batch.results[1].result, false);
  assert.deepEqual(run({ op: 'observe', env: 'a' }).result, a);
  assert.equal(run({ op: 'reset', env: 'a', options: { seed: 0 } }).ok, false);
  assert.equal(run({ op: 'step', env: 'a', command: { type: 'summon' }, viewer: 9 }).ok, false);
  assert.deepEqual(run({ op: 'observe', env: 'a' }).result, a);
  const sampled = run({ op: 'sample', env: 'a', command: { type: 'summon' }, sampleSeed: 42 });
  assert.equal(sampled.ok, true);
  assert.equal('seed' in sampled.result, false);
  assert.deepEqual(run({ op: 'observe', env: 'a' }).result, a);
  const stepped = run({ op: 'step', env: 'a', viewer: 2, command: { type: 'summon' } });
  assert.equal(stepped.ok, true);
  assert.equal(stepped.result.commands, 1);
  assert.equal(stepped.result.viewer, 2);
  assert.ok(stepped.result.observation.hands[1].length > 0);
  run({ op: 'close', env: 'a' });
  assert.equal(run({ op: 'observe', env: 'a' }).ok, false);
});

test('JSONL进程接受连续请求和批量请求，非法JSON后仍可继续，stdout没有界面文本', () => {
  const result = spawnSync(process.execPath, ['--import', 'tsx', 'scripts/training/serve.ts'], {
    input:
      [
        JSON.stringify({ id: 1, op: 'reset', env: 'a', options: { seed: 7 } }),
        '{broken',
        JSON.stringify({
          id: 2,
          op: 'batch',
          requests: [
            { op: 'step', env: 'a', command: { type: 'summon' }, observe: false },
            { op: 'step', env: 'a', command: { type: 'summon' }, observe: false },
            { op: 'step', env: 'a', command: { type: 'begin' } },
          ],
        }),
      ].join('\n') + '\n',
    encoding: 'utf8',
    timeout: 15000,
  });
  assert.equal(result.status, 0, result.stderr);
  const rows = result.stdout
    .trim()
    .split('\n')
    .map((line) => JSON.parse(line));
  assert.equal(rows.length, 3);
  assert.equal(rows[1].ok, false);
  assert.equal(rows[2].results[2].result.phase, 'play');
  assert.equal(rows[2].results[2].result.commands, 3);
});

test('热启动样本可按终局记录分组，教师无trace/伪MCTS标签，取消不自动补落子', async () => {
  const rows: any[] = [];
  const report = await runSelfPlay(
    { games: 1, seed: 7, maxCommands: 3, simulations: 40 },
    async (r) => {
      rows.push(r);
    },
  );
  assert.equal(report.results.length, 1);
  assert.equal(report.results[0].truncated, true);
  assert.equal(report.results[0].returns, null);
  assert.equal(rows.filter((r) => r.type === 'sample').length, 3);
  for (const r of rows.filter((r) => r.type === 'sample')) {
    assert.equal('seed' in r.observation, false);
    assert.equal('rng' in r.observation, false);
    assert.equal('trace' in r, false);
    assert.equal('visits' in r, false);
  }
  const controller = new AbortController();
  let samples = 0;
  const cancelled = await runSelfPlay(
    { games: 1, seed: 7, simulations: 40, signal: controller.signal },
    async (row) => {
      if ((row as any).type === 'sample') {
        samples++;
        controller.abort();
      }
    },
  );
  assert.equal(cancelled.cancelled, true);
  assert.equal(samples, 1);
  assert.equal(cancelled.results.length, 0);
});
