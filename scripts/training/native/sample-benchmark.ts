import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync, copyFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { cpus } from 'node:os';
import { build } from 'esbuild';
import { sampleWorker } from '../economics/run';
import { nativeClient } from './client';
import { nativeEnvironment } from './environment';
import { createGame, applyCommand } from '../../../src/engine/commands/game';
import { observe } from '../../../src/ai/observation';
import { withRecordOutput } from '../records/io';
import { readTrainingRecords } from '../records/replay';

const { values } = parseArgs({
  options: {
    output: { type: 'string' },
    commands: { type: 'string', default: '1000' },
    executable: {
      type: 'string',
      default: 'artifacts/native-target/release/haojie-engine-prototype.exe',
    },
  },
});
assert.ok(values.output);
const output = resolve(values.output),
  limit = Number(values.commands);
assert.ok(Number.isSafeInteger(limit) && limit > 0);
mkdirSync(output, { recursive: false });
const frozen = await build({
  entryPoints: ['scripts/training/native/sample-benchmark.ts'],
  bundle: true,
  platform: 'node',
  format: 'esm',
  target: 'node22',
  packages: 'external',
  write: false,
  metafile: true,
});
writeFileSync(join(output, 'runner.mjs'), frozen.outputFiles[0].contents);
const hash = (v: string | Buffer) => createHash('sha256').update(v).digest('hex');
const executable = join(output, 'engine.exe');
copyFileSync(values.executable, executable);
writeFileSync(
  join(output, 'manifest.json'),
  JSON.stringify(
    {
      cpu: cpus()[0].model,
      node: process.version,
      executableSha256: hash(readFileSync(executable)),
      sources: Object.fromEntries(
        Object.keys(frozen.metafile.inputs).map((path) => [path, hash(readFileSync(path))]),
      ),
      scope:
        'TS selector/encoder/tiny policy; interchangeable TS or native Rust authority; JSON IPC included; no GPU',
    },
    null,
    2,
  ),
);

const reports: any[] = [];
for (const rules of ['classic', 'shrine'] as const) {
  const options = {
    seconds: 3600,
    games: 1,
    workers: 1,
    worker: 0,
    rules,
    policy: 'tiny' as const,
    seed: rules === 'classic' ? 731270001 : 741270001,
    maxCommands: limit,
    maxPlies: 500,
  };
  let reference: any[] | undefined;
  const rounds: any[] = [];
  const run = async (native: boolean, round: number) => {
    const rows: any[] = [];
    const client = native ? await nativeClient(executable) : null;
    const started = performance.now();
    try {
      const result = await sampleWorker(
        options,
        async (row) => {
          rows.push(row);
        },
        client ? (options) => nativeEnvironment(client, options) : undefined,
      );
      const elapsedMs = performance.now() - started;
      assert.equal(result.results.length, 1);
      assert.equal(result.results[0].error, null);
      assert.ok(result.results[0].terminated || result.results[0].truncated);
      const decisions = rows
        .filter((r) => r.type === 'decision')
        .map((r) => ({ actor: r.actor, command: r.command }));
      if (!reference) reference = decisions;
      else assert.deepEqual(decisions, reference, `${rules}: 相同策略/抽样随机源必须选择相同命令`);
      let state = createGame(options.seed, rules);
      state.events = [];
      state.log = [];
      for (const { actor, command } of decisions) {
        state = applyCommand(
          state,
          command.type === 'choose-shrine' ? { ...command, player: actor } : command,
        );
        state.events = [];
        state.log = [];
      }
      if (client) {
        assert.deepEqual(
          (await client.request({ op: 'export' })).state,
          JSON.parse(JSON.stringify(state)),
        );
        for (const viewer of [1, 2] as const)
          assert.deepEqual(
            await client.request({ op: 'observe', viewer }),
            JSON.parse(JSON.stringify(observe(state, viewer))),
          );
      }
      // 只保存经共享读取器验证的正式 TS 指纹记录；Rust JSON 键序不能冒充旧格式指纹。
      if (!native && round === 0) {
        const path = join(output, `${rules}.jsonl.gz`);
        await withRecordOutput(path, async (emit) => {
          for (const row of rows) await emit(row);
        });
        for await (const _ of readTrainingRecords(path)) {
          /* 校验完整轨迹及实际截断上限。 */
        }
      }
      return {
        backend: native ? 'rust-with-ts-selector' : 'ts',
        elapsedMs,
        commands: decisions.length,
        commandsPerSecond: decisions.length / (elapsedMs / 1000),
        terminated: result.results[0].terminated,
        truncated: result.results[0].truncated,
        returns: result.results[0].returns,
        metrics: result.results[0].metrics,
      };
    } finally {
      client?.close();
    }
  };
  // 同一完整工作量预热，再交错三轮；计时之外验证轨迹、原生权威终态和双方公开观察。
  await run(false, -1);
  await run(true, -1);
  for (let round = 0; round < 3; round++) {
    const pair: any = {};
    for (const native of round % 2 ? [true, false] : [false, true])
      pair[native ? 'rust' : 'ts'] = await run(native, round);
    rounds.push(pair);
    console.log(JSON.stringify({ rules, round, ...pair }));
  }
  const median = (key: 'ts' | 'rust') =>
    rounds.map((r) => r[key].elapsedMs).sort((a, b) => a - b)[1];
  const report = {
    rules,
    options,
    rounds,
    median: {
      tsMs: median('ts'),
      rustMs: median('rust'),
      speedup: median('ts') / median('rust'),
    },
    commandsEqual: true,
    finalAuthorityAndBothObservationsEqual: true,
    pureNativeSelector: false,
  };
  reports.push(report);
  writeFileSync(join(output, 'benchmark.json'), JSON.stringify(reports, null, 2));
}
