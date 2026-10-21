// 固定采样段的双端性能对照；冻结输入与产物后交错计时，逐段核对实际工作。
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { gunzipSync } from 'node:zlib';
import { nativeClient } from '../client';
import { freeze, nativeSourcePaths } from '../sampling/artifacts';
import type * as API from './api';
import type { GameState } from '../../../../src/engine/types';

const { values } = parseArgs({
  options: {
    mode: { type: 'string', default: 'compare' },
    variant: { type: 'string' },
    before: { type: 'string' },
    after: { type: 'string' },
    executable: { type: 'string' },
    workset: { type: 'string' },
    output: { type: 'string' },
    rounds: { type: 'string', default: '5' },
    repetitions: { type: 'string', default: '3' },
    commands: { type: 'string', default: '8' },
    backend: { type: 'string', default: 'both' },
  },
});
const hash = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');
const canonical = <T>(value: T): T => JSON.parse(JSON.stringify(value));
const digest = (value: unknown): string => {
  const sorted = (v: any): any =>
    Array.isArray(v)
      ? v.map(sorted)
      : v && typeof v === 'object'
        ? Object.fromEntries(
            Object.keys(v)
              .sort()
              .map((k) => [k, sorted(v[k])]),
          )
        : v;
  return hash(JSON.stringify(sorted(canonical(value))));
};
const save = (value: unknown) => {
  assert.ok(values.output);
  writeFileSync(resolve(values.output), JSON.stringify(value, null, 2), { flag: 'wx' });
};
const positive = (text: string) => {
  const number = Number(text);
  assert.ok(Number.isSafeInteger(number) && number > 0);
  return number;
};
const apiFor = async (directory: string): Promise<typeof API> =>
  import(pathToFileURL(resolve(directory, 'runner.mjs')).href);

interface Entry {
  name: string;
  state: GameState;
}
interface Workset {
  format: 'haojie-engine-workset-v1';
  commands: number;
  maxPlies: number;
  entries: Entry[];
}

if (values.mode === 'freeze') {
  assert.ok(values.output && values.executable);
  const frozen = await freeze(
    'scripts/training/native/endurance/api.ts',
    resolve(values.output),
    resolve(values.executable),
  );
  frozen.save(
    'build.json',
    JSON.parse(
      execFileSync(frozen.executable, ['--version'], {
        encoding: 'utf8',
        windowsHide: true,
      }),
    ),
  );
  frozen.save(
    'native-source.json',
    Object.fromEntries(nativeSourcePaths().map((path) => [path, readFileSync(path, 'utf8')])),
  );
} else if (values.mode === 'prepare') {
  assert.ok(values.variant);
  const api = await apiFor(values.variant);
  const entries: Entry[] = [
    ...api.fixtures(),
    ...api.combatFixtures(),
    ...api.preparationFixtures(),
    ...api.completeFixtures(),
  ]
    .filter((_, index) => index % 10 === 0)
    .filter(({ job }) => job.state.phase !== 'shrine-draft' || job.state.shrineDraft)
    .map(({ name, job }) => ({ name: `fixture:${name}`, state: canonical(job.state) }));
  const source = readFileSync('tests/fixtures/native/late-starts.json.gz');
  const games = JSON.parse(gunzipSync(source).toString('utf8'));
  for (const game of games) {
    let state = api.createGame(game.seed, game.rules);
    const checkpoints = new Set([0, 100, Math.floor(game.prelude.length / 2), game.prelude.length]);
    for (let index = 0; index <= game.prelude.length; index++) {
      state.log = [];
      state.events = [];
      if (checkpoints.has(index))
        entries.push({ name: `${game.rules}:${game.seed}:${index}`, state: canonical(state) });
      if (index < game.prelude.length) {
        const { actor, command } = game.prelude[index];
        state = api.applyPlayerCommand(state, actor, command);
      }
    }
    console.log(JSON.stringify({ prepared: game.seed, commands: game.prelude.length }));
  }
  save({
    format: 'haojie-engine-workset-v1',
    commands: positive(values.commands),
    maxPlies: 500,
    entries,
    provenance: {
      variant: resolve(values.variant),
      apiSha256: hash(readFileSync(resolve(values.variant, 'runner.mjs'))),
      lateStartsSha256: hash(source),
    },
  });
} else {
  assert.ok(['run', 'compare'].includes(values.mode));
  assert.ok(values.workset && ['both', 'ts', 'rust'].includes(values.backend));
  const bytes = readFileSync(resolve(values.workset));
  const workset: Workset = JSON.parse(bytes.toString('utf8'));
  assert.equal(workset.format, 'haojie-engine-workset-v1');
  const repetitions = positive(values.repetitions);
  const rounds = positive(values.rounds);
  const backends = values.backend === 'both' ? ['ts', 'rust'] : [values.backend];
  type Client = Awaited<ReturnType<typeof nativeClient>>;
  interface Variant {
    directory: string;
    api: typeof API;
    client?: Client;
  }
  const variants: Record<string, Variant> = {};
  try {
    for (const [name, directory] of Object.entries(
      values.mode === 'compare'
        ? { before: values.before, after: values.after }
        : { current: values.variant },
    )) {
      assert.ok(directory);
      variants[name] = {
        directory: resolve(directory),
        api: await apiFor(directory),
        client: backends.includes('rust')
          ? await nativeClient(resolve(directory, 'engine.exe'), 1_800_000)
          : undefined,
      };
    }
    async function run(variant: Variant, backend: string) {
      const rows = [];
      const started = performance.now();
      const timing = { internalMs: 0, engineMs: 0, rulesMs: 0, inferenceMs: 0 };
      const groups: Record<string, typeof timing> = {};
      for (let repeat = 0; repeat < repetitions; repeat++) {
        for (const { name, state } of workset.entries) {
          let result: any;
          let commands: any[];
          let final: GameState;
          let observations: unknown;
          let status: any;
          if (backend === 'rust') {
            result = await variant.client!.request(
              {
                op: 'sample-game',
                seed: state.seed,
                rules: state.mode ?? 'classic',
                initialState: state,
                maxCommands: workset.commands,
                maxPlies: workset.maxPlies,
                policy: 'tiny',
              },
              true,
            );
            commands = result.commands;
            final = result.state;
            observations = result.observations;
            status = result.status;
          } else {
            const api = variant.api;
            const env = api.TrainingEnvironment.fromState(state, {
              maxCommands: workset.commands,
              maxPlies: workset.maxPlies,
            });
            commands = [];
            const sampled = await api.sampleWorker(
              {
                rules: state.mode ?? 'classic',
                seed: state.seed,
                maxCommands: workset.commands,
                maxPlies: workset.maxPlies,
                seconds: 1800,
                games: 1,
                workers: 1,
                worker: 0,
                policy: 'tiny',
              },
              async (row: any) => {
                if (row.type === 'decision')
                  commands.push({ actor: row.actor, command: row.command });
              },
              () => env,
              'commands',
            );
            result = sampled.results[0];
            assert.ok(
              result.interrupted === null || result.interrupted === 'decode-budget',
              `${name}: ${result.interrupted}`,
            );
            observations = [env.observation(1), env.observation(2)];
            status = env.status();
            if (result.interrupted === 'decode-budget')
              status = { ...status, truncated: true, truncation: 'decode-budget' };
            // 权威字段在计时段外独立重放；公开观察不能替代正式随机流和序号检查。
            final = canonical(state);
            final.log = [];
            final.events = [];
            for (const { actor, command } of commands) {
              final = api.applyPlayerCommand(final, actor, command);
              final.log = [];
              final.events = [];
            }
          }
          assert.equal(result.error, null, name);
          const metrics = result.metrics;
          // 原生 acceptance 的正式结算在现有 stepMs 之外；从总量扣非规则阶段才不会漏计。
          const segment = {
            internalMs: result.elapsedMs,
            inferenceMs: metrics.inferenceMs,
            engineMs: result.elapsedMs - metrics.inferenceMs,
            rulesMs:
              result.elapsedMs -
              metrics.inferenceMs -
              metrics.encodingMs -
              metrics.samplingMs -
              metrics.recordMs,
          };
          const group = (groups[name.startsWith('fixture:') ? 'fixtures' : 'natural'] ??= {
            internalMs: 0,
            engineMs: 0,
            rulesMs: 0,
            inferenceMs: 0,
          });
          for (const key of Object.keys(timing) as (keyof typeof timing)[]) {
            timing[key] += segment[key];
            group[key] += segment[key];
          }
          rows.push({
            name,
            authorityAndCommands: digest({ commands, state: final, observations }),
            counters: Object.fromEntries(
              Object.entries(metrics).filter(([key]) => !key.endsWith('Ms')),
            ),
            status: {
              commands: commands.length,
              ply: status.ply,
              phase: status.phase,
              terminated: status.terminated,
              truncated: status.truncated,
              truncation: status.truncation,
              winner: status.winner,
              returns: status.returns,
            },
          });
        }
      }
      return { wallMs: performance.now() - started, ...timing, groups, rows };
    }
    // 各版本先完整预热；每轮反转先后次序，避免只取较早慢基线与较晚快结果。
    let expected: unknown;
    for (const variant of Object.values(variants)) {
      for (const backend of backends) {
        const warm = await run(variant, backend);
        if (expected) assert.deepEqual(warm.rows, expected, `warm:${variant.directory}:${backend}`);
        else expected = warm.rows;
      }
    }
    const pairs: Record<string, any>[] = [];
    for (let round = 0; round < rounds; round++) {
      const pair: Record<string, any> = {};
      const order = Object.keys(variants);
      if (round % 2) order.reverse();
      for (const backend of backends) {
        for (const name of order) {
          const report = await run(variants[name], backend);
          assert.deepEqual(report.rows, expected, `${round}:${name}:${backend}`);
          const { rows: _, ...timing } = report;
          (pair[name] ??= {})[backend] = timing;
        }
      }
      pairs.push(pair);
      console.log(JSON.stringify({ round, pair }));
    }
    const median = (numbers: number[]) => {
      const sorted = [...numbers].sort((a, b) => a - b);
      const middle = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
    };
    save({
      format: 'haojie-engine-sampling-benchmark-v1',
      driverSha256: hash(readFileSync(fileURLToPath(import.meta.url))),
      worksetSha256: hash(bytes),
      fixtures: workset.entries.length,
      commands: workset.commands,
      repetitions,
      rounds,
      backends,
      identity: Object.fromEntries(
        Object.entries(variants).map(([name, variant]) => [
          name,
          {
            directory: variant.directory,
            apiSha256: hash(readFileSync(join(variant.directory, 'runner.mjs'))),
            engineSha256: hash(readFileSync(join(variant.directory, 'engine.exe'))),
            build: existsSync(join(variant.directory, 'build.json'))
              ? JSON.parse(readFileSync(join(variant.directory, 'build.json'), 'utf8'))
              : undefined,
            manifest: JSON.parse(readFileSync(join(variant.directory, 'manifest.json'), 'utf8')),
          },
        ]),
      ),
      equivalent: true,
      cases: expected,
      pairs,
      medians: Object.fromEntries(
        Object.keys(variants).map((name) => [
          name,
          Object.fromEntries(
            backends.map((backend) => [
              backend,
              Object.fromEntries(
                ['internalMs', 'engineMs', 'rulesMs', 'inferenceMs'].map((key) => [
                  key,
                  median(pairs.map((pair) => pair[name][backend][key])),
                ]),
              ),
            ]),
          ),
        ]),
      ),
      note: 'engineMs 为内部完整采样减 TinyPolicy 前向；rulesMs 再扣编码、策略随机排序和记录，保留导入、规则与循环开销。原生 stepMs 漏正式 acceptance 结算，不能用分项之和充当全规则时间。wallMs 还含宿主往返、摘要和 TS 独立权威重放，不用于规则引擎倍率。',
    });
  } finally {
    for (const variant of Object.values(variants)) variant.client?.close();
  }
}
