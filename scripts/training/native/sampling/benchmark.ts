import assert from 'node:assert/strict';
import { parseArgs } from 'node:util';
import { join, resolve } from 'node:path';
import { nativeClient } from '../client';
import { freeze } from './artifacts';
import { sampleWorker } from '../../economics/run';
import { createGame, applyCommand } from '../../../../src/engine/commands/game';
import { observe, fingerprint } from '../../../../src/ai/observation';
import { TrainingEnvironment } from '../../../../src/match/training';
import { recordHeader, readTrainingRecords } from '../../records/replay';
import { withRecordOutput } from '../../records/io';
import type { Command, Player } from '../../../../src/engine/types';

const { values } = parseArgs({
  options: {
    output: { type: 'string' },
    executable: {
      type: 'string',
      default: `artifacts/native-target/release/haojie-engine${process.platform === 'win32' ? '.exe' : ''}`,
    },
    commands: { type: 'string', default: '1000' },
    rounds: { type: 'string', default: '3' },
    rules: { type: 'string' },
    'no-warmup': { type: 'boolean', default: false },
  },
});
assert.ok(values.output);
const output = resolve(values.output),
  limit = Number(values.commands),
  rounds = Number(values.rounds);
assert.ok(Number.isSafeInteger(limit) && limit > 0 && Number.isSafeInteger(rounds) && rounds > 0);
assert.ok(!values.rules || ['classic', 'shrine'].includes(values.rules));
const { executable, save } = await freeze(
  'scripts/training/native/sampling/benchmark.ts',
  output,
  values.executable,
);
type Decision = { actor: Player; command: Command };
const canonical = <T>(v: T): T => JSON.parse(JSON.stringify(v));
const reports = [];
for (const rules of ['classic', 'shrine'] as const) {
  if (values.rules && values.rules !== rules) continue;
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
  let reference: Decision[] | undefined;
  let referenceMetrics: any;
  const run = async (native: boolean, round: number) => {
    const client = native ? await nativeClient(executable, 600_000) : null;
    try {
      const started = performance.now();
      let decisions: Decision[], status: any, metrics: any, nativeResult: any;
      if (client) {
        nativeResult = await client.request({ op: 'sample-game', ...options });
        assert.equal(nativeResult.error, null);
        decisions = nativeResult.commands;
        status = nativeResult.status;
        metrics = nativeResult.metrics;
      } else {
        const rows: any[] = [];
        const result = await sampleWorker(
          options,
          async (row) => {
            rows.push(row);
          },
          undefined,
          'commands',
        );
        assert.equal(result.results.length, 1);
        assert.equal(result.results[0].error, null);
        decisions = canonical(
          rows
            .filter((r) => r.type === 'decision')
            .map((r) => ({ actor: r.actor, command: r.command })),
        );
        status = result.results[0];
        metrics = status.metrics;
      }
      const elapsedMs = performance.now() - started;
      if (!reference) {
        reference = decisions;
        referenceMetrics = metrics;
      } else {
        const mismatch = decisions.findIndex((c, i) => {
          try {
            assert.deepEqual(c, reference![i]);
            return false;
          } catch {
            return true;
          }
        });
        if (mismatch >= 0 || reference.length !== decisions.length) {
          save(`${rules}-mismatch.json`, {
            index: mismatch,
            expected: reference,
            actual: decisions,
          });
          assert.fail(`${rules} command mismatch at ${mismatch}`);
        }
        for (const key of [
          'nodes',
          'evaluations',
          'forced',
          'backtracks',
          'maxEntities',
          'maxCandidates',
          'offTurnCommands',
          'offTurnPasses',
        ])
          assert.equal(metrics[key], referenceMetrics[key], `${rules}: ${key}`);
      }
      let state = createGame(options.seed, rules);
      state.log = [];
      state.events = [];
      for (const { actor, command } of decisions) {
        state = applyCommand(
          state,
          command.type === 'choose-shrine' ? { ...command, player: actor } : command,
        );
        state.log = [];
        state.events = [];
      }
      if (nativeResult) {
        assert.deepEqual(nativeResult.state, canonical(state));
        assert.deepEqual(
          nativeResult.observations,
          [1, 2].map((p) => canonical(observe(state, p as Player))),
        );
      }
      if (native && round === 0) {
        // 计时外通过共享 TS 引擎重放生成旧格式指纹，再交给唯一训练读取器审核。
        const env = new TrainingEnvironment(options),
          path = join(output, `${rules}.jsonl.gz`);
        await withRecordOutput(path, async (emit) => {
          await emit({
            type: 'game',
            ...recordHeader(env),
            game: 0,
            gameId: `native-sampler:${rules}:${options.seed}`,
            seed: options.seed,
            rules,
            source: 'neural',
            experiment: 'native-cold-start-economics',
          });
          for (const [index, { actor, command }] of decisions.entries()) {
            const before = fingerprint(env.observation(actor));
            env.step(actor, command);
            await emit({
              type: 'decision',
              game: 0,
              index,
              actor,
              command,
              before,
              after: fingerprint(env.observation()),
              source: 'tiny',
            });
          }
          const result = env.status();
          for (const key of [
            'commands',
            'ply',
            'phase',
            'terminated',
            'truncated',
            'truncation',
            'winner',
            'returns',
          ] as const)
            assert.deepEqual(status[key], result[key], `${rules}: ${key}`);
          await emit({
            type: 'outcome',
            game: 0,
            ...result,
            after: fingerprint(env.observation()),
            interrupted: null,
            error: null,
          });
        });
        for await (const _ of readTrainingRecords(path)) {
          /* 完整重放、权限及标签验证。 */
        }
      }
      const report = {
        backend: native ? 'rust' : 'ts',
        elapsedMs,
        commands: decisions.length,
        commandsPerSecond: (decisions.length / elapsedMs) * 1000,
        status: {
          ply: status.ply,
          terminated: status.terminated,
          truncated: status.truncated,
          returns: status.returns,
        },
        metrics,
      };
      console.log(JSON.stringify({ rules, round, ...report }));
      return report;
    } finally {
      client?.close();
    }
  };
  if (!values['no-warmup']) {
    await run(false, -1);
    await run(true, -1);
  }
  const pairs: any[] = [];
  for (let i = 0; i < rounds; i++) {
    const pair: any = {};
    for (const native of i % 2 ? [true, false] : [false, true])
      pair[native ? 'rust' : 'ts'] = await run(native, i);
    pairs.push(pair);
    save(`${rules}-round-${i}.json`, pair);
  }
  const median = (key: string) => {
    const times = pairs.map((p) => p[key].elapsedMs).sort((a, b) => a - b);
    return times[Math.floor(times.length / 2)];
  };
  reports.push({
    rules,
    options,
    pairs,
    median: { tsMs: median('ts'), rustMs: median('rust'), speedup: median('ts') / median('rust') },
    commandsEqual: true,
    workMetricsEqual: true,
    finalAuthorityAndBothObservationsEqual: true,
    boundary:
      'create + public observe + full action tree + entity encoding + tiny policy + sample + transition + in-memory commands; Rust includes one batch IPC and final export; process startup, static init, replay fingerprints, compressed file IO and audit excluded',
  });
}
save('benchmark.json', reports);
