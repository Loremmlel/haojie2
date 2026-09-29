import assert from 'node:assert/strict';
import { readFileSync, statSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { pathToFileURL } from 'node:url';
import { freeze } from './artifacts';
import { nativeClient } from '../client';
import { recordHeader } from '../../records/replay';
import * as currentApi from '../../performance/api';
import { withRecordOutput } from '../../records/io';
import type { Command, Player } from '../../../../src/engine/types';

const { values } = parseArgs({
  options: {
    output: { type: 'string' },
    executable: {
      type: 'string',
      default: 'artifacts/native-target/release/haojie-engine-prototype.exe',
    },
    references: { type: 'string', default: 'artifacts/training/rust-fresh-games-20260927' },
    workset: { type: 'string' },
    'native-source': { type: 'string' },
    api: { type: 'string' },
  },
});
assert.ok(values.output);
const output = resolve(values.output);
// 完整旧/新成本账可选择各自冻结的 TS 记录出口，不能混用新版复核后冒充旧端总成本。
const api: typeof currentApi = values.api
  ? await import(pathToFileURL(resolve(values.api)).href)
  : currentApi;
const { createGame, applyCommand, observe, fingerprint, TrainingEnvironment, readTrainingRecords } =
  api;
const { executable, save } = await freeze(
  'scripts/training/native/sampling/complete.ts',
  output,
  values.executable,
  values['native-source'],
);
const reports = [];
if (values.api)
  save('record-api.json', {
    path: resolve(values.api),
    sha256: createHash('sha256').update(readFileSync(values.api)).digest('hex'),
  });
const started = performance.now();
const cases = values.workset
  ? ([
      ['classic', 731270001],
      ['shrine', 741270001],
      ['classic', 731270017],
      ['shrine', 741270019],
    ] as const)
  : ([
      ['classic', 731270001],
      ['shrine', 741270001],
    ] as const);
for (const [rules, seed] of cases) {
  assert.ok(performance.now() - started < 7200_000, '整组原生长局验收超过两小时');
  const name = values.workset ? `${rules}-${seed}` : rules;
  const source = readFileSync(join(values.workset ?? values.references, `${name}.json`));
  const reference = JSON.parse(source.toString());
  const options = values.workset
    ? { seed, rules, policy: 'tiny', maxCommands: 12000, maxPlies: 500 }
    : reference.options;
  if (!values.workset) {
    assert.equal(options.rules, rules);
    assert.equal(options.policy, 'tiny');
    assert.equal(options.workers, 1);
    assert.equal(options.worker, 0);
    assert.equal(reference.finalStateAndBothObservationsEqual, true);
  } else {
    assert.equal(reference.seed, seed);
    assert.equal(reference.rules, rules);
  }
  const client = await nativeClient(executable, 900_000);
  try {
    // 原生采样仅接收种子/预算；已有 TS 选招轨迹仅在采样完成后用于断言，绝不送入选择器。
    const sampleStarted = performance.now();
    const native = await client.request({ op: 'sample-game', ...options });
    const sampleRequestMs = performance.now() - sampleStarted;
    const nativeUsage =
      process.platform === 'win32' && client.pid
        ? JSON.parse(
            execFileSync(
              'pwsh.exe',
              [
                '-NoProfile',
                '-NonInteractive',
                '-Command',
                `Get-Process -Id ${client.pid} | Select-Object CPU,WorkingSet64,PeakWorkingSet64,PrivateMemorySize64 | ConvertTo-Json -Compress`,
              ],
              { encoding: 'utf8', windowsHide: true },
            ),
          )
        : null;
    assert.equal(native.error, null);
    assert.equal(native.status.terminated, true);
    assert.equal(native.status.truncated, false);
    assert.deepEqual(native.commands, reference.decisions);
    const commands: { actor: Player; command: Command }[] = native.commands;
    const replayStarted = performance.now();
    let state = createGame(options.seed, rules);
    state.log = [];
    state.events = [];
    for (const { actor, command } of commands) {
      state = applyCommand(
        state,
        command.type === 'choose-shrine' ? { ...command, player: actor } : command,
      );
      state.log = [];
      state.events = [];
    }
    assert.deepEqual(native.state, JSON.parse(JSON.stringify(state)));
    if (values.workset) assert.deepEqual(native.state, reference.state);
    assert.deepEqual(
      native.observations,
      [1, 2].map((p) => JSON.parse(JSON.stringify(observe(state, p as Player)))),
    );
    const replayMs = performance.now() - replayStarted;
    const env = new TrainingEnvironment(options),
      path = join(output, `${name}.jsonl.gz`);
    const recordStarted = performance.now();
    await withRecordOutput(path, async (emit) => {
      await emit({
        type: 'game',
        ...recordHeader(env),
        game: 0,
        gameId: `native-complete:${rules}:${options.seed}`,
        seed: options.seed,
        rules,
        source: 'neural',
        experiment: 'native-cold-start-economics',
      });
      for (const [index, { actor, command }] of commands.entries()) {
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
      assert.deepEqual(env.status().returns, native.status.returns);
      await emit({
        type: 'outcome',
        game: 0,
        ...env.status(),
        after: fingerprint(env.observation()),
        interrupted: null,
        error: null,
      });
    });
    const recordExportMs = performance.now() - recordStarted;
    const auditStarted = performance.now();
    let checkedCommands = 0,
      checkedOutcomes = 0;
    for await (const row of readTrainingRecords(path)) {
      if (row.type === 'decision') checkedCommands++;
      if (row.type === 'outcome') {
        checkedOutcomes++;
        assert.equal(row.terminated, true);
      }
    }
    assert.equal(checkedCommands, commands.length);
    assert.equal(checkedOutcomes, 1);
    const report = {
      rules,
      seed: options.seed,
      commands: commands.length,
      status: native.status,
      work: native.metrics,
      sampleMs: native.elapsedMs,
      sampleRequestMs,
      replayMs,
      recordExportMs,
      auditMs: performance.now() - auditStarted,
      nativeUsage,
      nodeUsageCumulative: process.resourceUsage(),
      recordBytes: statSync(path).size,
      recordSha256: createHash('sha256').update(readFileSync(path)).digest('hex'),
      exactTsReferenceCommands: true,
      finalAuthorityAndBothObservationsEqual: true,
      verifiedTrainingRecord: path,
      referenceSha256: createHash('sha256').update(source).digest('hex'),
      note: '单次完整自然局与记录成本账；逐条命令和终局对照冻结参照。原生采样独立选招；正式记录仍经TS规则回放出口，导出及再次审计均单列，不能用此单次结果代替交错稳态倍率。',
    };
    save(`${name}.json`, report);
    reports.push(report);
    console.log(JSON.stringify(report));
  } finally {
    client.close();
  }
}
save('complete.json', reports);
