import assert from 'node:assert/strict';
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { nativeClient } from '../../native/client';
import type * as API from '../api';

const { values } = parseArgs({
  options: {
    output: { type: 'string' },
    baseline: { type: 'string' },
    candidate: { type: 'string' },
    original: { type: 'string' },
    api: { type: 'string' },
    workset: { type: 'string' },
    rounds: { type: 'string', default: '5' },
    commands: { type: 'string', default: '8' },
  },
});
assert.ok(values.output && values.baseline && values.candidate && values.api && values.workset);
const output = resolve(values.output);
mkdirSync(output);
const digest = (path: string) => createHash('sha256').update(readFileSync(path)).digest('hex');
const save = (name: string, value: unknown) =>
  writeFileSync(join(output, name), JSON.stringify(value, null, 2), {
    flag: 'wx',
  });
writeFileSync(join(output, 'native.ts'), readFileSync(new URL(import.meta.url)), { flag: 'wx' });
const rounds = Number(values.rounds),
  commands = Number(values.commands);
assert.ok(Number.isInteger(rounds) && rounds >= 1 && rounds <= 10);
assert.ok(Number.isInteger(commands) && commands >= 1 && commands <= 64);
save('manifest.json', {
  argv: process.argv,
  started: new Date().toISOString(),
  node: process.version,
  worksetSha256: digest(values.workset),
  apiSha256: digest(values.api),
  baselineSha256: digest(values.baseline),
  candidateSha256: digest(values.candidate),
  originalSha256: values.original ? digest(values.original) : undefined,
  clientSha256: digest('scripts/training/native/client.ts'),
});
for (const key of ['baseline', 'candidate'] as const)
  copyFileSync(values[key]!, join(output, `${key}.exe`));
const api: typeof API = await import(pathToFileURL(resolve(values.api)).href);
const cases = JSON.parse(readFileSync(values.workset, 'utf8'));
const canonical = (v: unknown) => JSON.parse(JSON.stringify(v));
const workKeys = [
  'nodes',
  'evaluations',
  'forced',
  'backtracks',
  'maxEntities',
  'maxCandidates',
  'offTurnCommands',
  'offTurnPasses',
] as const;
const started = performance.now(),
  references: any[] = [],
  log = console.log;
console.log = () => {};
// TS 参照生成在所有原生计时之前完成；每个原生请求独立采样，不接收参照命令。
for (const row of cases) {
  const env = api.TrainingEnvironment.fromState(row.state, {
      maxCommands: commands,
      maxPlies: 500,
    }),
    decisions: any[] = [];
  const result = await api.sampleWorker(
    {
      seconds: 900,
      games: 1,
      workers: 1,
      worker: 0,
      rules: row.rules,
      policy: 'tiny',
      seed: row.seed,
      maxCommands: commands,
      maxPlies: 500,
    },
    async (r: any) => {
      if (r.type === 'decision') decisions.push({ actor: r.actor, command: r.command });
    },
    () => env,
    'commands',
  );
  const outcome = result.results[0];
  assert.equal(outcome.error, null);
  assert.equal(outcome.interrupted, null);
  let state = row.state;
  for (const r of decisions) {
    state = api.applyPlayerCommand(state, r.actor, r.command);
    state.log = [];
    state.events = [];
  }
  references.push(
    canonical({
      commands: decisions,
      state,
      observations: [env.observation(1), env.observation(2)],
      metrics: outcome.metrics,
    }),
  );
}
const clients: Partial<Record<'baseline' | 'candidate', Awaited<ReturnType<typeof nativeClient>>>> =
  {};
const startupMs: Record<string, number> = {};
// 只读本驱动自己启动的原生进程；Windows 返回进程 CPU、峰值工作集和私有提交量。
const usage = (pid: number | undefined) => {
  if (process.platform !== 'win32' || !Number.isSafeInteger(pid)) return null;
  return JSON.parse(
    execFileSync(
      'pwsh.exe',
      [
        '-NoProfile',
        '-NonInteractive',
        '-Command',
        `Get-Process -Id ${pid} | Select-Object CPU,WorkingSet64,PeakWorkingSet64,PrivateMemorySize64 | ConvertTo-Json -Compress`,
      ],
      { encoding: 'utf8', windowsHide: true },
    ),
  );
};
let current: unknown;
try {
  for (const key of ['baseline', 'candidate'] as const) {
    const t = performance.now();
    clients[key] = await nativeClient(join(output, `${key}.exe`), 900_000);
    startupMs[key] = performance.now() - t;
  }
  if (values.original) {
    const original = await nativeClient(resolve(values.original), 900_000);
    try {
      for (const [rules, seed] of [
        ['classic', 731270001],
        ['shrine', 741270001],
      ] as const) {
        const request = {
          op: 'sample-game',
          rules,
          seed,
          policy: 'tiny',
          maxCommands: 32,
          maxPlies: 500,
        };
        const a = await original.request(request);
        const b = await clients.baseline!.request(request);
        for (const key of ['commands', 'state', 'observations', 'status', 'error'])
          assert.deepEqual(a[key], b[key], `旧基线起点适配:${rules}:${key}`);
        for (const key of workKeys) assert.equal(a.metrics[key], b.metrics[key]);
      }
      save('adapter-check.json', {
        rules: ['classic', 'shrine'],
        commandsEach: 32,
        originalEqual: true,
      });
    } finally {
      original.close();
    }
  }
  const run = async (label: 'baseline' | 'candidate') => {
    const client = clients[label]!,
      rows = [];
    const before = usage(client.pid);
    for (const [i, row] of cases.entries()) {
      assert.ok(performance.now() - started < 7200_000, '整组原生验收超过两小时');
      current = { label, path: row.path, index: row.index };
      const t = performance.now();
      const result = await client.request({
        op: 'sample-game',
        initialState: row.state,
        seed: row.seed,
        rules: row.rules,
        policy: 'tiny',
        maxCommands: commands,
        maxPlies: 500,
      });
      const requestMs = performance.now() - t;
      assert.equal(result.error, null);
      for (const key of ['commands', 'state', 'observations'] as const)
        assert.deepEqual(
          result[key],
          references[i][key],
          `${label}:${row.path}:${row.index}:${key}`,
        );
      for (const key of workKeys)
        assert.equal(result.metrics[key], references[i].metrics[key], key);
      rows.push({
        path: row.path,
        index: row.index,
        elapsedMs: result.elapsedMs,
        requestMs,
        commands: result.commands.length,
        metrics: result.metrics,
      });
    }
    return {
      rows,
      elapsedMs: rows.reduce((n, r) => n + r.elapsedMs, 0),
      requestMs: rows.reduce((n, r) => n + r.requestMs, 0),
      before,
      after: usage(client.pid),
    };
  };
  save('warmup.json', {
    baseline: await run('baseline'),
    candidate: await run('candidate'),
  });
  const measurements = [];
  for (let i = 0; i < rounds; i++) {
    const pair: any = {};
    for (const key of i % 2
      ? (['candidate', 'baseline'] as const)
      : (['baseline', 'candidate'] as const))
      pair[key] = await run(key);
    measurements.push(pair);
    save(`round-${i}.json`, pair);
    log(
      JSON.stringify({
        round: i + 1,
        baselineMs: pair.baseline.elapsedMs,
        candidateMs: pair.candidate.elapsedMs,
        speedup: pair.baseline.elapsedMs / pair.candidate.elapsedMs,
      }),
    );
  }
  save('summary.json', {
    startupMs,
    rounds,
    commands,
    cases: cases.length,
    measurements,
    elapsedMs: performance.now() - started,
    note: '原生内部计时包含初始局面导入与完整采样；requestMs 另含 JSON 请求/完整终态及双观察导出。逐轮比对冻结 TS 的命令、权威状态、双观察和离散工作量。',
  });
  // 外部拥有型规则出口单列，加载和结果对照不混进重复执行计时。
  const jobs = cases.map((row: any, i: number) => {
    const selected = references[i].commands[0];
    const command =
      selected.command.type === 'choose-shrine'
        ? { ...selected.command, player: selected.actor }
        : selected.command;
    return { state: row.state, command, probes: [] };
  });
  const publicOld = await clients.baseline!.request({ op: 'run', jobs });
  const publicNew = await clients.candidate!.request({ op: 'run', jobs });
  assert.deepEqual(publicNew, publicOld);
  for (const client of Object.values(clients)) {
    await client!.request({ op: 'load', jobs });
    await client!.request({ op: 'bench', repeats: 2 });
  }
  const publicRounds = [];
  for (let round = 0; round < 3; round++) {
    const pair: any = {};
    for (const label of round % 2
      ? (['candidate', 'baseline'] as const)
      : (['baseline', 'candidate'] as const))
      pair[label] = await clients[label]!.request({ op: 'bench', repeats: 32 });
    publicRounds.push(pair);
  }
  save('public.json', { cases: jobs.length, repeats: 32, rounds: publicRounds, exactStates: true });
} catch (error) {
  save('failure.json', { current, error: String(error) });
  throw error;
} finally {
  for (const client of Object.values(clients)) client?.close();
}
