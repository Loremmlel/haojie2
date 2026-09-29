import assert from 'node:assert/strict';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { createHash } from 'node:crypto';
import type * as API from '../api';

const { values } = parseArgs({
  options: {
    output: { type: 'string' },
    baseline: { type: 'string' },
    candidate: { type: 'string' },
    workset: { type: 'string' },
    rounds: { type: 'string', default: '3' },
    commands: { type: 'string', default: '8' },
    single: { type: 'boolean', default: false },
  },
});
assert.ok(values.output && values.baseline && values.candidate && values.workset);
const output = resolve(values.output);
mkdirSync(output);
writeFileSync(join(output, 'measure.ts'), readFileSync(new URL(import.meta.url)), { flag: 'wx' });
writeFileSync(
  join(output, 'invocation.json'),
  JSON.stringify(
    { argv: process.argv, node: process.version, started: new Date().toISOString() },
    null,
    2,
  ),
  { flag: 'wx' },
);
const apis: Record<string, typeof API> = {};
for (const key of ['baseline', 'candidate'] as const)
  apis[key] = await import(pathToFileURL(resolve(values[key]!)).href);
const cases = JSON.parse(readFileSync(values.workset, 'utf8'));
const rounds = Number(values.rounds),
  commands = Number(values.commands);
assert.ok(rounds >= 1 && rounds <= 10 && commands >= 1 && commands <= 64);
const canonical = (v: unknown) => JSON.parse(JSON.stringify(v));
const references: any[] = [];
const log = console.log;
// 原采样器逐局日志只影响可读性；诊断行仍完整保存，算法与循环不变。
console.log = () => {};
const run = async (label: string, verify: boolean) => {
  const api = apis[label],
    rows = [];
  const probe = (globalThis as any).__runtimeProbe;
  if (probe) probe.rows = {};
  const before = process.resourceUsage();
  for (const [i, row] of cases.entries()) {
    const decisions: any[] = [];
    const env = api.TrainingEnvironment.fromState(row.state, {
      maxCommands: commands,
      maxPlies: 500,
    });
    const start = performance.now();
    if (probe) probe.enabled = true;
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
    const elapsedMs = performance.now() - start;
    if (probe) probe.enabled = false;
    const outcome = result.results[0];
    assert.equal(outcome.error, null, row.path);
    assert.equal(outcome.interrupted, null, row.path);
    let replayed = row.state;
    for (const { actor, command } of decisions) {
      replayed = api.applyPlayerCommand(replayed, actor, command);
      replayed.log = [];
      replayed.events = [];
    }
    const observations = [1, 2].map((p) => env.observation(p as 1 | 2));
    assert.deepEqual(
      canonical(observations),
      canonical([1, 2].map((p) => api.observe(replayed, p as 1 | 2))),
    );
    const evidence = canonical({ decisions, state: replayed, observations, status: env.status() });
    if (verify) assert.deepEqual(evidence, references[i], `${row.path}:${row.index}`);
    else references[i] = evidence;
    rows.push({
      path: row.path,
      index: row.index,
      elapsedMs,
      commands: decisions.length,
      metrics: outcome.metrics,
      heapUsed: process.memoryUsage().heapUsed,
      rss: process.memoryUsage().rss,
    });
  }
  return {
    elapsedMs: rows.reduce((n, r) => n + r.elapsedMs, 0),
    rows,
    probe: probe ? structuredClone(probe.rows) : undefined,
    cpuUserUs: process.resourceUsage().userCPUTime - before.userCPUTime,
    cpuSystemUs: process.resourceUsage().systemCPUTime - before.systemCPUTime,
    maxRSS: process.resourceUsage().maxRSS,
  };
};
const save = (name: string, value: unknown) =>
  writeFileSync(join(output, name), JSON.stringify(value, null, 2), { flag: 'wx' });
const warmup = { baseline: await run('baseline', false), candidate: await run('candidate', true) };
save('warmup.json', warmup);
const measurements = [];
for (let i = 0; i < rounds; i++) {
  const pair: any = {};
  for (const key of values.single
    ? ['candidate']
    : i % 2
      ? ['candidate', 'baseline']
      : ['baseline', 'candidate'])
    pair[key] = await run(key, true);
  measurements.push(pair);
  save(`round-${i}.json`, pair);
  log(
    JSON.stringify({
      round: i + 1,
      baselineMs: pair.baseline?.elapsedMs,
      candidateMs: pair.candidate.elapsedMs,
      speedup: pair.baseline?.elapsedMs / pair.candidate.elapsedMs,
    }),
  );
}
const sha256 = (path: string) => createHash('sha256').update(readFileSync(path)).digest('hex');
save('summary.json', {
  worksetSha256: sha256(values.workset),
  baselineSha256: sha256(values.baseline),
  candidateSha256: sha256(values.candidate),
  commands,
  rounds,
  measurements,
});
