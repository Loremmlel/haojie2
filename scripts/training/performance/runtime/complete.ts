import assert from 'node:assert/strict';
import { mkdirSync, readFileSync, writeFileSync, statSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { withRecordOutput } from '../../records/io';
import type * as API from '../api';

const { values } = parseArgs({
  options: {
    output: { type: 'string' },
    api: { type: 'string' },
    workset: { type: 'string' },
  },
});
assert.ok(values.output && values.api && values.workset);
const output = resolve(values.output);
mkdirSync(output);
const save = (name: string, data: unknown) =>
  writeFileSync(join(output, name), JSON.stringify(data, null, 2), { flag: 'wx' });
const digest = (p: string) => createHash('sha256').update(readFileSync(p)).digest('hex');
writeFileSync(join(output, 'complete.ts'), readFileSync(new URL(import.meta.url)), { flag: 'wx' });
save('manifest.json', {
  argv: process.argv,
  node: process.version,
  apiSha256: digest(values.api),
  started: new Date().toISOString(),
});
const api: typeof API = await import(pathToFileURL(resolve(values.api)).href);
const reports = [],
  started = performance.now(),
  usage = process.resourceUsage();
for (const [rules, seed] of [
  ['classic', 731270001],
  ['shrine', 741270001],
  ['classic', 731270017],
  ['shrine', 741270019],
] as const) {
  assert.ok(performance.now() - started < 7200_000, '整组验收超过两小时');
  const referencePath = join(values.workset, `${rules}-${seed}.json`);
  const reference = JSON.parse(readFileSync(referencePath, 'utf8'));
  const options = {
    seconds: 900,
    games: 1,
    workers: 1,
    worker: 0,
    rules,
    policy: 'tiny' as const,
    seed,
    maxCommands: 12000,
    maxPlies: 500,
  };
  const path = join(output, `${rules}-${seed}.jsonl.gz`),
    decisions: any[] = [];
  const t = performance.now();
  const sampled = await withRecordOutput(path, (emit) =>
    api.sampleWorker(options, async (row: any) => {
      if (row.type === 'decision') decisions.push({ actor: row.actor, command: row.command });
      await emit(row);
    }),
  );
  const sampleAndWriteMs = performance.now() - t,
    outcome = sampled.results[0];
  assert.equal(outcome.error, null);
  assert.equal(outcome.interrupted, null);
  assert.equal(outcome.terminated, true);
  assert.equal(outcome.truncated, false);
  assert.deepEqual(decisions, reference.decisions);
  const auditStart = performance.now();
  let checked = 0,
    terminal = 0;
  for await (const row of api.readTrainingRecords(path)) {
    if (row.type === 'decision') checked++;
    if (row.type === 'outcome') {
      assert.equal(row.terminated, true);
      terminal++;
    }
  }
  assert.equal(checked, decisions.length);
  assert.equal(terminal, 1);
  const report = {
    rules,
    seed,
    sampleAndWriteMs,
    auditMs: performance.now() - auditStart,
    outcome,
    exactFrozenCommands: true,
    checked,
    recordBytes: statSync(path).size,
    recordSha256: digest(path),
    referenceSha256: digest(referencePath),
    memory: process.memoryUsage(),
  };
  reports.push(report);
  save(`${rules}-${seed}.json`, report);
  console.log(
    JSON.stringify({
      rules,
      seed,
      commands: checked,
      sampleAndWriteMs: report.sampleAndWriteMs,
      auditMs: report.auditMs,
    }),
  );
}
const after = process.resourceUsage();
save('summary.json', {
  reports,
  elapsedMs: performance.now() - started,
  cpuUserUs: after.userCPUTime - usage.userCPUTime,
  cpuSystemUs: after.systemCPUTime - usage.systemCPUTime,
  maxRSS: after.maxRSS,
  note: '单次自然完整采样与正式记录成本账；不是稳态多轮性能倍率。',
});
