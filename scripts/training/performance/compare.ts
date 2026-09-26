import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { cpus } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { build } from 'esbuild';
import type * as API from './api';

const { values } = parseArgs({
  options: {
    output: { type: 'string' },
    baseline: { type: 'string' },
    rounds: { type: 'string', default: '3' },
    cases: { type: 'string', default: '32' },
  },
});
assert.ok(values.output, '必须指定新的 --output 目录；未给 baseline 时只冻结源码。');
const output = resolve(values.output);
mkdirSync(output, { recursive: false });
const digest = (data: string | Buffer) => createHash('sha256').update(data).digest('hex');
const built = await build({
  entryPoints: ['scripts/training/performance/api.ts'],
  bundle: true,
  platform: 'node',
  format: 'esm',
  target: 'node22',
  write: false,
  metafile: true,
});
const bundlePath = join(output, 'api.mjs');
writeFileSync(bundlePath, built.outputFiles[0].contents, { flag: 'wx' });
writeFileSync(
  join(output, 'manifest.json'),
  JSON.stringify(
    {
      head: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
      node: process.version,
      cpu: cpus()[0].model,
      bundleSha256: digest(Buffer.from(built.outputFiles[0].contents)),
      sources: Object.fromEntries(
        Object.keys(built.metafile.inputs).map((path) => [path, digest(readFileSync(path))]),
      ),
    },
    null,
    2,
  ),
  { flag: 'wx' },
);
if (!values.baseline) {
  console.log(JSON.stringify({ frozen: bundlePath }));
} else {
  const baselinePath = resolve(values.baseline);
  const baseline: typeof API = await import(pathToFileURL(baselinePath).href);
  const current: typeof API = await import(pathToFileURL(bundlePath).href);
  const count = Number(values.cases);
  const rounds = Number(values.rounds);
  assert.ok(Number.isSafeInteger(count) && count >= 4 && count <= 128 && count % 4 === 0);
  assert.ok(Number.isSafeInteger(rounds) && rounds >= 1 && rounds <= 10);
  const records = ['classic-tiny', 'shrine-tiny', 'classic-uniform', 'shrine-uniform'].map(
    (name) => `artifacts/training/economics-20260926/${name}/worker-0.jsonl.gz`,
  );
  const cases: any[] = [];
  for (const path of records) {
    const rows: any[] = [];
    // 从已校验轨迹重建，只在内存保留公开观察；磁盘仍保存种子和命令。
    for await (const row of baseline.readTrainingRecords(path))
      if (row.type === 'decision') rows.push(row);
    for (let i = 0; i < count / 4; i++) {
      const row = rows[Math.floor(((i + 0.5) * rows.length) / (count / 4))];
      cases.push({ ...row, path });
    }
  }
  const signatures: string[] = [];
  const run = (api: typeof API, verify: boolean) => {
    const policy = new api.TinyPolicy(73129);
    const metrics = api.emptyMetrics();
    let elapsedMs = 0;
    for (const [i, row] of cases.entries()) {
      const before = JSON.stringify(row.observation);
      const started = performance.now();
      const result = api.sampleCommand(
        row.observation,
        row.actor,
        policy,
        api.randomStream(982451653 + i),
        metrics,
      );
      elapsedMs += performance.now() - started;
      assert.equal(JSON.stringify(row.observation), before, '公开观察被修改');
      const signature = JSON.stringify(result);
      if (verify) assert.equal(signature, signatures[i], `${row.path}:${row.index} 选择不同`);
      else signatures[i] = signature;
    }
    return { elapsedMs, metrics };
  };
  // 先用相同输入预热；计时顺序逐轮反转，排除编译、读盘和轨迹重放成本。
  run(baseline, false);
  run(current, true);
  const measurements: { baseline: ReturnType<typeof run>; current: ReturnType<typeof run> }[] = [];
  for (let i = 0; i < rounds; i++) {
    let a: ReturnType<typeof run>, b: ReturnType<typeof run>;
    if (i % 2 === 0) {
      a = run(baseline, true);
      b = run(current, true);
    } else {
      b = run(current, true);
      a = run(baseline, true);
    }
    for (const key of ['nodes', 'evaluations', 'forced', 'backtracks'] as const)
      assert.equal(a.metrics[key], b.metrics[key], `采样工作量改变：${key}`);
    measurements.push({ baseline: a, current: b });
    console.log(JSON.stringify({ round: i + 1, speedup: a.elapsedMs / b.elapsedMs }));
  }
  const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
  const baselineMs = median(measurements.map((m) => m.baseline.elapsedMs));
  const currentMs = median(measurements.map((m) => m.current.elapsedMs));
  const report = {
    baseline: baselinePath,
    baselineSha256: digest(readFileSync(baselinePath)),
    records: records.map((path) => ({ path, sha256: digest(readFileSync(path)) })),
    cases: cases.map((r) => ({ path: r.path, game: r.game, index: r.index, before: r.before })),
    rounds,
    baselineMs,
    currentMs,
    speedup: baselineMs / currentMs,
    selectionsEqual: true,
    measurements,
  };
  writeFileSync(join(output, 'summary.json'), JSON.stringify(report, null, 2), { flag: 'wx' });
  console.log(JSON.stringify({ baselineMs, currentMs, speedup: report.speedup, cases: count }));
}
