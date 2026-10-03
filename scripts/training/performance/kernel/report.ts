// 汇总已完成的交错测量；中位总耗时、成对倍率与局面尾延迟分别计算，不能相互替代。
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';

const { values } = parseArgs({ options: { root: { type: 'string' }, output: { type: 'string' } } });
assert.ok(values.root && values.output);
const inputs: Record<string, string> = {};
const read = (path: string) => {
  const data = readFileSync(join(values.root!, path));
  inputs[path] = createHash('sha256').update(data).digest('hex');
  return JSON.parse(data.toString('utf8'));
};
const median = (values: number[]) => {
  assert.ok(values.length);
  const a = values.slice().sort((a, b) => a - b);
  return (a[Math.floor((a.length - 1) / 2)] + a[Math.floor(a.length / 2)]) / 2;
};
const sum = (values: number[]) => values.reduce((a, b) => a + b, 0);
const p95 = (values: number[]) =>
  values.slice().sort((a, b) => a - b)[Math.ceil(values.length * 0.95) - 1];
const sides = ['baseline', 'candidate'] as const;
const engine = (folder: string) => {
  const report = read(`${folder}/summary.json`);
  const rounds = report.measurements.map((pair: any) =>
    Object.fromEntries(
      sides.map((side) => {
        const { elapsedMs, requestMs, rows } = pair[side];
        const metrics = Object.fromEntries(
          Object.keys(rows[0].metrics)
            .filter((k) => k.endsWith('Ms'))
            .map((key) => [key, sum(rows.map((r: any) => r.metrics[key]))]),
        );
        const commands = sum(rows.map((r: any) => r.commands));
        return [
          side,
          {
            elapsedMs,
            requestMs,
            commands,
            commandsPerSecond: (commands * 1000) / elapsedMs,
            caseP95Ms: p95(rows.map((r: any) => r.elapsedMs)),
            withoutForwardMs: elapsedMs - metrics.inferenceMs,
            metrics,
          },
        ];
      }),
    ),
  );
  const medians = Object.fromEntries(
    sides.map((side) => [
      side,
      {
        ...Object.fromEntries(
          Object.keys(rounds[0][side])
            .filter((k) => typeof rounds[0][side][k] === 'number')
            .map((key) => [key, median(rounds.map((r: any) => r[side][key]))]),
        ),
        metrics: Object.fromEntries(
          Object.keys(rounds[0][side].metrics).map((key) => [
            key,
            median(rounds.map((r: any) => r[side].metrics[key])),
          ]),
        ),
      },
    ]),
  );
  return {
    rounds,
    medians,
    speedups: rounds.map((r: any) => r.baseline.elapsedMs / r.candidate.elapsedMs),
    pairedSpeedupMedian: median(
      rounds.map((r: any) => r.baseline.elapsedMs / r.candidate.elapsedMs),
    ),
    withoutForwardSpeedupMedian: median(
      rounds.map((r: any) => r.baseline.withoutForwardMs / r.candidate.withoutForwardMs),
    ),
  };
};
const ts = engine('ts-accepted-paired'),
  rust = engine('rust-accepted-paired');
const app = read('application-accepted-paired/summary.json');
const application = {
  pairedSpeedupMedian: app.speedup_median,
  rounds: app.pairs.map((pair: any) =>
    Object.fromEntries(
      sides.map((side) => [
        side,
        {
          seconds: pair[side].seconds,
          phases: pair[side].phases,
          metrics: pair[side].metrics,
        },
      ]),
    ),
  ),
  medians: Object.fromEntries(
    sides.map((side) => [
      side,
      {
        seconds: median(app.pairs.map((r: any) => r[side].seconds)),
        phases: Object.fromEntries(
          Object.keys(app.pairs[0][side].phases).map((key) => [
            key,
            median(app.pairs.map((r: any) => r[side].phases[key])),
          ]),
        ),
      },
    ]),
  ),
};
const profiles: Record<string, any> = {};
for (const row of read('rust-separated-cost-profile/profile.json').rows)
  for (const p of row.profile.rows) {
    const total = (profiles[p.name] ??= {
      calls: 0,
      selfMs: 0,
      inclusiveMs: 0,
      allocations: [0, 0, 0, 0],
    });
    total.calls += p.calls;
    total.selfMs += p.selfMs;
    total.inclusiveMs += p.inclusiveMs;
    p.selfAllocations.forEach((n: number, i: number) => (total.allocations[i] += n));
  }
const report = {
  ts,
  rust,
  application,
  profiles,
  applicationProfile: read('application-separated-cost-profile/profile.json'),
  delivery: read('delivery/manifest.json'),
  inputHashes: inputs,
  note: '倍率为逐轮旧/新之比的中位数；局面 p95 是 111 个连续 8 命令段的 nearest-rank p95，再取轮间中位数。分项中位数不可相加。探针累计申请/释放不是峰值或复制字节。',
};
writeFileSync(values.output, JSON.stringify(report, null, 2), { flag: 'wx' });
console.log(
  JSON.stringify({
    ts: ts.pairedSpeedupMedian,
    rust: rust.pairedSpeedupMedian,
    application: application.pairedSpeedupMedian,
  }),
);
