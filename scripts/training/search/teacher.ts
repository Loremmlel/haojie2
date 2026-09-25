import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { positions } from './positions';
import { encodingSourceHash } from '../encode';
import { decide } from '../../../src/ai/planning/search';
import { TrainingActionTree } from '../../../src/ai/training/action-tree';

/** 主探针之后追加的探索性教师参照；使用原30局面，不据教师表现重选题或调搜索参数。 */
const { values } = parseArgs({ options: { input: { type: 'string' } } });
assert.ok(values.input, '指定--input目录');
const directory = values.input;
const protocol = JSON.parse(readFileSync(resolve(directory, 'protocol.json'), 'utf8'));
const references: any[] = JSON.parse(readFileSync(resolve(directory, 'reference.json'), 'utf8'));
assert.equal(encodingSourceHash(), protocol.sourceSha256);
const configs = [
  { difficulty: 'easy' as const, simulations: 40 },
  { difficulty: 'hard' as const, simulations: 800 },
];
const digest = (value: Buffer | string) => createHash('sha256').update(value).digest('hex');
writeFileSync(
  resolve(directory, 'teacher-protocol.json'),
  JSON.stringify(
    {
      exploratory: true,
      configs,
      sourceSha256: protocol.sourceSha256,
      baseProtocolSha256: digest(readFileSync(resolve(directory, 'protocol.json'))),
      scriptSha256: digest(readFileSync('scripts/training/search/teacher.ts')),
    },
    null,
    2,
  ),
  { flag: 'wx' },
);
const rows = [];
for (const p of positions()) {
  assert.equal(
    digest(JSON.stringify(p.observation)),
    protocol.cases.find((c: any) => c.name === p.name).observationSha256,
  );
  const oracle = references.find((r) => r.name === p.name);
  for (const config of configs) {
    const before = JSON.stringify(p.observation);
    const start = performance.now();
    const decision = decide(p.observation, p.actor, config.difficulty, {
      simulations: config.simulations,
      milliseconds: 1000,
      mode: 'work',
    });
    const elapsedMs = performance.now() - start;
    assert.equal(JSON.stringify(p.observation), before);
    assert.ok(decision.command, p.name);
    const trace = new TrainingActionTree(p.observation, p.actor).trace(decision.command);
    const last = trace.at(-1)!;
    const canonical = last.node.choices[last.selected].command;
    const value = oracle.rootValues.find(
      (r: any) => JSON.stringify(r.command) === JSON.stringify(canonical),
    )?.value;
    assert.notEqual(value, undefined, '教师完整命令必须落在同一参照域');
    rows.push({
      name: p.name,
      family: p.family,
      actor: p.actor,
      ...config,
      command: decision.command,
      canonical,
      elapsedMs,
      stats: decision.stats,
      value,
      best: oracle.best,
      regret: oracle.best - value,
      optimal: Math.abs(oracle.best - value) < 1e-9,
    });
  }
}
assert.equal(encodingSourceHash(), protocol.sourceSha256);
const groups = [];
for (const family of ['all', ...new Set(rows.map((r) => r.family))])
  for (const { difficulty } of configs) {
    const selected = rows.filter(
      (r) => r.difficulty === difficulty && (family === 'all' || r.family === family),
    );
    groups.push({
      family,
      difficulty,
      decisions: selected.length,
      optimal: selected.filter((r) => r.optimal).length,
      meanRegret: selected.reduce((n, r) => n + r.regret, 0) / selected.length,
      elapsedMs: selected.reduce((n, r) => n + r.elapsedMs, 0),
    });
  }
writeFileSync(
  resolve(directory, 'teacher.json'),
  JSON.stringify({ passed: true, exploratory: true, rows, groups }, null, 2),
  { flag: 'wx' },
);
console.log(JSON.stringify(groups));
