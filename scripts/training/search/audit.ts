import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { positions } from './positions';
import { reference } from './reference';
import { search } from './puct';
import { encodingSourceHash } from '../encode';
import { inspectTrainingCommand } from '../../../src/ai/training/queries';

const { values } = parseArgs({ options: { input: { type: 'string' } } });
assert.ok(values.input, '指定--input目录');
const directory = values.input;
const read = (name: string) => JSON.parse(readFileSync(resolve(directory, name), 'utf8'));
const digest = (path: string) => createHash('sha256').update(readFileSync(path)).digest('hex');
const protocol = read('protocol.json');
const summary = read('summary.json');
const rows: any[] = read('decisions.json');
const references: any[] = read('reference.json');
const cases = positions();
assert.equal(summary.passed, true);
assert.equal(summary.protocolSha256, digest(resolve(directory, 'protocol.json')));
assert.equal(encodingSourceHash(), protocol.sourceSha256);
for (const [file, expected] of Object.entries(protocol.scriptHashes))
  assert.equal(digest(file), expected);
assert.deepEqual(protocol.budgets, [0, 32, 64, 128]);
assert.equal(protocol.horizon, 2);
assert.equal(cases.length, 30);
assert.equal(rows.length, 480);
assert.equal(references.length, 30);
const keys = new Set<string>();
let deterministicRechecks = 0;
for (const p of cases) {
  const savedCase = protocol.cases.find((r: any) => r.name === p.name);
  assert.equal(
    savedCase.observationSha256,
    createHash('sha256').update(JSON.stringify(p.observation)).digest('hex'),
  );
  const oracle = reference(p.observation, protocol.horizon);
  const oldReference = references.find((r) => r.name === p.name);
  assert.deepEqual(oracle.rootValues, oldReference.rootValues);
  assert.deepEqual(oracle.stats, oldReference.stats);
  assert.equal(oracle.best, oldReference.best);
  for (const seed of protocol.seeds) {
    const paired = rows.filter((r) => r.name === p.name && r.seed === seed);
    assert.deepEqual(
      paired.map((r) => r.budget),
      protocol.budgets,
    );
    for (const row of paired) {
      const key = `${row.name}:${seed}:${row.budget}`;
      assert.ok(!keys.has(key));
      keys.add(key);
      assert.equal(row.actor, p.actor);
      assert.equal(row.family, p.family);
      assert.equal(row.status, 'command');
      assert.notEqual(
        inspectTrainingCommand(p.observation, p.actor, row.command).status,
        'invalid',
      );
      assert.deepEqual(row.baseline, paired[0].command);
      assert.deepEqual(
        row.edges.map((e: any) => e.command),
        paired[0].edges.map((e: any) => e.command),
      );
      assert.equal(row.edges.length, oracle.rootValues.length);
      assert.equal(row.stats.simulations, row.budget);
      assert.equal(
        row.edges.reduce((n: number, e: any) => n + e.visits, 0),
        row.budget,
      );
      assert.equal(
        row.stats.terminalLeaves + row.stats.cutoffLeaves + row.stats.expansionLeaves,
        row.budget,
      );
      assert.equal(row.stats.sameActorEdges + row.stats.changedActorEdges, row.stats.transitions);
      assert.ok(row.stats.transitions >= row.budget && row.stats.transitions <= 2 * row.budget);
      assert.equal(row.stats.networkCalls, 0);
      for (const edge of row.edges) {
        assert.ok(Number.isSafeInteger(edge.visits) && edge.visits >= 0);
        assert.ok(edge.chanceOutcomesSeen <= edge.visits);
        assert.ok(
          edge.visits === 0
            ? edge.value === null
            : Number.isFinite(edge.value) && Math.abs(edge.value) <= 1,
        );
      }
      const choice = oracle.rootValues.find(
        (r) => JSON.stringify(r.command) === JSON.stringify(row.command),
      );
      assert.ok(choice);
      assert.equal(row.referenceValue, choice.value);
      assert.equal(row.referenceBest, oracle.best);
      assert.equal(row.regret, oracle.best - choice.value);
      assert.equal(row.optimal, row.regret < 1e-9);
      assert.ok(Number.isFinite(row.elapsedMs) && row.elapsedMs >= 0);
      // 每个夹具重放一个最大预算样本，核对整棵根统计，不只核对所选命令。
      if (seed === protocol.seeds[0] && row.budget === 128) {
        const repeated = search(p.observation, { simulations: 128, horizon: 2, sampleSeed: seed });
        assert.equal(repeated.status, 'command');
        if (repeated.status === 'command') {
          assert.deepEqual(repeated.command, row.command);
          assert.deepEqual(repeated.edges, row.edges);
          assert.deepEqual(repeated.stats, row.stats);
        }
        deterministicRechecks++;
      }
    }
  }
}
for (const group of summary.groups) {
  const selected = rows.filter(
    (r) => r.budget === group.budget && (group.family === 'all' || r.family === group.family),
  );
  assert.equal(group.decisions, selected.length);
  assert.equal(group.optimal, selected.filter((r) => r.optimal).length);
  assert.equal(group.meanRegret, selected.reduce((n, r) => n + r.regret, 0) / selected.length);
  for (const field of ['transitions', 'actionNodes', 'networkCalls'])
    assert.equal(
      group[field],
      selected.reduce((n, r) => n + r.stats[field], 0),
    );
  assert.equal(
    group.elapsedMs,
    selected.reduce((n, r) => n + r.elapsedMs, 0),
  );
}
const details = [];
for (const budget of protocol.budgets) {
  const selected = rows.filter((r) => r.budget === budget);
  const times = selected.map((r) => r.elapsedMs).sort((a, b) => a - b);
  details.push({
    budget,
    p50Ms: times[Math.floor((times.length - 1) * 0.5)],
    p95Ms: times[Math.floor((times.length - 1) * 0.95)],
    byActor: [1, 2].map((actor) => ({
      actor,
      decisions: selected.filter((r) => r.actor === actor).length,
      optimal: selected.filter((r) => r.actor === actor && r.optimal).length,
    })),
    bySeed: protocol.seeds.map((seed: number) => ({
      seed,
      optimal: selected.filter((r) => r.seed === seed && r.optimal).length,
    })),
    improved: selected.filter(
      (r) =>
        r.optimal &&
        !rows.find((b) => b.name === r.name && b.seed === r.seed && b.budget === 0).optimal,
    ).length,
    regressed: selected.filter(
      (r) =>
        !r.optimal &&
        rows.find((b) => b.name === r.name && b.seed === r.seed && b.budget === 0).optimal,
    ).length,
  });
}
const report = {
  passed: true,
  decisions: rows.length,
  exactReferenceRechecks: 30,
  deterministicRechecks,
  sourceAndScriptHashesMatch: true,
  publicCommandsValid: true,
  completeBudgetAccounting: true,
  noNetworkCalls: true,
  noTrainingLabels: true,
  details,
  failed32: rows
    .filter((r) => r.budget === 32 && !r.optimal)
    .map(({ name, seed, command, referenceValue, referenceBest }) => ({
      name,
      seed,
      command,
      referenceValue,
      referenceBest,
    })),
  auditScriptSha256: digest('scripts/training/search/audit.ts'),
};
writeFileSync(resolve(directory, 'audit.json'), JSON.stringify(report, null, 2), { flag: 'wx' });
console.log(JSON.stringify(report));
