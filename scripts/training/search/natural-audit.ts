import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { encodingSourceHash } from '../encode';
import { hashRecordFile } from '../records/io';
import { readTrainingRecords } from '../records/replay';
import { parallelRechecks } from './rechecks';
import { inspectTrainingCommand } from '../../../src/ai/training/queries';
import { TrainingActionTree } from '../../../src/ai/training/action-tree';

// 独立重放选样与全部来源；核对参照、合法性、预算，并复跑每个位置的一条最大预算搜索。
// 失败位置同样参与审计，不换题、不把未知估计写成训练价值。
const { values } = parseArgs({ options: { input: { type: 'string' } } });
assert.ok(values.input);
const directory = values.input;
const read = (name: string) => JSON.parse(readFileSync(resolve(directory, name), 'utf8'));
const digest = (data: Buffer | string) => createHash('sha256').update(data).digest('hex');
const protocol = read('protocol.json');
const selection = read('selection.json');
const rows: any[] = read('decisions.json');
const references: any[] = read('references.json');
const summary = read('summary.json');
assert.equal(summary.complete, true);
assert.equal(encodingSourceHash(), protocol.sourceSha256);
assert.equal(await hashRecordFile(protocol.input), protocol.inputSha256);
assert.equal(digest(readFileSync(protocol.manifest)), protocol.manifestSha256);
for (const [path, hash] of Object.entries(protocol.scriptHashes))
  assert.equal(digest(readFileSync(path)), hash);
const manifest = JSON.parse(readFileSync(protocol.manifest, 'utf8'));
const trainingGroups = new Set([
  ...manifest.splits.train.groups,
  ...manifest.splits.validation.groups,
]);
const candidates = new Map<number, any[]>();
const outcomes: any[] = [];
let windows = new Set<string>();
for await (const row of readTrainingRecords(protocol.input)) {
  if (row.type === 'game') {
    const group = `${row.ruleset}:${row.rules}:${row.seed}`;
    assert.ok(!trainingGroups.has(group));
    assert.equal(selection.headers.find((h: any) => h.game === row.game).group, group);
    assert.ok(!candidates.has(row.game));
    candidates.set(row.game, []);
    windows = new Set();
  } else if (row.type === 'outcome') {
    outcomes.push(row);
  } else if (row.type === 'decision') {
    const o = row.observation;
    if ((o.phase !== 'play' && !o.pending.length) || o.units.length < 2) continue;
    const window = `${o.ply}:${row.actor}:${o.pending.length > 0}`;
    if (windows.has(window)) continue;
    windows.add(window);
    candidates.get(row.game)!.push({
      ...row,
      rank: digest(`${protocol.selectionSalt}:${row.game}:${row.index}:${row.before}`),
    });
  }
}
assert.equal(candidates.size, 10);
assert.equal(outcomes.length, 10);
const selected: any[] = [];
for (const [game, positions] of candidates) {
  assert.equal(positions.length, selection.eligible[game]);
  positions.sort((a, b) => a.rank.localeCompare(b.rank));
  for (const p of positions.slice(0, protocol.perGame)) {
    const stored = selection.positions.find((s: any) => s.game === game && s.index === p.index);
    assert.ok(stored);
    for (const key of ['rank', 'before', 'actor', 'policy']) assert.equal(stored[key], p[key]);
    selected.push(p);
  }
}
assert.equal(selected.length, 20);
assert.equal(selection.positions.length, 20);
assert.equal(rows.length, 180);
assert.equal(references.length, 20);
let legalCommands = 0;
let deterministicRechecks = 0;
const parallel = await parallelRechecks(
  selected.map((p) => p.observation),
  protocol,
);
for (const [index, p] of selected.entries()) {
  const old = references.find((r) => r.game === p.game && r.index === p.index);
  const { oracle, failure, repeated } = parallel.results[index];
  assert.equal(old.status, oracle ? 'exact' : 'unresolved');
  if (oracle) {
    assert.deepEqual(old.rootValues, oracle.rootValues);
    assert.deepEqual(old.stats, oracle.stats);
  } else assert.equal(old.reason, failure);
  const group = rows.filter((r) => r.game === p.game && r.index === p.index);
  assert.equal(group.length, 9);
  assert.equal(group.filter((r) => r.method === 'hard-800').length, 1);
  for (const seed of protocol.searchSeeds)
    assert.deepEqual(
      group.filter((r) => r.method === 'uniform-puct' && r.seed === seed).map((r) => r.budget),
      protocol.budgets,
    );
  for (const row of group) {
    assert.equal(row.before, p.before);
    assert.equal(row.actor, p.actor);
    assert.ok(Number.isFinite(row.elapsedMs) && row.elapsedMs >= 0);
    if (row.command) {
      assert.notEqual(
        inspectTrainingCommand(p.observation, p.actor, row.command).status,
        'invalid',
      );
      legalCommands++;
      if (oracle) {
        const step = new TrainingActionTree(p.observation, p.actor).trace(row.command).at(-1)!;
        const command = step.node.choices[step.selected].command;
        const choice = oracle.rootValues.find(
          (r) => JSON.stringify(r.command) === JSON.stringify(command),
        );
        assert.ok(choice);
        assert.equal(row.value, choice.value);
        assert.equal(row.regret, oracle.best - choice.value);
        assert.equal(row.optimal, Math.abs(row.regret) < 1e-9);
      }
    }
    assert.equal(row.reference, oracle ? 'exact' : 'unresolved');
    if (row.method !== 'uniform-puct') continue;
    assert.equal(row.stats.networkCalls, 0);
    assert.ok(row.stats.simulations <= row.budget);
    if (row.status === 'command') {
      assert.equal(row.stats.simulations, row.budget);
      assert.equal(
        row.edges.reduce((n: number, e: any) => n + e.visits, 0),
        row.budget,
      );
      assert.equal(
        row.stats.terminalLeaves + row.stats.cutoffLeaves + row.stats.expansionLeaves,
        row.budget,
      );
      for (const edge of row.edges) {
        assert.ok(edge.visits >= 0 && Number.isSafeInteger(edge.visits));
        assert.ok(
          edge.value === null || (Number.isFinite(edge.value) && Math.abs(edge.value) <= 1),
        );
      }
    } else {
      assert.equal(row.status, 'paused');
      assert.ok(row.reason && !row.command);
    }
    if (row.seed === protocol.searchSeeds[0] && row.budget === 128) {
      for (const [key, value] of Object.entries(repeated)) assert.deepEqual(row[key], value);
      deterministicRechecks++;
    }
  }
}
const quantile = (items: number[], q: number) =>
  [...items].sort((a, b) => a - b)[Math.ceil(q * items.length) - 1];
const methods = ['hard-800', ...protocol.budgets.map((b: number) => `puct-${b}`)].map((method) => {
  const group = rows.filter((r) =>
    method === 'hard-800'
      ? r.method === method
      : r.method === 'uniform-puct' && method === `puct-${r.budget}`,
  );
  return {
    method,
    requested: group.length,
    commands: group.filter((r) => r.command).length,
    terminalLeaves: group.reduce((n, r) => n + (r.stats.terminalLeaves ?? 0), 0),
    medianMs: quantile(
      group.map((r) => r.elapsedMs),
      0.5,
    ),
    p95Ms: quantile(
      group.map((r) => r.elapsedMs),
      0.95,
    ),
    pausedReasons: group.reduce((acc, r) => {
      if (r.reason) acc[r.reason] = (acc[r.reason] ?? 0) + 1;
      return acc;
    }, {}),
  };
});
const report = {
  passed: true,
  protocolSha256: digest(readFileSync(resolve(directory, 'protocol.json'))),
  auditScriptSha256: digest(readFileSync('scripts/training/search/natural-audit.ts')),
  recheckScriptSha256: digest(readFileSync('scripts/training/search/rechecks.ts')),
  parallel: {
    workers: parallel.workers,
    maxActive: parallel.maxActive,
    elapsedMs: parallel.elapsedMs,
  },
  sourceOutcomes: {
    terminal: outcomes.filter((r) => r.terminated).length,
    truncated: outcomes.filter((r) => r.truncated).length,
    interrupted: outcomes.filter((r) => r.interrupted).length,
  },
  replayedGames: candidates.size,
  selectedPositions: selected.length,
  referenceRechecks: selected.length,
  exactPositions: references.filter((r) => r.status === 'exact').length,
  informativePositions: references.filter((r) => r.informative).length,
  legalCommands,
  deterministicRechecks,
  methods,
};
writeFileSync(resolve(directory, 'audit.json'), JSON.stringify(report, null, 2), { flag: 'wx' });
console.log(JSON.stringify(report));
