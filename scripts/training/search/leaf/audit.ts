import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { basename, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { search } from '../puct';
import { positions } from '../positions';
import { reference } from '../reference';
import { evaluate } from '../../../../src/ai/evaluation/evaluate';
import { decisionOwner, imagined } from '../../../../src/ai/observation';
import { inspectTrainingCommand } from '../../../../src/ai/training/queries';
import { readTrainingRecords } from '../../records/replay';
import { hashRecordFile } from '../../records/io';
import { encodingSourceHash } from '../../encode';

// 重新重放来源、核对协议/旧源码快照；重算所有廉价变体，原始昂贵基线核对配对完整根统计。
const { positionals } = parseArgs({ allowPositionals: true });
assert.equal(positionals.length, 2, '提供首轮与FPU补充实验目录');
const digest = (file: string) => createHash('sha256').update(readFileSync(file)).digest('hex');
const read = (dir: string, file: string) => JSON.parse(readFileSync(resolve(dir, file), 'utf8'));
const observations = new Map(positions().map((p) => [p.name, p.observation]));
const oracles = new Map(positions().map((p) => [p.name, reference(p.observation, 2)]));
const protocols = positionals.map((dir) => read(dir, 'protocol.json'));
assert.deepEqual(protocols[0].selected, protocols[1].selected);
const previous = read(protocols[0].source, 'protocol.json');
assert.equal(await hashRecordFile(previous.input), previous.inputSha256);
assert.equal(digest(previous.manifest), previous.manifestSha256);
let games = 0;
const outcomes: any[] = [];
for await (const r of readTrainingRecords(previous.input)) {
  if (r.type === 'game') games++;
  if (r.type === 'outcome') outcomes.push(r);
  if (
    r.type === 'decision' &&
    protocols[0].selected.some((p: any) => p.game === r.game && p.index === r.index)
  ) {
    assert.equal(
      protocols[0].selected.find((p: any) => p.game === r.game && p.index === r.index).before,
      r.before,
    );
    observations.set(`game-${r.game}-index-${r.index}`, r.observation);
  }
}
assert.equal(games, 10);
assert.equal(outcomes.length, 10);
assert.equal(observations.size, 34);
const checked = new Map<string, any>();
let rechecks = 0;
let finiteNumbers = 0;
const finite = (value: any) => {
  if (typeof value === 'number') {
    assert.ok(Number.isFinite(value));
    finiteNumbers++;
  } else if (value && typeof value === 'object') Object.values(value).forEach(finite);
};
for (const [experiment, dir] of positionals.entries()) {
  const protocol = protocols[experiment];
  const summary = read(dir, 'summary.json');
  const rows: any[] = read(dir, 'decisions.json');
  assert.equal(summary.complete, true);
  assert.equal(summary.protocolSha256, digest(resolve(dir, 'protocol.json')));
  assert.equal(protocol.sourceSha256, encodingSourceHash());
  assert.equal(protocol.previousProtocolSha256, digest(resolve(protocol.source, 'protocol.json')));
  assert.equal(protocol.selectionSha256, digest(resolve(protocol.source, 'selection.json')));
  for (const [file, hash] of Object.entries(protocol.scriptHashes)) {
    const path = experiment === 0 ? resolve(dir, 'source-snapshot', basename(file)) : file;
    assert.equal(digest(path), hash);
  }
  assert.equal(rows.length, 34 * 2 * protocol.variants.length);
  assert.equal(summary.rows, rows.length);
  const keys = new Set<string>();
  const beforeRechecks = rechecks;
  for (const row of rows) {
    finite(row);
    const key = `${row.name}:${row.seed}:${row.variant}`;
    assert.ok(!keys.has(key));
    keys.add(key);
    const o = observations.get(row.name)!;
    assert.ok(o);
    assert.equal(row.budget, 16);
    assert.ok(protocol.seeds.includes(row.seed) && protocol.variants.includes(row.variant));
    assert.equal(row.stats.networkCalls, 0);
    if (row.status === 'command') {
      assert.notEqual(inspectTrainingCommand(o, decisionOwner(o), row.command).status, 'invalid');
      assert.equal(row.stats.simulations, 16);
      assert.equal(
        row.edges.reduce((n: number, e: any) => n + e.visits, 0),
        16,
      );
      assert.equal(
        row.stats.terminalLeaves + row.stats.expansionLeaves + row.stats.cutoffLeaves,
        16,
      );
      if (row.kind === 'fixture') {
        const oracle = oracles.get(row.name)!;
        const expected = oracle.rootValues.find(
          (r) => JSON.stringify(r.command) === JSON.stringify(row.command),
        )!.value;
        assert.equal(row.referenceValue, expected);
        assert.equal(row.regret, oracle.best - expected);
      } else assert.equal(row.optimal, undefined);
    } else {
      assert.equal(row.status, 'paused');
      assert.ok(row.reason && !row.command);
    }
    if (row.variant === 'eager-zero') {
      const paired = rows.find(
        (r) => r.name === row.name && r.seed === row.seed && r.variant === 'deferred-zero',
      );
      for (const key of ['status', 'command', 'reason', 'edges', 'baseline'])
        assert.deepEqual(paired[key], row[key]);
    } else if (checked.has(key)) {
      const old = checked.get(key);
      for (const field of ['status', 'command', 'reason', 'edges', 'baseline', 'stats'])
        assert.deepEqual(row[field], old[field]);
    } else {
      const before = JSON.stringify(o);
      const result = search(o, {
        simulations: 16,
        horizon: 2,
        maxActionNodes: 512,
        sampleSeed: row.seed,
        deferExpansion: true,
        firstPlayValue: row.variant === 'deferred-parent' ? 'parent' : 'zero',
        leafValue:
          row.variant !== 'deferred-zero'
            ? (o, p) => Math.tanh(evaluate(imagined(o), p) / 1000)
            : undefined,
      });
      for (const [field, value] of Object.entries(result)) assert.deepEqual(row[field], value);
      assert.equal(JSON.stringify(o), before);
      rechecks++;
    }
    checked.set(key, row);
  }
  const report = {
    passed: true,
    protocolSha256: summary.protocolSha256,
    decisionsSha256: digest(resolve(dir, 'decisions.json')),
    auditScriptSha256: digest('scripts/training/search/leaf/audit.ts'),
    rows: rows.length,
    independentReruns: rechecks - beforeRechecks,
    cumulativeReruns: rechecks,
    finiteNumericFields: finiteNumbers,
    replayedGames: games,
    outcomes: {
      terminal: outcomes.filter((r) => r.terminated).length,
      truncated: outcomes.filter((r) => r.truncated).length,
      interrupted: outcomes.filter((r) => r.interrupted).length,
    },
    reusedManualRows: experiment === 1 ? 68 : 0,
    noTrainingLabels: true,
  };
  writeFileSync(resolve(dir, 'audit.json'), JSON.stringify(report, null, 2), { flag: 'wx' });
  console.log(JSON.stringify(report));
}
assert.equal(rechecks, 204);
