import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { positions } from '../positions';
import { reference } from '../reference';
import { immediateCertificate } from './rollout';
import { decisionOwner } from '../../../../src/ai/observation';
import { decide } from '../../../../src/ai/planning/search';
import { inspectTrainingCommand } from '../../../../src/ai/training/queries';
import { trainingDistribution } from '../../../../src/ai/training/simulation';
import { readTrainingRecords } from '../../records/replay';
import { hashRecordFile } from '../../records/io';
import { encodingSourceHash } from '../../encode';

// 独立重放与统计审计；比对完整复跑，另用概率树核验指定续招，不把教师估值当真值。
const { positionals } = parseArgs({ allowPositionals: true });
assert.equal(positionals.length, 2, '提供正式实验及完整复跑目录');
const [dir, rerun] = positionals;
const digest = (file: string) => createHash('sha256').update(readFileSync(file)).digest('hex');
const read = (dir: string, file: string) => JSON.parse(readFileSync(resolve(dir, file), 'utf8'));
const protocol = read(dir, 'protocol.json');
assert.deepEqual(protocol, read(rerun, 'protocol.json'));
assert.equal(protocol.budget, 16);
assert.equal(protocol.horizon, 2);
assert.equal(protocol.maxActionNodes, 512);
assert.deepEqual(protocol.seeds, [2026092731, 2026092732]);
assert.deepEqual(protocol.variants, ['deferred-zero', 'deferred-parent', 'terminal-rollout']);
const seen = new Set<number>();
const selected = read(protocol.source, 'selection.json').positions.filter((p: any) => {
  if (seen.has(p.seed) || seen.size === 4) return false;
  seen.add(p.seed);
  return true;
});
assert.deepEqual(protocol.selected, selected);
assert.equal(protocol.sourceSha256, encodingSourceHash());
assert.equal(protocol.previousProtocolSha256, digest(resolve(protocol.source, 'protocol.json')));
assert.equal(protocol.selectionSha256, digest(resolve(protocol.source, 'selection.json')));
for (const [file, hash] of Object.entries(protocol.scriptHashes)) assert.equal(digest(file), hash);
const previous = read(protocol.source, 'protocol.json');
assert.equal(await hashRecordFile(previous.input), previous.inputSha256);
assert.equal(digest(previous.manifest), previous.manifestSha256);
const observations = new Map(positions().map((p) => [p.name, p.observation]));
const oracles = new Map(positions().map((p) => [p.name, reference(p.observation, 2)]));
const outcomes: any[] = [],
  terminal: any[] = [];
let games = 0,
  last: any;
for await (const r of readTrainingRecords(previous.input)) {
  if (r.type === 'game') {
    games++;
    last = undefined;
  }
  if (r.type === 'decision') {
    last = r;
    const selected = protocol.selected.find((p: any) => p.game === r.game && p.index === r.index);
    if (selected) {
      assert.equal(selected.before, r.before);
      observations.set(`game-${r.game}-index-${r.index}`, r.observation);
    }
  }
  if (r.type === 'outcome') {
    outcomes.push(r);
    if (r.terminated) {
      assert.ok(last && last.game === r.game && last.index === r.commands - 1);
      const name = `terminal-game-${r.game}-index-${last.index}`;
      terminal.push({
        name,
        kind: 'terminal-natural',
        game: r.game,
        index: last.index,
        before: last.before,
        recordedCommand: last.command,
      });
      observations.set(name, last.observation);
    }
  }
}
assert.equal(games, 10);
assert.equal(outcomes.length, 10);
assert.equal(observations.size, 42);
assert.deepEqual(terminal, read(dir, 'terminal-selection.json'));
assert.deepEqual(terminal, read(rerun, 'terminal-selection.json'));
const rows: any[] = read(dir, 'decisions.json');
const repeats: any[] = read(rerun, 'decisions.json');
assert.equal(rows.length, 252);
assert.equal(repeats.length, 252);
for (const path of [dir, rerun]) {
  const summary = read(path, 'summary.json');
  assert.equal(summary.complete, true);
  assert.equal(summary.protocolSha256, digest(resolve(path, 'protocol.json')));
  assert.equal(summary.rows, 252);
  assert.equal(summary.maxActive, 4);
  const decisions = read(path, 'decisions.json');
  assert.equal(summary.groups.length, 9);
  for (const g of summary.groups) {
    const group = decisions.filter((r: any) => r.kind === g.kind && r.variant === g.variant);
    const ok = group.filter((r: any) => r.status === 'command');
    assert.equal(g.requested, group.length);
    assert.equal(g.commands, ok.length);
    assert.equal(g.optimal, g.kind === 'fixture' ? ok.filter((r: any) => r.optimal).length : null);
    assert.equal(
      g.regret,
      g.kind === 'fixture' ? ok.reduce((n: number, r: any) => n + r.regret, 0) : null,
    );
    for (const [field, stat] of [
      ['rolloutTransitions', 'transitions'],
      ['rolloutTerminals', 'terminal'],
      ['rolloutUnknown', 'unknown'],
      ['teacherWork', 'teacherWork'],
    ])
      assert.equal(
        g[field],
        group.reduce((n: number, r: any) => n + (r.rollout?.[stat] ?? 0), 0),
      );
    for (const [field, certificate] of [
      ['certifiedRecordedWins', 'immediateRecorded'],
      ['certifiedChosenWins', 'immediateChosen'],
    ])
      assert.equal(
        g[field],
        ok.filter(
          (r: any) => r[certificate]?.status === 'exact' && r[certificate].winProbability === 1,
        ).length,
      );
    assert.equal(g.reasons.length, group.length - ok.length);
  }
}
// 只剔除计时；其它协议、命令、概率证书、工作量与访问统计必须逐字段一致。
const untimed = (value: any): any =>
  Array.isArray(value)
    ? value.map(untimed)
    : value && typeof value === 'object'
      ? Object.fromEntries(
          Object.entries(value)
            .filter(([key]) => !key.endsWith('Ms'))
            .map(([key, item]) => [key, untimed(item)]),
        )
      : value;
assert.deepEqual(untimed(rows), untimed(repeats));
const oldRows = [
  ...read('artifacts/training/leaf-cost-20260925', 'decisions.json'),
  ...read('artifacts/training/leaf-fpu-20260925', 'decisions.json'),
];
let finiteNumbers = 0,
  oldComparisons = 0,
  legalCommands = 0;
const finite = (value: any) => {
  if (typeof value === 'number') {
    assert.ok(Number.isFinite(value));
    finiteNumbers++;
  } else if (value && typeof value === 'object') Object.values(value).forEach(finite);
};
const keys = new Set<string>(),
  continuationCertificates: any[] = [];
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
  assert.ok(row.stats.transitions + (row.rollout?.transitions ?? 0) <= 32);
  if (row.rollout) {
    const s = row.rollout;
    assert.equal(s.calls, row.profile.leafCalls);
    assert.equal(s.calls, s.terminal + s.unknown);
    assert.equal(s.terminal, s.wins + s.losses + s.draws);
    assert.equal(s.teacherCalls, s.transitions);
    assert.ok(s.teacherWork <= s.teacherCalls * 40);
    assert.ok(s.reactionCalls <= s.teacherCalls);
  }
  if (row.status === 'command') {
    assert.notEqual(inspectTrainingCommand(o, decisionOwner(o), row.command).status, 'invalid');
    legalCommands++;
    assert.equal(row.stats.simulations, 16);
    assert.equal(
      row.edges.reduce((n: number, e: any) => n + e.visits, 0),
      16,
    );
    assert.equal(row.stats.terminalLeaves + row.stats.expansionLeaves + row.stats.cutoffLeaves, 16);
    if (row.kind === 'fixture') {
      const oracle = oracles.get(row.name)!;
      const expected = oracle.rootValues.find(
        (r) => JSON.stringify(r.command) === JSON.stringify(row.command),
      )!.value;
      assert.equal(row.referenceValue, expected);
      assert.equal(row.regret, oracle.best - expected);
      assert.equal(row.optimal, Math.abs(oracle.best - expected) < 1e-9);
    } else assert.equal(row.optimal, undefined);
    if (row.kind === 'terminal-natural') {
      const recorded = terminal.find((r) => r.name === row.name).recordedCommand;
      assert.deepEqual(row.immediateRecorded, immediateCertificate(o, recorded));
      assert.deepEqual(row.immediateChosen, immediateCertificate(o, row.command));
      if (row.variant === 'terminal-rollout') {
        // 选招后再做诊断，不把历史命令或证书提供给搜索；只认证同操作者的两命令路线。
        const d = trainingDistribution(o, decisionOwner(o), row.command, 0);
        assert.ok(!d.sampled && d.outcomes.length);
        assert.ok(Math.abs(d.outcomes.reduce((n, r) => n + r.weight, 0) - 1) < 1e-9);
        let winProbability = 0;
        const continuations = d.outcomes.map((r) => {
          assert.equal(decisionOwner(r.observation), decisionOwner(o));
          const teacher = decide(r.observation, decisionOwner(o), 'easy', {
            simulations: 40,
            mode: 'work',
          });
          assert.ok(teacher.command);
          const certificate = immediateCertificate(r.observation, teacher.command);
          assert.equal(certificate.status, 'exact');
          if (certificate.status === 'exact')
            winProbability += r.weight * certificate.winProbability;
          return { probability: r.weight, command: teacher.command, certificate };
        });
        const recordedEdge = row.edges.find(
          (e: any) => JSON.stringify(e.command) === JSON.stringify(recorded),
        );
        assert.ok(recordedEdge);
        continuationCertificates.push({
          name: row.name,
          seed: row.seed,
          command: row.command,
          winProbability,
          continuations,
          rootCommands: row.edges.length,
          visitedRootCommands: row.edges.filter((e: any) => e.visits).length,
          recordedVisits: recordedEdge.visits,
        });
      }
    }
  } else {
    assert.equal(row.status, 'paused');
    assert.ok(row.reason && !row.command);
  }
  if (row.kind !== 'terminal-natural' && row.variant !== 'terminal-rollout') {
    const old = oldRows.find(
      (r: any) => r.name === row.name && r.seed === row.seed && r.variant === row.variant,
    );
    assert.ok(old);
    for (const field of ['status', 'command', 'reason', 'edges', 'baseline', 'stats'])
      assert.deepEqual(row[field], old[field]);
    oldComparisons++;
  }
}
assert.equal(oldComparisons, 136);
const report = {
  passed: true,
  protocolSha256: digest(resolve(dir, 'protocol.json')),
  decisionsSha256: digest(resolve(dir, 'decisions.json')),
  repeatDecisionsSha256: digest(resolve(rerun, 'decisions.json')),
  terminalSelectionSha256: digest(resolve(dir, 'terminal-selection.json')),
  auditScriptSha256: digest('scripts/training/search/leaf/rollout-audit.ts'),
  rows: rows.length,
  fullDeterministicReruns: repeats.length,
  oldComparisons,
  legalCommands,
  finiteNumericFields: finiteNumbers,
  fixtureReferences: oracles.size,
  replayedGames: games,
  outcomes: {
    terminal: outcomes.filter((r) => r.terminated).length,
    truncated: outcomes.filter((r) => r.truncated).length,
    interrupted: outcomes.filter((r) => r.interrupted).length,
  },
  continuationCertificates,
  noTrainingLabels: true,
};
writeFileSync(resolve(dir, 'audit.json'), JSON.stringify(report, null, 2), { flag: 'wx' });
console.log(JSON.stringify(report));
