import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { readTrainingRecords } from '../../records/replay';
import { hashRecordFile } from '../../records/io';
import { encodingSourceHash } from '../../encode';
import { decisionOwner, fingerprint } from '../../../../src/ai/observation';
import { decide } from '../../../../src/ai/planning/search';
import {
  sampleTrainingTransition,
  simulationRandomSource,
} from '../../../../src/ai/training/simulation';
import { inspectTrainingCommand } from '../../../../src/ai/training/queries';
import { terminalValue } from '../puct';
import { immediateCertificate } from './rollout';
import type { Observation } from '../../../../src/ai/types';

// 独立重放每个执行步骤，并离线证明该步是否已有立即获胜命令；不更改实验选招或训练标签。
const { positionals } = parseArgs({ allowPositionals: true });
assert.equal(positionals.length, 2, '提供连续重决策实验及完整复跑目录');
const [dir, repeat] = positionals;
const read = (dir: string, file: string) => JSON.parse(readFileSync(resolve(dir, file), 'utf8'));
const digest = (file: string) => createHash('sha256').update(readFileSync(file)).digest('hex');
const protocol = read(dir, 'protocol.json');
const rows: any[] = read(dir, 'results.json');
assert.deepEqual(protocol, read(repeat, 'protocol.json'));
assert.equal(rows.length, 32);
assert.equal(protocol.maxCommands, 8);
assert.deepEqual(protocol.pairs, [
  { searchSeed: 2026092731, executionSeed: 2026092741 },
  { searchSeed: 2026092732, executionSeed: 2026092742 },
]);
assert.equal(protocol.sourceSha256, encodingSourceHash());
assert.equal(await hashRecordFile(protocol.input), protocol.inputSha256);
assert.equal(digest(protocol.manifest), protocol.manifestSha256);
for (const [file, hash] of Object.entries(protocol.scriptHashes)) assert.equal(digest(file), hash);
assert.equal(protocol.previousProtocolSha256, digest(resolve(protocol.source, 'protocol.json')));
for (const path of [dir, repeat]) {
  const summary = read(path, 'summary.json');
  assert.equal(summary.complete, true);
  assert.equal(summary.protocolSha256, digest(resolve(path, 'protocol.json')));
  assert.equal(summary.resultsSha256, digest(resolve(path, 'results.json')));
  assert.equal(summary.paths, 32);
  assert.deepEqual(summary.sourceOutcomes, { terminal: 8, truncated: 2, interrupted: 0 });
}
assert.equal(read(repeat, 'summary.json').repeatedPaths, 32);
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
assert.deepEqual(untimed(rows), untimed(read(repeat, 'results.json')));
const observations = new Map<string, Observation>();
let games = 0;
for await (const r of readTrainingRecords(protocol.input)) {
  if (r.type === 'game') games++;
  if (r.type === 'decision') {
    const selected = protocol.selected.find((s: any) => s.game === r.game && s.index === r.index);
    if (selected) {
      assert.equal(selected.before, r.before);
      assert.deepEqual(selected.recordedCommand, r.command);
      observations.set(selected.name, r.observation);
    }
  }
}
assert.equal(games, 10);
assert.equal(observations.size, 8);
let finiteNumbers = 0;
const finite = (value: any) => {
  if (typeof value === 'number') {
    assert.ok(Number.isFinite(value));
    finiteNumbers++;
  } else if (value && typeof value === 'object') Object.values(value).forEach(finite);
};
const keys = new Set<string>(),
  evidence: any[] = [];
let commands = 0;
for (const row of rows) {
  finite(row);
  const key = `${row.name}:${row.searchSeed}:${row.variant}`;
  assert.ok(!keys.has(key));
  keys.add(key);
  assert.ok(protocol.variants.includes(row.variant));
  assert.ok(
    protocol.pairs.some(
      (p: any) => p.searchSeed === row.searchSeed && p.executionSeed === row.executionSeed,
    ),
  );
  assert.ok(row.commands <= 8);
  let o = observations.get(row.name)!;
  assert.ok(o);
  assert.equal(row.rootActor, decisionOwner(o));
  const random = simulationRandomSource(row.executionSeed);
  for (const [index, step] of row.steps.entries()) {
    assert.equal(step.index, index);
    assert.equal(step.before, fingerprint(o));
    assert.equal(step.actor, decisionOwner(o));
    if (!step.after) {
      assert.equal(step.result.status, 'paused');
      continue;
    }
    assert.equal(step.result.status, 'command');
    assert.equal(step.executionSample, Math.floor(random() * 4294967296));
    assert.notEqual(inspectTrainingCommand(o, step.actor, step.result.command).status, 'invalid');
    if (row.variant === 'terminal-rollout') {
      const stats = step.result.stats,
        s = step.rollout;
      assert.equal(stats.simulations, 16);
      assert.equal(stats.networkCalls, 0);
      assert.equal(stats.terminalLeaves + stats.expansionLeaves + stats.cutoffLeaves, 16);
      assert.equal(
        step.result.edges.reduce((n: number, e: any) => n + e.visits, 0),
        16,
      );
      assert.ok(stats.transitions + s.transitions <= 32);
      assert.equal(s.calls, s.terminal + s.unknown);
      assert.equal(s.terminal, s.wins + s.losses + s.draws);
      assert.equal(s.teacherCalls, s.transitions);
      assert.ok(s.teacherWork <= s.teacherCalls * 40);
      const teacher = decide(o, step.actor, 'easy', { simulations: 40, mode: 'work' });
      assert.ok(teacher.command);
      const teacherCertificate = immediateCertificate(o, teacher.command);
      const chosenCertificate = immediateCertificate(o, step.result.command);
      const edge = step.result.edges.find(
        (e: any) => JSON.stringify(e.command) === JSON.stringify(teacher.command),
      );
      assert.ok(edge);
      evidence.push({
        name: row.name,
        searchSeed: row.searchSeed,
        index,
        actor: step.actor,
        rootCommands: step.result.edges.length,
        visitedRootCommands: step.result.edges.filter((e: any) => e.visits).length,
        teacherCommand: teacher.command,
        teacherVisits: edge.visits,
        teacherCertificate,
        chosenCommand: step.result.command,
        chosenCertificate,
      });
    } else assert.ok(step.result.stats.simulations <= 40);
    o = sampleTrainingTransition(o, step.actor, step.result.command, step.executionSample);
    assert.equal(fingerprint(o), step.after);
    commands++;
  }
  assert.equal(row.final, fingerprint(o));
  assert.equal(row.value, terminalValue(o, row.rootActor));
  assert.equal(row.commands, row.steps.filter((s: any) => s.after).length);
  if (row.stop === 'terminal') assert.notEqual(o.winner, undefined);
  else assert.equal(o.winner, undefined);
  if (row.stop === 'command-limit') {
    assert.equal(row.commands, 8);
    const teacher = decide(o, decisionOwner(o), 'easy', { simulations: 40, mode: 'work' });
    assert.ok(teacher.command);
    evidence.push({
      name: row.name,
      searchSeed: row.searchSeed,
      index: 8,
      stopped: true,
      teacherCommand: teacher.command,
      teacherCertificate: immediateCertificate(o, teacher.command),
    });
  }
}
const diagnosticSteps = evidence.filter((r) => !r.stopped);
const certain = (c: any) => c.status === 'exact' && c.winProbability === 1;
const report = {
  passed: true,
  protocolSha256: digest(resolve(dir, 'protocol.json')),
  resultsSha256: digest(resolve(dir, 'results.json')),
  repeatResultsSha256: digest(resolve(repeat, 'results.json')),
  scriptSha256: digest('scripts/training/search/leaf/follow-audit.ts'),
  replayedGames: games,
  paths: rows.length,
  commands,
  repeatedPaths: 32,
  finiteNumbers,
  diagnosticSteps: diagnosticSteps.length,
  availableCertainWins: diagnosticSteps.filter((r) => certain(r.teacherCertificate)).length,
  chosenCertainWins: diagnosticSteps.filter((r) => certain(r.chosenCertificate)).length,
  unvisitedTeacherWins: diagnosticSteps.filter(
    (r) => certain(r.teacherCertificate) && r.teacherVisits === 0,
  ).length,
  singleRootVisits: diagnosticSteps.filter((r) => r.visitedRootCommands === 1).length,
  evidence,
  noTrainingLabels: true,
};
writeFileSync(resolve(dir, 'audit.json'), JSON.stringify(report, null, 2), { flag: 'wx' });
const { evidence: details, ...summary } = report;
console.log(JSON.stringify(summary));
