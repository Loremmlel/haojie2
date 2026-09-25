import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { readTrainingRecords } from '../../records/replay';
import { hashRecordFile } from '../../records/io';
import { encodingSourceHash } from '../../encode';
import { searchExamples } from './encoding';
import { bootstrapDecision, bootstrapValueDecision } from './policy';
import { openValueModel } from '../value-cycle/model';
import { fingerprint, hash } from '../../../../src/ai/observation';

// 独立重放所有实局，并重新校验每个访问候选的完整解码与部分真实决策确定性。
const dir = process.argv[2];
assert.ok(dir);
const read = (file: string) => JSON.parse(readFileSync(resolve(dir, file), 'utf8'));
const digest = (file: string) => createHash('sha256').update(readFileSync(file)).digest('hex');
const protocol = read('protocol.json'),
  summary = read('summary.json');
assert.equal(summary.complete, true);
assert.equal(summary.protocolSha256, digest(resolve(dir, 'protocol.json')));
assert.equal(protocol.sourceSha256, encodingSourceHash());
for (const [file, sha] of Object.entries(protocol.scriptHashes)) assert.equal(digest(file), sha);
assert.deepEqual(
  summary.results.map((r: any) => r.seed),
  protocol.jobs.map((r: any) => r.seed),
);
if (protocol.format === 'haojie-bootstrap-readiness-v1')
  assert.deepEqual(
    protocol.jobs.map((r: any) => r.seed),
    [2026092801, 2026092801, 2026092802, 2026092802, 2026092803, 2026092804],
  );
else if (protocol.format === 'haojie-bootstrap-selfplay-batch-v1') {
  assert.equal(protocol.format, 'haojie-bootstrap-selfplay-batch-v1');
  assert.ok(protocol.jobs.every((r: any) => r.kind === 'selfplay'));
  assert.equal(new Set(protocol.jobs.map((r: any) => r.seed)).size, protocol.jobs.length);
} else {
  assert.equal(protocol.format, 'haojie-bootstrap-value-cycle-v1');
  assert.equal(digest(protocol.checkpoint.path), protocol.checkpoint.sha256);
  assert.equal(protocol.checkpoint.scale, 0.25);
}
const learned = protocol.checkpoint ? await openValueModel(protocol.checkpoint.path) : undefined;
try {
  let commands = 0,
    search = 0,
    fallback = 0,
    repeats = 0,
    softRoots = 0,
    encodedNodes = 0;
  let maximumTeacherWorkPerDecision = 0;
  const games: any[] = [];
  let networkCalls = 0;
  for (const result of summary.results) {
    assert.equal(await hashRecordFile(result.output), result.sha256);
    let header: any,
      outcome: any,
      count = 0,
      selected: any[] = [],
      lastSearch: any,
      gameCalls = 0;
    for await (const row of readTrainingRecords(result.output)) {
      if (row.type === 'game') {
        assert.equal(header, undefined);
        header = row;
        if (learned) assert.equal(header.valueModelSha256, learned.model.ready.checkpoint_sha256);
      }
      if (row.type === 'outcome') {
        assert.equal(outcome, undefined);
        outcome = row;
      }
      if (row.type !== 'sample') continue;
      commands++;
      count++;
      const s = row.searchStats;
      assert.ok(
        Object.values(s).every((n) => typeof n === 'number' && Number.isFinite(n) && n >= 0),
      );
      assert.ok(s.teacherCalls <= 17 && s.teacherWork <= s.teacherCalls * 256);
      maximumTeacherWorkPerDecision = Math.max(maximumTeacherWorkPerDecision, s.teacherWork);
      assert.ok(s.searchTransitions + s.rolloutTransitions <= 32);
      if (row.valueStats) {
        assert.ok(learned);
        assert.ok(
          Number.isInteger(row.valueStats.calls) &&
            row.valueStats.calls >= 0 &&
            row.valueStats.calls <= 16,
        );
        assert.ok(
          [row.valueStats.min, row.valueStats.max].every(
            (v) => Number.isFinite(v) && Math.abs(v) <= 0.25,
          ),
        );
        networkCalls += row.valueStats.calls;
        gameCalls += row.valueStats.calls;
      }
      assert.equal(
        row.searchSeed,
        hash(`bootstrap-experiment:2026092851:${fingerprint(row.observation)}`),
      );
      if (row.policyMode === 'search') {
        assert.equal(s.searchSimulations, 16);
        const examples = searchExamples(row.observation, row.actor, row.command, row.searchPolicy);
        assert.equal(examples.filter((e) => e.step === 0).length, 1);
        for (const e of examples) {
          assert.ok(e.policy.every((v) => Number.isFinite(v) && v >= 0));
          assert.ok(Math.abs(e.policy.reduce((a, b) => a + b, 0) - 1) < 1e-9);
          assert.ok(!('seed' in e.input) && !('rng' in e.input));
        }
        if (examples[0].policy.filter((v) => v > 0).length > 1) softRoots++;
        encodedNodes += examples.length;
        search++;
        if (selected.length < 2) selected.push(row);
        lastSearch = row;
      } else {
        assert.equal(row.searchPolicy, null);
        fallback++;
      }
    }
    assert.ok(header && outcome && !outcome.interrupted);
    assert.equal(count, result.commands);
    if (learned) assert.equal(gameCalls, result.inference.calls);
    for (const key of ['terminated', 'truncated', 'winner', 'returns'])
      assert.deepEqual(outcome[key], result[key]);
    if (lastSearch && !selected.some((r) => r.index === lastSearch.index))
      selected.push(lastSearch);
    for (const row of selected) {
      const d = learned
        ? await bootstrapValueDecision(row.observation, row.searchSeed, learned.value)
        : bootstrapDecision(row.observation, row.searchSeed);
      assert.deepEqual(d.command, row.command);
      assert.deepEqual(d.policy, row.searchPolicy);
      assert.deepEqual(d.stats, row.searchStats);
      if ('valueStats' in d) assert.deepEqual(d.valueStats, row.valueStats);
      repeats++;
    }
    games.push({
      game: header.game,
      kind: header.experimentKind,
      seed: header.seed,
      commands: count,
      terminated: outcome.terminated,
      truncated: outcome.truncated,
      interrupted: !!outcome.interrupted,
      winner: outcome.winner,
    });
  }
  assert.equal(commands, summary.replayed);
  assert.equal(search, summary.policyTargets);
  assert.equal(fallback, summary.fallbackTargets);
  assert.ok(softRoots > 0);
  const report = {
    passed: true,
    protocolSha256: summary.protocolSha256,
    summarySha256: digest(resolve(dir, 'summary.json')),
    auditScriptSha256: digest('scripts/training/search/bootstrap/audit.ts'),
    commands,
    search,
    fallback,
    repeats,
    softRoots,
    encodedNodes,
    maximumTeacherWorkPerDecision,
    networkCalls,
    games,
    note: 'fallback count includes opponent decisions; full replay and candidate legality, not strength proof',
  };
  writeFileSync(resolve(dir, 'audit.json'), JSON.stringify(report, null, 2), { flag: 'wx' });
  console.log(JSON.stringify(report));
} finally {
  learned?.model.close();
}
