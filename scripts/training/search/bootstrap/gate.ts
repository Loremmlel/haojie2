import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { bootstrapDecision, bootstrapValueDecision } from './policy';
import { openValueModel } from '../value-cycle/model';
import type { Observation } from '../../../../src/ai/types';
import { positions } from '../positions';
import { reference } from '../reference';
import { search, terminalValue } from '../puct';
import { immediateCertificate } from '../leaf/rollout';
import { readTrainingRecords } from '../../records/replay';
import { decisionOwner, fingerprint } from '../../../../src/ai/observation';
import { sampleTrainingTransition } from '../../../../src/ai/training/simulation';
import { encodingSourceHash } from '../../encode';
import { hashRecordFile } from '../../records/io';

// 不调整门槛：人工两步战术必须全通过；旧近终局所有支持位置必须两命令内兑现。
const output = process.argv[2];
assert.ok(output);
mkdirSync(output, { recursive: true });
const write = (file: string, value: unknown) =>
  writeFileSync(resolve(output, file), JSON.stringify(value, null, 2), { flag: 'wx' });
const digest = (file: string) => createHash('sha256').update(readFileSync(file)).digest('hex');
const input = 'artifacts/training/diagnostic-20260925/with-corrections-easy/games.jsonl.gz';
const scriptHashes = Object.fromEntries(
  [
    'scripts/training/search/bootstrap/policy.ts',
    'scripts/training/search/bootstrap/gate.ts',
    'scripts/training/search/puct.ts',
    'scripts/training/search/value-cycle/model.ts',
    'scripts/training/python-policy.ts',
    'training/haojie_training/inference.py',
    'training/haojie_training/model.py',
    'training/haojie_training/runtime.py',
    'training/haojie_training/data.py',
  ].map((f) => [f, digest(f)]),
);
const neural = process.argv[3] ? await openValueModel(process.argv[3]) : undefined;
const decideBootstrap = async (o: Observation, seed: number) =>
  neural ? bootstrapValueDecision(o, seed, neural.value) : bootstrapDecision(o, seed);
try {
  const protocol = {
    sourceSha256: encodingSourceHash(),
    input,
    inputSha256: await hashRecordFile(input),
    scriptHashes,
    seeds: [2026092731, 2026092732],
    simulations: 16,
    maxCandidates: 8,
    maxTeacherCalls: 17,
    teacherWorkPerCall: 40,
    actualTeacherWorkSafetyCap: 256,
    maxTransitions: 32,
    maxFollowCommands: 2,
    thresholds: { fixtureOptimal: 60, supportedNaturalWins: 10, unexpectedErrors: 0 },
    note: 'development tactical gate, not held-out win rate',
    checkpoint: neural?.model.ready,
  };
  write('protocol.json', protocol);
  const fixtures: any[] = [],
    natural: any[] = [];
  const old = JSON.parse(
    readFileSync('artifacts/training/search-probe-final-20260925/decisions.json', 'utf8'),
  );
  for (const p of positions()) {
    const baseline = search(p.observation, { simulations: 32, horizon: 2, sampleSeed: 2026092721 });
    const saved = old.find(
      (r: any) => r.name === p.name && r.seed === 2026092721 && r.budget === 32,
    );
    for (const [key, value] of Object.entries(baseline)) assert.deepEqual(saved[key], value);
    const oracle = reference(p.observation, 2);
    for (const seed of protocol.seeds) {
      const before = JSON.stringify(p.observation);
      const result = await decideBootstrap(p.observation, seed);
      const value = oracle.rootValues.find(
        (r) => JSON.stringify(r.command) === JSON.stringify(result.command),
      )?.value;
      assert.notEqual(value, undefined);
      fixtures.push({
        name: p.name,
        family: p.family,
        seed,
        result,
        value,
        optimal: value === oracle.best,
      });
      assert.equal(JSON.stringify(p.observation), before);
      assert.deepEqual(result, await decideBootstrap(p.observation, seed));
    }
  }
  write('fixtures.json', fixtures);
  let last: any,
    games = 0,
    outcomes = { terminal: 0, truncated: 0, interrupted: 0 };
  for await (const r of readTrainingRecords(input)) {
    if (r.type === 'game') {
      games++;
      last = undefined;
    }
    if (r.type === 'decision') last = r;
    if (r.type !== 'outcome') continue;
    if (r.truncated) outcomes.truncated++;
    if (r.interrupted) outcomes.interrupted++;
    if (!r.terminated) continue;
    outcomes.terminal++;
    assert.ok(last && last.index === r.commands - 1);
    for (const seed of protocol.seeds) {
      let o = last.observation;
      const root = decisionOwner(o),
        steps: any[] = [];
      for (let i = 0; i < 2 && o.winner === undefined; i++) {
        const result = await decideBootstrap(o, seed);
        const certificate = immediateCertificate(o, result.command);
        const next = sampleTrainingTransition(o, decisionOwner(o), result.command, 2026092741 + i);
        steps.push({ before: fingerprint(o), after: fingerprint(next), result, certificate });
        assert.deepEqual(result, await decideBootstrap(o, seed));
        o = next;
      }
      natural.push({ game: r.game, index: last.index, seed, steps, value: terminalValue(o, root) });
    }
  }
  write('natural.json', natural);
  for (const [file, hash] of Object.entries(scriptHashes)) assert.equal(digest(file), hash);
  assert.equal(encodingSourceHash(), protocol.sourceSha256);
  if (neural) assert.equal(digest(process.argv[3]), neural.model.ready.checkpoint_sha256);
  const summary = {
    complete: true,
    protocolSha256: digest(resolve(output, 'protocol.json')),
    fixtureOptimal: fixtures.filter((r) => r.optimal).length,
    fixtureTotal: fixtures.length,
    naturalWins: natural.filter((r) => r.value === 1).length,
    supportedNaturalWins: natural.filter(
      (r) => r.steps[0].result.mode === 'search' && r.value === 1,
    ).length,
    naturalTotal: natural.length,
    fallbackPaths: natural.filter((r) => r.steps[0].result.mode !== 'search').length,
    maxCommands: Math.max(...natural.map((r) => r.steps.length)),
    defaultRechecks: 30,
    games,
    outcomes,
    noTrainingLabels: true,
    networkCalls: neural?.model.totals.calls ?? 0,
  };
  write('summary.json', summary);
  console.log(JSON.stringify(summary));
  if (!process.argv.includes('--report-only')) {
    assert.equal(summary.fixtureOptimal, protocol.thresholds.fixtureOptimal);
    assert.equal(summary.supportedNaturalWins, protocol.thresholds.supportedNaturalWins);
  }
} finally {
  neural?.model.close();
}
