import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { readTrainingRecords } from '../records/replay';
import { encodingSourceHash } from '../encode';
import { decisionOwner, fingerprint, imagined, hash } from '../../../src/ai/observation';
import { evaluate } from '../../../src/ai/evaluation/evaluate';
import { TrainingTeacher } from '../../../src/ai/training/teacher';
import { sampleTrainingTransition } from '../../../src/ai/training/simulation';
import { checkPosition, terminalValue } from '../search/puct';
import type { Observation } from '../../../src/ai/types';
import { pairedRanking } from './rollout';

/** 从原轨迹重建公开根，再独立重放研究命令；原子概率展开的实际成本单独核算。 */
const { values } = parseArgs({
  options: {
    source: { type: 'string' },
    stage: { type: 'string', default: 'pilot' },
    output: { type: 'string' },
    'transitions-only': { type: 'boolean', default: false },
  },
});
assert.ok(values.source && values.output);
const read = (name: string) => JSON.parse(readFileSync(resolve(values.source!, name), 'utf8'));
const digest = (path: string) => createHash('sha256').update(readFileSync(path)).digest('hex');
const protocol = read('protocol.json');
assert.equal(digest(protocol.input), protocol.inputHash);
assert.equal(encodingSourceHash(), protocol.sourceHash);
const summary = read(`${values.stage}/summary.json`);
assert.equal(digest(resolve(values.source, 'protocol.json')), summary.protocolHash);
assert.equal(digest(resolve(values.source, values.stage!, 'results.json')), summary.resultsHash);
assert.equal(
  digest(resolve(values.source, values.stage!, 'candidates.json')),
  summary.candidateHash,
);
const results = read(`${values.stage}/results.json`);
const candidates = read(`${values.stage}/candidates.json`);
assert.equal(candidates.length, summary.roots);
assert.equal(
  results.length,
  candidates.reduce((n: number, r: any) => n + r.candidates.length * summary.scenarios, 0),
);
assert.equal(
  new Set(results.map((r: any) => `${r.index}:${r.candidate}:${r.scene}`)).size,
  results.length,
);
const roots = new Map<string, Observation>();
let replayedSource = 0,
  games = 0;
for await (const row of readTrainingRecords(protocol.input)) {
  if (row.type === 'game') games++;
  if (row.type !== 'decision') continue;
  replayedSource++;
  const p = protocol.positions.find((p: any) => p.game === row.game && p.index === row.index);
  if (p) {
    assert.equal(p.before, row.before);
    assert.equal(p.actor, row.actor);
    roots.set(p.before, row.observation);
  }
}
assert.equal(roots.size, 32);
const coverage = protocol.positions.map((p: any, index: number) => {
  try {
    checkPosition(roots.get(p.before)!);
    return { index, supported: true };
  } catch (error) {
    return { index, supported: false, reason: String(error) };
  }
});
let replayedSimulation = 0;
let teacherRechecks = 0;
const details = [];
for (const row of results) {
  let o = roots.get(protocol.positions[row.index].before)!;
  const root = decisionOwner(o);
  const initialPly = o.ply;
  const initialActive = o.active;
  assert.equal(row.sample, hash(`counterfactual-${values.stage}:${row.index}:${row.scene}`));
  const teacher = new TrainingTeacher('easy', 40);
  let actualWork = 0,
    maxDecisionWork = 0;
  for (const [i, step] of (row.steps ?? []).entries()) {
    assert.equal(step.before, fingerprint(o));
    assert.equal(step.actor, decisionOwner(o));
    assert.equal(step.sample, hash(`counterfactual-transition:${row.sample}:${i}`));
    if (i === 0) assert.deepEqual(step.command, candidates[row.index].candidates[row.candidate]);
    else if (!values['transitions-only']) {
      const d = teacher.next(o);
      teacherRechecks++;
      assert.deepEqual(d.command, step.command);
      assert.equal(d.stats.simulations, step.work);
      assert.equal(d.stats.cached ?? false, step.cached);
      actualWork += d.stats.simulations;
      maxDecisionWork = Math.max(maxDecisionWork, d.stats.simulations);
    } else {
      assert.ok(Number.isSafeInteger(step.work) && step.work >= 0);
      actualWork += step.work;
      maxDecisionWork = Math.max(maxDecisionWork, step.work);
    }
    o = sampleTrainingTransition(o, step.actor, step.command, step.sample);
    assert.equal(step.after, fingerprint(o));
    replayedSimulation++;
  }
  if (row.stop !== 'wall-limit') {
    assert.equal(fingerprint(o), row.final);
    assert.equal(terminalValue(o, root), row.value);
    assert.equal(row.complete, ['boundary', 'terminal'].includes(row.stop));
    if (row.complete) assert.equal(evaluate(imagined(o), root), row.heuristic);
    else assert.equal(row.heuristic, null);
    if (row.stop === 'boundary') {
      assert.equal(o.ply, initialPly + 2);
      assert.equal(o.active, initialActive);
      assert.equal(o.pending.length, 0);
      assert.ok(['summon', 'synthesis'].includes(o.phase));
    }
  }
  if (row.stop !== 'wall-limit') assert.equal(row.work, actualWork + (row.unexecutedWork ?? 0));
  let failedDecision;
  if (row.stop === 'teacher-paused' && row.reason?.includes('cost <= 40')) {
    const d = teacher.next(o);
    assert.ok(d.command && d.stats.simulations > 40);
    failedDecision = {
      phase: o.phase,
      pending: o.pending.length,
      command: d.command,
      stats: d.stats,
    };
  }
  details.push({
    index: row.index,
    candidate: row.candidate,
    scene: row.scene,
    stop: row.stop,
    actualWork,
    maxDecisionWork,
    failedDecision,
  });
}
const ranks = read(`${values.stage}/rankings.json`);
for (const r of candidates) {
  const matrix = r.candidates.map((_: unknown, i: number) =>
    results.filter((v: any) => v.index === r.index && v.candidate === i),
  );
  const ranking = pairedRanking(matrix);
  for (const [key, value] of Object.entries(ranking)) assert.deepEqual(ranks[r.index][key], value);
}
assert.equal(ranks.filter((r: any) => r.paired.length).length, summary.completeRoots);
const audit = {
  passed: true,
  games,
  replayedSource,
  replayedSimulation,
  teacherRechecks,
  costAudit: values['transitions-only']
    ? 'recorded work sum; no repeat planner execution'
    : 'repeat planner execution',
  coverage,
  details,
  inputHash: protocol.inputHash,
  protocolHash: summary.protocolHash,
  resultsHash: summary.resultsHash,
};
writeFileSync(values.output, JSON.stringify(audit, null, 2) + '\n', { flag: 'wx' });
console.log(
  JSON.stringify({
    passed: true,
    games,
    replayedSource,
    replayedSimulation,
    teacherRechecks,
    unsupported: coverage.filter((r: any) => !r.supported).length,
    failedDecisions: details.filter((d) => d.failedDecision).length,
    maxFailedCost: Math.max(0, ...details.map((d) => d.failedDecision?.stats.simulations ?? 0)),
  }),
);
