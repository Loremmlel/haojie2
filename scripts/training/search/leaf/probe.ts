import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { search, type ProbeOptions, type ProbeProfile } from '../puct';
import { reference } from '../reference';
import { positions } from '../positions';
import { encodingSourceHash } from '../../encode';
import { hashRecordFile } from '../../records/io';
import { readTrainingRecords } from '../../records/replay';
import { evaluate } from '../../../../src/ai/evaluation/evaluate';
import { imagined, decisionOwner } from '../../../../src/ai/observation';
import { inspectTrainingCommand } from '../../../../src/ai/training/queries';
import type { Observation } from '../../../../src/ai/types';
import type { Command } from '../../../../src/engine/types';
import { terminalRollout, immediateCertificate } from './rollout';

const fpu = process.argv.includes('--fpu');
const rollout = process.argv.includes('--rollout');
assert.ok(!(fpu && rollout));
const variants = rollout
  ? ['deferred-zero', 'deferred-parent', 'terminal-rollout']
  : fpu
    ? ['deferred-heuristic', 'deferred-parent']
    : ['eager-zero', 'deferred-zero', 'deferred-heuristic'];
const seeds = [2026092731, 2026092732];
const budget = 16;
const digest = (data: Buffer | string) => createHash('sha256').update(data).digest('hex');
interface Job {
  name: string;
  kind: 'natural' | 'fixture' | 'terminal-natural';
  family?: string;
  observation: Observation;
  game?: number;
  index?: number;
  before?: string;
  recordedCommand?: Command;
}

/** 同一局面独立比较成本与信号；手工评分只作估计，人工夹具仍用规则参照判定。 */
function probe(job: Job) {
  const before = JSON.stringify(job.observation);
  const oracle = job.kind === 'fixture' ? reference(job.observation, 2) : undefined;
  const rows: any[] = [];
  for (const [seedIndex, seed] of seeds.entries()) {
    // 两个固定种子反转方法顺序，减少固定热身顺序对计时比较的偏置。
    const order = seedIndex === 0 ? variants : [...variants].reverse();
    for (const variant of order) {
      const profile: ProbeProfile = { enumerationMs: 0, transitionMs: 0, leafMs: 0, leafCalls: 0 };
      const continuation = variant === 'terminal-rollout' ? terminalRollout(seed) : undefined;
      const options: ProbeOptions = {
        simulations: budget,
        horizon: 2,
        maxActionNodes: 512,
        sampleSeed: seed,
        profile,
        deferExpansion: variant !== 'eager-zero',
        firstPlayValue: variant === 'deferred-parent' ? 'parent' : 'zero',
        leafValue: continuation
          ? continuation.leafValue
          : variant === 'deferred-heuristic' || variant === 'deferred-parent'
            ? (o, actor) => Math.tanh(evaluate(imagined(o), actor) / 1000)
            : undefined,
      };
      const start = performance.now();
      const result = search(job.observation, options);
      const elapsedMs = performance.now() - start;
      if (continuation)
        assert.ok(result.stats.transitions + continuation.stats.transitions <= budget * 2);
      let scoring = {};
      if (result.status === 'command') {
        assert.notEqual(
          inspectTrainingCommand(job.observation, decisionOwner(job.observation), result.command)
            .status,
          'invalid',
        );
        if (oracle) {
          const value = oracle.rootValues.find(
            (r) => JSON.stringify(r.command) === JSON.stringify(result.command),
          )!.value;
          scoring = {
            referenceValue: value,
            best: oracle.best,
            regret: oracle.best - value,
            optimal: Math.abs(oracle.best - value) < 1e-9,
          };
        }
        if (job.recordedCommand)
          scoring = {
            ...scoring,
            immediateChosen: immediateCertificate(job.observation, result.command),
            immediateRecorded: immediateCertificate(job.observation, job.recordedCommand),
          };
      }
      rows.push({
        name: job.name,
        kind: job.kind,
        family: job.family,
        game: job.game,
        index: job.index,
        before: job.before,
        seed,
        budget,
        variant,
        elapsedMs,
        profile,
        ...(continuation ? { rollout: { ...continuation.stats } } : {}),
        ...result,
        ...scoring,
      });
      assert.equal(JSON.stringify(job.observation), before);
    }
    const eager = rows.find((r) => r.seed === seed && r.variant === 'eager-zero');
    const deferred = rows.find((r) => r.seed === seed && r.variant === 'deferred-zero');
    if (eager?.status === 'command' && deferred?.status === 'command') {
      assert.deepEqual(deferred.command, eager.command);
      assert.deepEqual(deferred.edges, eager.edges);
      assert.equal(deferred.stats.transitions, eager.stats.transitions);
      assert.equal(deferred.stats.terminalLeaves, eager.stats.terminalLeaves);
      assert.ok(deferred.stats.actionNodes <= eager.stats.actionNodes);
    }
  }
  return rows;
}

if (process.argv.includes('--worker')) {
  process.on('disconnect', () => process.exit(0));
  process.once('message', (job: Job) => {
    try {
      process.send!({ rows: probe(job) });
    } catch (error) {
      process.send!({ error: error instanceof Error ? error.stack : String(error) });
    }
  });
} else {
  const { values } = parseArgs({
    options: {
      source: { type: 'string' },
      output: { type: 'string' },
      fpu: { type: 'boolean' },
      rollout: { type: 'boolean' },
    },
  });
  assert.ok(values.source && values.output);
  const source = values.source;
  const output = values.output;
  mkdirSync(output, { recursive: true });
  const read = (name: string) => JSON.parse(readFileSync(resolve(source, name), 'utf8'));
  const write = (name: string, data: unknown) =>
    writeFileSync(resolve(output, name), JSON.stringify(data, null, 2), { flag: 'wx' });
  const previous = read('protocol.json');
  const selection = read('selection.json');
  assert.equal(read('audit.json').passed, true);
  assert.equal(encodingSourceHash(), previous.sourceSha256);
  assert.equal(await hashRecordFile(previous.input), previous.inputSha256);
  assert.equal(digest(readFileSync(previous.manifest)), previous.manifestSha256);
  // 只做成本筛查：原清单中前四个不同种子族各取第一个位置，保留不支持的位置。
  const seen = new Set<number>();
  const selected = selection.positions.filter((p: any) => {
    if (seen.has(p.seed) || seen.size === 4) return false;
    seen.add(p.seed);
    return true;
  });
  assert.equal(selected.length, 4);
  const scripts = [
    'scripts/training/search/puct.ts',
    'scripts/training/search/reference.ts',
    'scripts/training/search/positions.ts',
    'scripts/training/search/leaf/probe.ts',
    'scripts/training/search/leaf/rollout.ts',
  ];
  const scriptHashes = Object.fromEntries(scripts.map((f) => [f, digest(readFileSync(f))]));
  const protocol = {
    format: 'haojie-leaf-cost-probe-v1',
    source,
    previousProtocolSha256: digest(readFileSync(resolve(source, 'protocol.json'))),
    selectionSha256: digest(readFileSync(resolve(source, 'selection.json'))),
    sourceSha256: encodingSourceHash(),
    scriptHashes,
    selected,
    workers: 4,
    budget,
    seeds,
    variants,
    horizon: 2,
    maxActionNodes: 512,
    exploratoryFollowup: fpu,
    terminalRollout: rollout,
    terminalSelection: rollout
      ? 'last decision in each terminated source game; keep unsupported; exclude truncated from this subset'
      : null,
    rolloutBudget: rollout
      ? { commands: 1, teacher: 'easy', teacherWorkPerCall: 40, outerPlusRolloutTransitionsMax: 32 }
      : null,
    leaf: 'tanh(existing public-state heuristic / 1000), root actor perspective; estimate only',
    note: '四个原开发族的固定首位置成本筛查；30个人工规则夹具检查，均非新盲测或比赛',
  };
  write('protocol.json', protocol);
  const jobs: Job[] = [];
  const terminalJobs: Job[] = [];
  let last: any;
  for await (const r of readTrainingRecords(previous.input)) {
    if (r.type === 'game') last = undefined;
    if (r.type === 'decision') last = r;
    if (rollout && r.type === 'outcome' && r.terminated) {
      assert.ok(last && last.game === r.game && last.index === r.commands - 1);
      terminalJobs.push({
        name: `terminal-game-${last.game}-index-${last.index}`,
        kind: 'terminal-natural',
        game: last.game,
        index: last.index,
        before: last.before,
        observation: last.observation,
        recordedCommand: last.command,
      });
    }
    if (
      r.type === 'decision' &&
      selected.some((p: any) => p.game === r.game && p.index === r.index)
    ) {
      assert.equal(
        selected.find((p: any) => p.game === r.game && p.index === r.index).before,
        r.before,
      );
      jobs.push({
        name: `game-${r.game}-index-${r.index}`,
        kind: 'natural',
        game: r.game,
        index: r.index,
        before: r.before,
        observation: r.observation,
      });
    }
  }
  assert.equal(jobs.length, 4);
  if (rollout) {
    assert.equal(terminalJobs.length, 8);
    write(
      'terminal-selection.json',
      terminalJobs.map(({ observation, ...job }) => job),
    );
    jobs.push(...terminalJobs);
  }
  const fixtures = positions();
  // 不重写旧指纹；只用旧已保存结果验证默认算法改造前后完全一致。
  const saved: any[] = JSON.parse(
    readFileSync('artifacts/training/search-probe-final-20260925/decisions.json', 'utf8'),
  );
  for (const p of fixtures) {
    const old = saved.find((r) => r.name === p.name && r.seed === 2026092721 && r.budget === 32);
    assert.ok(old);
    const result = search(p.observation, { simulations: 32, horizon: 2, sampleSeed: 2026092721 });
    for (const [key, value] of Object.entries(result)) assert.deepEqual(old[key], value);
    jobs.push({ name: p.name, family: p.family, kind: 'fixture', observation: p.observation });
  }
  const results: any[][] = Array(jobs.length);
  let next = 0;
  let active = 0;
  let maxActive = 0;
  const started = performance.now();
  await Promise.all(
    Array.from({ length: 4 }, async () => {
      while (next < jobs.length) {
        const index = next++;
        const child = spawn(
          process.execPath,
          [
            '--import',
            'tsx',
            fileURLToPath(import.meta.url),
            '--worker',
            ...(fpu ? ['--fpu'] : []),
            ...(rollout ? ['--rollout'] : []),
          ],
          { stdio: ['ignore', 'ignore', 'inherit', 'ipc'], windowsHide: true },
        );
        active++;
        maxActive = Math.max(maxActive, active);
        try {
          const response = Promise.race([
            once(child, 'message').then(([message]) => message),
            once(child, 'exit').then(([code]) => {
              throw new Error(`实验子进程提前退出：${code}`);
            }),
          ]);
          child.send!(jobs[index]);
          const message = await response;
          assert.ok(!message.error, message.error);
          assert.equal(message.rows.length, variants.length * seeds.length);
          results[index] = message.rows;
          write(`part-${index}.json`, message.rows);
          console.log(JSON.stringify({ completed: jobs[index].name, rows: message.rows.length }));
        } finally {
          active--;
          child.kill();
        }
      }
    }),
  );
  for (const [f, hash] of Object.entries(scriptHashes)) assert.equal(digest(readFileSync(f)), hash);
  assert.equal(encodingSourceHash(), protocol.sourceSha256);
  assert.equal(await hashRecordFile(previous.input), previous.inputSha256);
  const rows = results.flat();
  write('decisions.json', rows);
  const groups = (
    rollout ? ['natural', 'fixture', 'terminal-natural'] : ['natural', 'fixture']
  ).flatMap((kind) =>
    variants.map((variant) => {
      const group = rows.filter((r) => r.kind === kind && r.variant === variant);
      const ok = group.filter((r) => r.status === 'command');
      const times = group.map((r) => r.elapsedMs).sort((a, b) => a - b);
      return {
        kind,
        variant,
        requested: group.length,
        commands: ok.length,
        changedFromBaseline: ok.filter(
          (r) => JSON.stringify(r.command) !== JSON.stringify(r.baseline),
        ).length,
        informativeVisitedValues: ok.filter((r) => {
          const v = r.edges.filter((e: any) => e.visits).map((e: any) => e.value);
          return Math.max(...v) - Math.min(...v) > 1e-9;
        }).length,
        optimal: kind === 'fixture' ? ok.filter((r) => r.optimal).length : null,
        regret: kind === 'fixture' ? ok.reduce((n, r) => n + r.regret, 0) : null,
        elapsedMs: group.reduce((n, r) => n + r.elapsedMs, 0),
        medianMs: times[Math.ceil(times.length / 2) - 1],
        enumerationMs: group.reduce((n, r) => n + r.profile.enumerationMs, 0),
        transitionMs: group.reduce((n, r) => n + r.profile.transitionMs, 0),
        leafMs: group.reduce((n, r) => n + r.profile.leafMs, 0),
        expanded: group.reduce((n, r) => n + r.stats.expanded, 0),
        terminalLeaves: group.reduce((n, r) => n + r.stats.terminalLeaves, 0),
        rolloutTransitions: group.reduce((n, r) => n + (r.rollout?.transitions ?? 0), 0),
        rolloutTerminals: group.reduce((n, r) => n + (r.rollout?.terminal ?? 0), 0),
        rolloutUnknown: group.reduce((n, r) => n + (r.rollout?.unknown ?? 0), 0),
        teacherWork: group.reduce((n, r) => n + (r.rollout?.teacherWork ?? 0), 0),
        certifiedRecordedWins: ok.filter(
          (r) =>
            r.immediateRecorded?.status === 'exact' && r.immediateRecorded.winProbability === 1,
        ).length,
        certifiedChosenWins: ok.filter(
          (r) => r.immediateChosen?.status === 'exact' && r.immediateChosen.winProbability === 1,
        ).length,
        reasons: group
          .filter((r) => r.status === 'paused')
          .map((r) => ({ name: r.name, seed: r.seed, reason: r.reason })),
      };
    }),
  );
  const summary = {
    complete: true,
    protocolSha256: digest(readFileSync(resolve(output, 'protocol.json'))),
    rows: rows.length,
    maxActive,
    elapsedMs: performance.now() - started,
    legacyRechecks: fixtures.length,
    noTrainingLabels: true,
    groups,
  };
  write('summary.json', summary);
  console.log(JSON.stringify(summary));
}
