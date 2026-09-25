import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { checkPosition, search, terminalValue, windowBoundary } from '../puct';
import { terminalRollout } from './rollout';
import { encodingSourceHash } from '../../encode';
import { hashRecordFile } from '../../records/io';
import { readTrainingRecords } from '../../records/replay';
import { decisionOwner, fingerprint } from '../../../../src/ai/observation';
import { decide } from '../../../../src/ai/planning/search';
import { inspectTrainingCommand } from '../../../../src/ai/training/queries';
import {
  sampleTrainingTransition,
  simulationRandomSource,
} from '../../../../src/ai/training/simulation';
import type { Observation } from '../../../../src/ai/types';

const maxCommands = 8;
const variants = ['terminal-rollout', 'easy-40'] as const;
interface Job {
  name: string;
  observation: Observation;
  searchSeed: number;
  executionSeed: number;
  variant: (typeof variants)[number];
}
const digest = (file: string) => createHash('sha256').update(readFileSync(file)).digest('hex');
const read = (dir: string, file: string) => JSON.parse(readFileSync(resolve(dir, file), 'utf8'));
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

/**
 * 从已重放的公开局面连续重搜，每次只执行新决策；不缓存教师续招、不读历史最后命令。
 * 执行随机流独立于搜索；仅保存命令、指纹、独立模拟编号及统计，不生成逐步快照或训练标签。
 * 到终局、窗口边界、不支持、完全重复局面或固定8命令上限时明确停止；未知不得称作平局。
 */
function follow(job: Job) {
  const initial = JSON.stringify(job.observation);
  const rootActor = decisionOwner(job.observation);
  const execution = simulationRandomSource(job.executionSeed);
  const seen = new Set<string>();
  const steps: any[] = [];
  let o = job.observation;
  let stop = 'command-limit',
    reason: string | undefined;
  for (let index = 0; index < maxCommands; index++) {
    if (o.winner !== undefined) {
      stop = 'terminal';
      break;
    }
    if (windowBoundary(o)) {
      stop = 'window-boundary';
      break;
    }
    if (seen.has(JSON.stringify(o))) {
      stop = 'repeated-position';
      break;
    }
    seen.add(JSON.stringify(o));
    try {
      checkPosition(o);
    } catch (error) {
      stop = 'paused';
      reason = String(error);
      break;
    }
    const before = JSON.stringify(o);
    const actor = decisionOwner(o);
    const continuation = terminalRollout(job.searchSeed);
    const started = performance.now();
    const result =
      job.variant === 'terminal-rollout'
        ? search(o, {
            simulations: 16,
            horizon: 2,
            maxActionNodes: 512,
            sampleSeed: job.searchSeed,
            deferExpansion: true,
            leafValue: continuation.leafValue,
          })
        : (() => {
            const decision = decide(o, actor, 'easy', { simulations: 40, mode: 'work' });
            assert.ok(decision.stats.simulations <= 40);
            return {
              status: decision.command ? 'command' : 'paused',
              command: decision.command,
              stats: decision.stats,
            };
          })();
    const elapsedMs = performance.now() - started;
    assert.equal(JSON.stringify(o), before);
    if ('transitions' in result.stats)
      assert.ok(result.stats.transitions + continuation.stats.transitions <= 32);
    if (result.status !== 'command' || !result.command) {
      stop = 'paused';
      reason = 'reason' in result ? result.reason : '教师未返回命令';
      steps.push({
        index,
        before: fingerprint(o),
        actor,
        elapsedMs,
        result,
        ...(job.variant === 'terminal-rollout' ? { rollout: continuation.stats } : {}),
      });
      break;
    }
    assert.notEqual(inspectTrainingCommand(o, actor, result.command).status, 'invalid');
    const executionSample = Math.floor(execution() * 4294967296);
    const next = sampleTrainingTransition(o, actor, result.command, executionSample);
    steps.push({
      index,
      before: fingerprint(o),
      after: fingerprint(next),
      actor,
      executionSample,
      elapsedMs,
      result,
      ...(job.variant === 'terminal-rollout' ? { rollout: continuation.stats } : {}),
    });
    o = next;
  }
  if (o.winner !== undefined) stop = 'terminal';
  else if (windowBoundary(o)) stop = 'window-boundary';
  assert.equal(JSON.stringify(job.observation), initial);
  const { observation, ...identity } = job;
  return {
    ...identity,
    rootActor,
    stop,
    reason,
    winner: o.winner ?? null,
    value: terminalValue(o, rootActor),
    commands: steps.filter((s) => s.after).length,
    final: fingerprint(o),
    steps,
  };
}

if (process.argv.includes('--worker')) {
  process.on('disconnect', () => process.exit(0));
  process.once('message', (job: Job) => {
    try {
      process.send!({ result: follow(job) });
    } catch (error) {
      process.send!({ error: error instanceof Error ? error.stack : String(error) });
    }
  });
} else {
  const { values } = parseArgs({
    options: {
      source: { type: 'string' },
      output: { type: 'string' },
      compare: { type: 'string' },
    },
  });
  assert.ok(values.source && values.output);
  const output = values.output;
  mkdirSync(output, { recursive: true });
  const write = (name: string, value: unknown) =>
    writeFileSync(resolve(output, name), JSON.stringify(value, null, 2), { flag: 'wx' });
  const previous = read(values.source, 'protocol.json');
  const priorAudit = read(values.source, 'audit.json');
  assert.equal(priorAudit.passed, true);
  assert.equal(priorAudit.protocolSha256, digest(resolve(values.source, 'protocol.json')));
  assert.equal(priorAudit.decisionsSha256, digest(resolve(values.source, 'decisions.json')));
  assert.equal(
    priorAudit.terminalSelectionSha256,
    digest(resolve(values.source, 'terminal-selection.json')),
  );
  const origin = read(previous.source, 'protocol.json');
  assert.equal(encodingSourceHash(), origin.sourceSha256);
  assert.equal(await hashRecordFile(origin.input), origin.inputSha256);
  assert.equal(digest(origin.manifest), origin.manifestSha256);
  const selected = read(values.source, 'terminal-selection.json');
  const scriptHashes = Object.fromEntries(
    [
      'scripts/training/search/puct.ts',
      'scripts/training/search/leaf/rollout.ts',
      'scripts/training/search/leaf/follow-through.ts',
    ].map((file) => [file, digest(file)]),
  );
  const protocol = {
    format: 'haojie-follow-through-probe-v1',
    source: values.source,
    previousProtocolSha256: priorAudit.protocolSha256,
    sourceSha256: encodingSourceHash(),
    input: origin.input,
    inputSha256: origin.inputSha256,
    manifest: origin.manifest,
    manifestSha256: origin.manifestSha256,
    scriptHashes,
    selected,
    pairs: [
      { searchSeed: 2026092731, executionSeed: 2026092741 },
      { searchSeed: 2026092732, executionSeed: 2026092742 },
    ],
    maxCommands,
    variants,
    workers: 4,
    simulations: 16,
    horizon: 2,
    maxActionNodes: 512,
    teacherWorkPerCall: 40,
    searchSeedPolicy: 'reuse fixed search seed at every fresh decision',
    note: 'old development terminal subset; public simulation continuations, not full matches or training records',
  };
  write('protocol.json', protocol);
  const observations = new Map<string, Observation>();
  let games = 0,
    last: any;
  const outcomes: any[] = [],
    reconstructed: any[] = [];
  for await (const r of readTrainingRecords(origin.input)) {
    if (r.type === 'game') {
      games++;
      last = undefined;
    }
    if (r.type === 'decision') last = r;
    if (r.type === 'outcome') {
      outcomes.push(r);
      if (r.terminated) {
        assert.ok(last && last.game === r.game && last.index === r.commands - 1);
        const name = `terminal-game-${r.game}-index-${last.index}`;
        reconstructed.push({
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
  assert.deepEqual(reconstructed, selected);
  assert.equal(games, 10);
  assert.equal(outcomes.length, 10);
  assert.equal(observations.size, 8);
  // 仅传公开局面和独立实验编号，不把历史命令或正式随机状态传给子进程。
  const jobs: Job[] = [...observations].flatMap(([name, observation]) =>
    protocol.pairs.flatMap((pair) =>
      variants.map((variant) => ({ name, observation, ...pair, variant })),
    ),
  );
  const results: ReturnType<typeof follow>[] = Array(jobs.length);
  const children = new Set<ReturnType<typeof spawn>>();
  let next = 0,
    active = 0,
    maxActive = 0;
  const started = performance.now();
  try {
    await Promise.all(
      Array.from({ length: 4 }, async () => {
        while (next < jobs.length) {
          const index = next++;
          const child = spawn(
            process.execPath,
            ['--import', 'tsx', fileURLToPath(import.meta.url), '--worker'],
            { stdio: ['ignore', 'ignore', 'inherit', 'ipc'], windowsHide: true },
          );
          children.add(child);
          active++;
          maxActive = Math.max(maxActive, active);
          try {
            const response = Promise.race([
              once(child, 'message').then(([message]) => message),
              once(child, 'exit').then(([code]) => {
                throw new Error(`续局子进程提前退出：${code}`);
              }),
            ]);
            child.send!(jobs[index]);
            const message = await response;
            assert.ok(!message.error, message.error);
            results[index] = message.result;
            write(`part-${index}.json`, message.result);
            console.log(
              JSON.stringify({
                index,
                name: jobs[index].name,
                variant: jobs[index].variant,
                stop: message.result.stop,
                commands: message.result.commands,
              }),
            );
          } finally {
            active--;
            child.kill();
            children.delete(child);
          }
        }
      }),
    );
  } finally {
    for (const child of children) child.kill();
  }
  const elapsedMs = performance.now() - started;
  for (const [file, hash] of Object.entries(scriptHashes)) assert.equal(digest(file), hash);
  assert.equal(encodingSourceHash(), protocol.sourceSha256);
  assert.equal(await hashRecordFile(origin.input), origin.inputSha256);
  write('results.json', results);
  // 主进程从已验证起点逐条重放子进程命令，核对权限、独立随机编号、指纹和终态。
  let replayedCommands = 0,
    initialComparisons = 0;
  const oldRows = read(values.source, 'decisions.json');
  for (const [i, row] of results.entries()) {
    const job = jobs[i];
    assert.equal(row.name, job.name);
    assert.equal(row.variant, job.variant);
    let o = job.observation;
    const random = simulationRandomSource(job.executionSeed);
    for (const step of row.steps) {
      assert.equal(step.before, fingerprint(o));
      assert.equal(step.actor, decisionOwner(o));
      if (!step.after) {
        assert.equal(step.result.status, 'paused');
        continue;
      }
      assert.equal(step.executionSample, Math.floor(random() * 4294967296));
      assert.notEqual(inspectTrainingCommand(o, step.actor, step.result.command).status, 'invalid');
      o = sampleTrainingTransition(o, step.actor, step.result.command, step.executionSample);
      assert.equal(step.after, fingerprint(o));
      replayedCommands++;
    }
    assert.equal(row.final, fingerprint(o));
    assert.equal(row.winner, o.winner ?? null);
    assert.equal(row.value, terminalValue(o, row.rootActor));
    assert.equal(row.commands, row.steps.filter((s) => s.after).length);
    if (row.stop === 'terminal') assert.notEqual(o.winner, undefined);
    else assert.equal(row.value, null);
    if (job.variant === 'terminal-rollout' && row.steps.length) {
      const old = oldRows.find(
        (r: any) => r.name === row.name && r.seed === row.searchSeed && r.variant === row.variant,
      );
      assert.ok(old);
      for (const [key, value] of Object.entries(row.steps[0].result))
        assert.deepEqual(old[key], value);
      assert.deepEqual(untimed(old.rollout), untimed(row.steps[0].rollout));
      initialComparisons++;
    }
  }
  let repeatedPaths = 0;
  if (values.compare) {
    assert.deepEqual(protocol, read(values.compare, 'protocol.json'));
    assert.deepEqual(untimed(results), untimed(read(values.compare, 'results.json')));
    repeatedPaths = results.length;
  }
  const summary = {
    complete: true,
    protocolSha256: digest(resolve(output, 'protocol.json')),
    resultsSha256: digest(resolve(output, 'results.json')),
    elapsedMs,
    maxActive,
    paths: results.length,
    replayedCommands,
    initialComparisons,
    repeatedPaths,
    sourceOutcomes: {
      terminal: outcomes.filter((r) => r.terminated).length,
      truncated: outcomes.filter((r) => r.truncated).length,
      interrupted: outcomes.filter((r) => r.interrupted).length,
    },
    groups: variants.map((variant) => {
      const rows = results.filter((r) => r.variant === variant);
      return {
        variant,
        paths: rows.length,
        terminal: rows.filter((r) => r.stop === 'terminal').length,
        wins: rows.filter((r) => r.value === 1).length,
        losses: rows.filter((r) => r.value === -1).length,
        draws: rows.filter((r) => r.value === 0).length,
        stops: Object.fromEntries(
          [...new Set(rows.map((r) => r.stop))].map((stop) => [
            stop,
            rows.filter((r) => r.stop === stop).length,
          ]),
        ),
        commands: rows.reduce((n, r) => n + r.commands, 0),
        teacherWork: rows.reduce(
          (n, r) =>
            n +
            r.steps.reduce((n, s) => n + (s.rollout?.teacherWork ?? s.result.stats.simulations), 0),
          0,
        ),
        elapsedMs: rows.reduce((n, r) => n + r.steps.reduce((n, s) => n + s.elapsedMs, 0), 0),
      };
    }),
    noTrainingLabels: true,
  };
  write('summary.json', summary);
  console.log(JSON.stringify(summary));
}
