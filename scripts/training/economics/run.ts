import assert from 'node:assert/strict';
import { spawn, execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { once } from 'node:events';
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { cpus, totalmem } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { setImmediate } from 'node:timers/promises';
import { fingerprint } from '../../../src/ai/observation';
import { trainingPosition } from '../../../src/ai/training/queries';
import { allPieces, hasTrait } from '../../../src/engine/core/traits';
import type { Player } from '../../../src/engine/types';
import { TrainingEnvironment } from '../../../src/match/training';
import { withRecordOutput, hashRecordFile } from '../records/io';
import { readTrainingRecords, recordHeader } from '../records/replay';
import { TinyPolicy, randomStream } from './policy';
import { sampleCommand, emptyMetrics } from './sample';

export interface Options {
  seconds: number;
  workers: number;
  worker: number;
  rules: 'classic' | 'shrine';
  policy: 'tiny' | 'uniform';
  seed: number;
  maxCommands: number;
  maxPlies: number;
}

/** 单进程真实自然开局；全部成功命令持久化，未知终局不填收益，异常不吞掉。 */
export async function sampleWorker(options: Options, emit: (row: unknown) => Promise<void>) {
  const started = performance.now();
  const deadline = started + options.seconds * 1000;
  const policySeed = [73129, 95267, 117101][options.worker % 3];
  const policy = options.policy === 'tiny' ? new TinyPolicy(policySeed) : null;
  const results: any[] = [];
  let stopped = false;
  const stop = () => {
    stopped = true;
  };
  process.once('SIGINT', stop);
  process.once('SIGTERM', stop);
  try {
    for (let localGame = 0; !stopped && performance.now() < deadline; localGame++) {
      const game = localGame * options.workers + options.worker;
      const seed = options.seed + game;
      const samplerSeed = (0x6ab921d3 + Math.imul(game + 1, 2654435761)) >>> 0;
      const random = randomStream(samplerSeed);
      const env = new TrainingEnvironment({
        seed,
        rules: options.rules,
        ...{
          maxCommands: options.maxCommands,
          maxPlies: options.maxPlies,
        },
      });
      const metrics = emptyMetrics();
      const gameStart = performance.now();
      const commandTypes: Record<string, number> = {};
      await emit({
        type: 'game',
        ...recordHeader(env),
        game,
        gameId: `economics:${options.rules}:${options.policy}:${seed}`,
        source: 'neural',
        experiment: 'cold-start-economics',
        seed,
        rules: options.rules,
        policy: options.policy,
        policySeed,
        samplerSeed,
        parameters: policy?.parameters ?? 0,
      });
      let interrupted: string | null = null;
      let error: string | null = null;
      let offerInterrupt = true;
      while (!env.status().terminated && !env.status().truncated) {
        if (stopped || performance.now() >= deadline) {
          interrupted = stopped ? 'cancelled' : 'wall-time';
          break;
        }
        let actor = env.status().toPlay!;
        let start = performance.now();
        let observation = env.observation(actor);
        if (observation.phase === 'shrine-draft' && observation.shrineDraft?.committed[actor]) {
          actor = (3 - actor) as Player;
          observation = env.observation(actor);
        }
        metrics.observationMs += performance.now() - start;
        try {
          let selected: ReturnType<typeof sampleCommand> | undefined;
          const other = (3 - actor) as Player;
          start = performance.now();
          const offTurn =
            offerInterrupt &&
            !observation.pending.length &&
            observation.phase !== 'shrine-draft' &&
            allPieces(trainingPosition(observation)).some(
              (u) => u.owner === other && hasTrait(u, 'u7'),
            );
          metrics.treeMs += performance.now() - start;
          if (offTurn) {
            start = performance.now();
            const interruptObservation = env.observation(other);
            metrics.observationMs += performance.now() - start;
            selected = sampleCommand(interruptObservation, other, policy, random, metrics, true);
            if (selected.command) {
              actor = other;
              observation = interruptObservation;
              metrics.offTurnCommands++;
              offerInterrupt = false;
            } else if (selected.passed) metrics.offTurnPasses++;
            else throw new Error(selected.error);
          }
          if (!selected?.command) {
            selected = sampleCommand(observation, actor, policy, random, metrics);
            offerInterrupt = true;
          }
          assert.ok(selected.command, selected.error ?? '常规决策不能pass');
          start = performance.now();
          const before = fingerprint(observation);
          metrics.recordMs += performance.now() - start;
          const index = env.status().commands;
          start = performance.now();
          env.step(actor, selected.command);
          metrics.stepMs += performance.now() - start;
          commandTypes[selected.command.type] = (commandTypes[selected.command.type] ?? 0) + 1;
          start = performance.now();
          await emit({
            type: 'decision',
            game,
            index,
            actor,
            command: selected.command,
            before,
            after: fingerprint(env.observation()),
            source: options.policy,
          });
          metrics.recordMs += performance.now() - start;
        } catch (reason) {
          error = reason instanceof Error ? (reason.stack ?? reason.message) : String(reason);
          interrupted = 'implementation-error';
          await emit({
            type: 'error',
            game,
            index: env.status().commands,
            actor,
            before: fingerprint(env.observation(actor)),
            error,
          });
          break;
        }
        // 让压缩输出、取消信号及时处理；不把时钟作为策略特征或选择预算。
        await setImmediate();
      }
      const result = {
        type: 'outcome',
        game,
        seed,
        ...env.status(),
        interrupted,
        after: fingerprint(env.observation()),
        elapsedMs: performance.now() - gameStart,
        metrics,
        commandTypes,
        error,
      };
      await emit(result);
      results.push(result);
      console.log(
        JSON.stringify({
          worker: options.worker,
          game,
          commands: result.commands,
          ply: result.ply,
          terminated: result.terminated,
          truncated: result.truncated,
          interrupted,
        }),
      );
      if (error) break;
    }
  } finally {
    process.removeListener('SIGINT', stop);
    process.removeListener('SIGTERM', stop);
  }
  return {
    options,
    policySeed,
    parameters: policy?.parameters ?? 0,
    elapsedMs: performance.now() - started,
    results,
  };
}

function sourceIdentity() {
  const digest = createHash('sha256');
  const files: string[] = [];
  const scan = (directory: string) => {
    for (const item of readdirSync(directory, { withFileTypes: true }).sort((a, b) =>
      a.name.localeCompare(b.name),
    )) {
      const path = join(directory, item.name);
      if (item.isDirectory()) scan(path);
      else if (/\.(ts|json|md)$/.test(path)) files.push(path);
    }
  };
  for (const dir of [
    'src/engine',
    'src/ai/training',
    'src/match',
    'scripts/training/economics',
    'scripts/training/records',
  ])
    scan(dir);
  files.push(
    'src/ai/observation.ts',
    'src/ai/types.ts',
    'package-lock.json',
    'docs/ai/research/ECONOMICS-PROTOCOL-2026-09-26.md',
  );
  for (const file of files)
    digest.update(file.replaceAll('\\', '/') + '\0').update(readFileSync(file));
  return { sha256: digest.digest('hex'), files };
}

async function main() {
  const { values } = parseArgs({
    options: Object.fromEntries(
      [
        'seconds',
        'workers',
        'worker',
        'rules',
        'policy',
        'seed',
        'commands',
        'plies',
        'output',
      ].map((name) => [name, { type: 'string' as const }]),
    ),
  });
  const positive = (value: string | undefined, fallback: number) => {
    const n = Number(value ?? fallback);
    assert.ok(Number.isSafeInteger(n) && n > 0);
    return n;
  };
  const options: Options = {
    seconds: positive(values.seconds, 300),
    workers: positive(values.workers, 8),
    worker: Number(values.worker ?? 0),
    rules: (values.rules ?? 'classic') as Options['rules'],
    policy: (values.policy ?? 'tiny') as Options['policy'],
    seed: positive(values.seed, 271828001),
    maxCommands: positive(values.commands, 10000),
    maxPlies: positive(values.plies, 500),
  };
  assert.ok(
    ['classic', 'shrine'].includes(options.rules) && ['tiny', 'uniform'].includes(options.policy),
  );
  assert.ok(
    Number.isSafeInteger(options.worker) && options.worker >= 0 && options.worker < options.workers,
  );
  assert.ok(values.output, '必须提供新的输出目录');
  const directory = resolve(values.output);
  if (values.worker !== undefined) {
    const prefix = join(directory, `worker-${options.worker}`);
    const report = await withRecordOutput(`${prefix}.jsonl.gz`, (emit) =>
      sampleWorker(options, emit),
    );
    writeFileSync(`${prefix}.json`, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
    return;
  }
  mkdirSync(directory, { recursive: false });
  const source = sourceIdentity();
  const manifest = {
    options,
    source,
    commit: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
    startedAt: new Date().toISOString(),
    cpu: cpus().map((c) => c.model),
    totalmem: totalmem(),
    node: process.version,
    mode: 'timed-sampling-workload',
  };
  writeFileSync(join(directory, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n', {
    flag: 'wx',
  });
  const started = performance.now();
  const jobs = await Promise.all(
    Array.from({ length: options.workers }, async (_, worker) => {
      const args = [
        '--import',
        'tsx',
        fileURLToPath(import.meta.url),
        '--worker',
        String(worker),
        '--seconds',
        String(options.seconds),
        '--workers',
        String(options.workers),
        '--rules',
        options.rules,
        '--policy',
        options.policy,
        '--seed',
        String(options.seed),
        '--commands',
        String(options.maxCommands),
        '--plies',
        String(options.maxPlies),
        '--output',
        directory,
      ];
      const child = spawn(process.execPath, args, {
        stdio: ['ignore', 'pipe', 'pipe'],
        windowsHide: true,
      });
      let log = '';
      child.stdout.on('data', (chunk) => {
        log += chunk;
      });
      child.stderr.on('data', (chunk) => {
        log += chunk;
      });
      const timeout = setTimeout(() => child.kill(), (options.seconds + 120) * 1000);
      try {
        const [code, signal] = await once(child, 'close');
        writeFileSync(join(directory, `worker-${worker}.log`), log, { flag: 'wx' });
        return { worker, pid: child.pid, code, signal };
      } finally {
        clearTimeout(timeout);
      }
    }),
  );
  const elapsedMs = performance.now() - started;
  assert.equal(sourceIdentity().sha256, source.sha256, '测量期间源代码改变');
  writeFileSync(join(directory, 'jobs.json'), JSON.stringify({ jobs, elapsedMs }, null, 2) + '\n', {
    flag: 'wx',
  });
  assert.ok(
    jobs.every((j) => j.code === 0),
    '子进程异常；保留已有记录，不能当成完成',
  );
  const reports = jobs.map((j) =>
    JSON.parse(readFileSync(join(directory, `worker-${j.worker}.json`), 'utf8')),
  );
  const replayStart = performance.now();
  const audits = [];
  for (const job of jobs) {
    const path = join(directory, `worker-${job.worker}.jsonl.gz`);
    let commands = 0,
      games = 0,
      outcomes = 0;
    for await (const row of readTrainingRecords(path)) {
      if (row.type === 'game') games++;
      if (row.type === 'decision') commands++;
      if (row.type === 'outcome') outcomes++;
    }
    const report = reports[job.worker];
    assert.equal(
      commands,
      report.results.reduce((n: number, r: any) => n + r.commands, 0),
    );
    assert.equal(games, report.results.length);
    assert.equal(outcomes, games);
    audits.push({
      worker: job.worker,
      commands,
      games,
      outcomes,
      sha256: await hashRecordFile(path),
    });
  }
  const results = reports.flatMap((r) => r.results);
  const sum = (f: (r: any) => number) => results.reduce((n: number, r: any) => n + f(r), 0);
  const terminal = results.filter((r) => r.terminated);
  const metrics = Object.fromEntries(
    Object.keys(emptyMetrics()).map((key) => [
      key,
      key.startsWith('max')
        ? Math.max(0, ...results.map((r) => r.metrics[key]))
        : sum((r) => r.metrics[key]),
    ]),
  );
  const summary = {
    ...manifest,
    finishedAt: new Date().toISOString(),
    elapsedMs,
    replayMs: performance.now() - replayStart,
    audits,
    jobs,
    parameters: reports.map((r) => r.parameters),
    counts: {
      games: results.length,
      terminal: terminal.length,
      decisive: terminal.filter((r) => r.winner !== 'draw').length,
      truncated: results.filter((r) => r.truncated).length,
      wallCensored: results.filter((r) => r.interrupted === 'wall-time').length,
      errors: results.filter((r) => r.error).length,
      commands: sum((r) => r.commands),
      terminalCommands: terminal.reduce((n, r) => n + r.commands, 0),
    },
    commandsPerSecond: sum((r) => r.commands) / (elapsedMs / 1000),
    terminalPerHour: terminal.length / (elapsedMs / 3600000),
    daysPerMillionTerminal: terminal.length
      ? 1e6 / (terminal.length / (elapsedMs / 86400000))
      : null,
    metrics,
    results,
  };
  writeFileSync(join(directory, 'summary.json'), JSON.stringify(summary, null, 2) + '\n', {
    flag: 'wx',
  });
  console.log(
    JSON.stringify({
      directory,
      elapsedMs,
      counts: summary.counts,
      commandsPerSecond: summary.commandsPerSecond,
      terminalPerHour: summary.terminalPerHour,
    }),
  );
  if (summary.counts.errors) process.exitCode = 1;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
