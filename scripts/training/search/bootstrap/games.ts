import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { setImmediate } from 'node:timers/promises';
import { TrainingEnvironment } from '../../../../src/match/training';
import { decisionOwner, fingerprint, hash } from '../../../../src/ai/observation';
import { decide } from '../../../../src/ai/planning/search';
import { bootstrapDecision, bootstrapValueDecision } from './policy';
import { openValueModel } from '../value-cycle/model';
import { recordHeader, readTrainingRecords } from '../../records/replay';
import { withRecordOutput, hashRecordFile } from '../../records/io';
import { encodingSourceHash } from '../../encode';
import type { Player } from '../../../../src/engine/types';

interface Job {
  game: number;
  seed: number;
  primary: Player;
  kind: 'evaluation' | 'selfplay';
  output: string;
  checkpoint?: string;
  opponentCheckpoint?: string;
  maxPlies?: number;
  maxCommands?: number;
}
const digest = (file: string) => createHash('sha256').update(readFileSync(file)).digest('hex');

/** 真正从开局执行完整规则；宿主才持有正式随机状态，搜索只收公开观察及独立模拟编号。 */
async function run(job: Job) {
  const env = new TrainingEnvironment({
    seed: job.seed,
    rules: 'classic',
    maxPlies: job.maxPlies ?? 120,
    maxCommands: job.maxCommands ?? 1800,
  });
  const cancellation = new AbortController();
  process.once('SIGINT', () => cancellation.abort());
  const path = resolve(job.output, `game-${job.game}.jsonl.gz`);
  const counts: Record<string, number> = {};
  const work = { teacher: 0, simulations: 0, transitions: 0 };
  const started = performance.now();
  const gameId = `bootstrap-v1:${job.kind}:${job.seed}:${job.primary}`;
  const learned = job.checkpoint
    ? await openValueModel(job.checkpoint, cancellation.signal)
    : undefined;
  let opponent: Awaited<ReturnType<typeof openValueModel>> | undefined;
  try {
    if (job.opponentCheckpoint) {
      assert.equal(job.kind, 'evaluation');
      opponent = await openValueModel(job.opponentCheckpoint, cancellation.signal);
    }
    await withRecordOutput(path, async (emit) => {
      await emit({
        type: 'game',
        game: job.game,
        gameId,
        ...recordHeader(env),
        source: 'teacher',
        seed: job.seed,
        rules: 'classic',
        primaryPlayer: job.primary,
        policyKind: 'teacher-assisted-restricted-puct-v1',
        experimentKind: job.kind,
        policyTarget:
          'conditional candidate visits; null on fallback/opponent; not full legal domain',
        budget: { outerSimulations: 16, teacherWorkPerCall: 40, teacherCallsMax: 17 },
        actualTeacherWorkSafetyCap: 256,
        ...(learned
          ? { valueModelSha256: learned.model.ready.checkpoint_sha256, neuralLeafScale: 0.25 }
          : {}),
        ...(opponent ? { opponentModelSha256: opponent.model.ready.checkpoint_sha256 } : {}),
      });
      let interrupted: string | null = null,
        error: string | null = null;
      while (!env.status().terminated && !env.status().truncated) {
        if (cancellation.signal.aborted) {
          interrupted = 'cancelled';
          break;
        }
        const observation = env.observation(),
          actor = decisionOwner(observation);
        const index = env.status().commands;
        const searchSeed = hash(`bootstrap-experiment:2026092851:${fingerprint(observation)}`);
        const before = fingerprint(observation);
        let result;
        const begin = performance.now();
        try {
          if (job.kind === 'selfplay' || actor === job.primary)
            result = learned
              ? await bootstrapValueDecision(
                  observation,
                  searchSeed,
                  learned.value,
                  cancellation.signal,
                )
              : bootstrapDecision(observation, searchSeed);
          else if (opponent)
            result = await bootstrapValueDecision(
              observation,
              searchSeed,
              opponent.value,
              cancellation.signal,
            );
          else {
            const d = decide(observation, actor, 'easy', { simulations: 40, mode: 'work' });
            assert.ok(
              d.command && Number.isSafeInteger(d.stats.simulations) && d.stats.simulations <= 256,
            );
            result = {
              command: d.command,
              mode: 'opponent',
              policy: null,
              stats: {
                teacherCalls: 1,
                teacherWork: d.stats.simulations,
                searchSimulations: 0,
                searchTransitions: 0,
                rolloutTransitions: 0,
              },
            };
          }
          await setImmediate();
          if (cancellation.signal.aborted) {
            interrupted = 'cancelled';
            break;
          }
          env.step(actor, result.command);
        } catch (e) {
          interrupted = 'decision-or-command-error';
          error = String(e);
          break;
        }
        const mode = result.mode === 'teacher-fallback' ? `fallback:${result.reason}` : result.mode;
        counts[mode] = (counts[mode] ?? 0) + 1;
        work.teacher += result.stats.teacherWork;
        work.simulations += result.stats.searchSimulations;
        work.transitions += result.stats.searchTransitions + result.stats.rolloutTransitions;
        await emit({
          type: 'sample',
          game: job.game,
          index,
          actor,
          before,
          command: result.command,
          after: fingerprint(env.observation()),
          searchSeed,
          policyMode: mode,
          searchPolicy: result.policy,
          searchStats: result.stats,
          valueStats: 'valueStats' in result ? result.valueStats : null,
          elapsedMs: performance.now() - begin,
        });
        if (env.status().commands % 50 === 0) {
          const progress = {
            progress: {
              game: job.game,
              commands: env.status().commands,
              ply: env.status().ply,
              counts,
            },
          };
          if (process.send) process.send(progress);
          else console.log(JSON.stringify(progress));
        }
      }
      await emit({
        type: 'outcome',
        game: job.game,
        gameId,
        ...env.status(),
        interrupted,
        error,
        after: fingerprint(env.observation()),
      });
      if (error) throw new Error(error);
    });
    return {
      ...job,
      output: path,
      sha256: await hashRecordFile(path),
      ...env.status(),
      counts,
      work,
      elapsedMs: performance.now() - started,
      ...(learned ? { model: learned.model.ready, inference: learned.model.totals } : {}),
      ...(opponent
        ? { opponentModel: opponent.model.ready, opponentInference: opponent.model.totals }
        : {}),
    };
  } finally {
    learned?.model.close();
    opponent?.model.close();
  }
}

if (process.argv.includes('--worker')) {
  process.on('disconnect', () => process.exit(0));
  process.once('message', async (job: Job) => {
    try {
      process.send!({ result: await run(job) });
    } catch (error) {
      process.send!({ error: error instanceof Error ? error.stack : String(error) });
    }
  });
} else if (process.argv[2] === '--job') {
  // 连续调度器每次只启动一个独立任务；临时目录由宿主独占管理，完成回执最后发布。
  const job = JSON.parse(readFileSync(process.argv[3], 'utf8')) as Job;
  assert.ok(Number.isSafeInteger(job.seed) && job.seed > 0 && job.seed <= 0xffffffff);
  assert.ok(job.primary === 1 || job.primary === 2);
  assert.ok(job.kind === 'evaluation' || job.kind === 'selfplay');
  for (const n of [job.maxCommands ?? 1800, job.maxPlies ?? 120])
    assert.ok(Number.isSafeInteger(n) && n > 0);
  const result = await run(job);
  writeFileSync(resolve(job.output, 'result.json'), JSON.stringify(result, null, 2), {
    flag: 'wx',
  });
} else {
  const output = process.argv[2];
  assert.ok(output);
  mkdirSync(output, { recursive: true });
  const gate = JSON.parse(readFileSync(process.argv[3], 'utf8'));
  assert.equal(gate.fixtureOptimal, 60);
  assert.equal(gate.supportedNaturalWins, 10);
  const jobs: Job[] = [0, 1, 2, 3].map((game) => ({
    game,
    seed: 2026092801 + Math.floor(game / 2),
    primary: game % 2 ? 2 : 1,
    kind: 'evaluation',
    output,
  }));
  jobs.push(
    ...[4, 5].map(
      (game): Job => ({ game, seed: 2026092799 + game, primary: 1, kind: 'selfplay', output }),
    ),
  );
  const write = (name: string, value: unknown) =>
    writeFileSync(resolve(output, name), JSON.stringify(value, null, 2), { flag: 'wx' });
  const scriptHashes = Object.fromEntries(
    [
      'scripts/training/search/puct.ts',
      'scripts/training/search/bootstrap/policy.ts',
      'scripts/training/search/bootstrap/games.ts',
    ].map((f) => [f, digest(f)]),
  );
  const protocol = {
    format: 'haojie-bootstrap-readiness-v1',
    jobs,
    scriptHashes,
    sourceSha256: encodingSourceHash(),
    workers: 4,
    limits: { maxCommands: 1800, maxPlies: 120 },
    teacherWorkRequest: 40,
    actualTeacherWorkSafetyCap: 256,
    gateSha256: digest(process.argv[3]),
    minimum: {
      selfplayTerminalGames: 2,
      illegalOrInterrupted: 0,
      evaluationTerminalGames: 2,
      replayAll: true,
      nonzeroSearchTargets: true,
    },
    note: 'new held-out smoke seeds; not statistically powered strength evaluation; teacher-assisted search',
  };
  write('protocol.json', protocol);
  const results: Awaited<ReturnType<typeof run>>[] = Array(jobs.length);
  const children = new Set<ReturnType<typeof spawn>>();
  let next = 0;
  try {
    const settled = await Promise.allSettled(
      Array.from({ length: 4 }, async () => {
        while (next < jobs.length) {
          const index = next++;
          const child = spawn(
            process.execPath,
            ['--import', 'tsx', fileURLToPath(import.meta.url), '--worker'],
            { windowsHide: true, stdio: ['ignore', 'ignore', 'inherit', 'ipc'] },
          );
          children.add(child);
          try {
            const response = new Promise<any>((accept, reject) => {
              child.on('error', reject);
              child.on('exit', (code) => reject(new Error(`对弈子进程提前退出：${code}`)));
              child.on('message', (message: any) => {
                if (message.progress) {
                  console.log(JSON.stringify(message));
                  return;
                }
                if (message.error) reject(new Error(message.error));
                else accept(message.result);
              });
            });
            child.send!(jobs[index]);
            results[index] = await response;
            write(`result-${index}.json`, results[index]);
            console.log(JSON.stringify({ complete: results[index] }));
          } finally {
            child.kill();
            children.delete(child);
          }
        }
      }),
    );
    const failures = settled.filter((r) => r.status === 'rejected');
    if (failures.length) {
      write(
        'failure.json',
        failures.map((r) => String(r.reason)),
      );
      throw new Error(`对弈失败：${failures.map((r) => String(r.reason)).join('\n')}`);
    }
  } finally {
    for (const child of children) child.kill();
  }
  for (const [file, hash] of Object.entries(scriptHashes)) assert.equal(digest(file), hash);
  assert.equal(encodingSourceHash(), protocol.sourceSha256);
  let replayed = 0,
    policyTargets = 0,
    fallbackTargets = 0;
  for (const result of results) {
    assert.equal(await hashRecordFile(result.output), result.sha256);
    for await (const row of readTrainingRecords(result.output)) {
      if (row.type === 'outcome') assert.ok(!row.interrupted);
      if (row.type !== 'sample') continue;
      replayed++;
      if (row.policyMode === 'search') {
        const p = row.searchPolicy;
        assert.ok(p.length > 0 && p.length <= 8);
        assert.equal(
          p.reduce((n: number, e: any) => n + e.visits, 0),
          16,
        );
        assert.ok(Math.abs(p.reduce((n: number, e: any) => n + e.probability, 0) - 1) < 1e-9);
        assert.ok(p.every((e: any) => e.visits > 0 && e.probability === e.visits / 16));
        policyTargets++;
      } else {
        assert.equal(row.searchPolicy, null);
        fallbackTargets++;
      }
    }
  }
  const summary = {
    complete: true,
    protocolSha256: digest(resolve(output, 'protocol.json')),
    results,
    replayed,
    policyTargets,
    fallbackTargets,
    selfplayTerminal: results.filter((r) => r.kind === 'selfplay' && r.terminated).length,
    evaluationTerminal: results.filter((r) => r.kind === 'evaluation' && r.terminated).length,
    truncated: results.filter((r) => r.truncated).length,
    interrupted: 0,
  };
  write('summary.json', summary);
  console.log(JSON.stringify(summary));
  assert.equal(summary.selfplayTerminal, protocol.minimum.selfplayTerminalGames);
  assert.ok(
    summary.evaluationTerminal >= protocol.minimum.evaluationTerminalGames && policyTargets > 0,
  );
}
