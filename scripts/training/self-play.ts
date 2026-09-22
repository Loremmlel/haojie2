import { createWriteStream, mkdirSync, writeFileSync, readFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { cpus } from 'node:os';
import { execFileSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { once } from 'node:events';
import { setImmediate } from 'node:timers/promises';
import {
  TrainingEnvironment,
  type TrainingOptions,
  type TrainingStatus,
} from '../../src/match/training';
import { TrainingTeacher } from '../../src/ai/training/teacher';
import { decisionOwner, fingerprint } from '../../src/ai/observation';
import type { Difficulty } from '../../src/ai/types';
import type { Player } from '../../src/engine/types';
import { encodingSourceHash } from './encode';

export interface TeacherProfile {
  difficulty: Difficulty;
  simulations?: number;
}

export interface SelfPlayOptions extends TrainingOptions {
  games: number;
  difficulty?: Difficulty;
  simulations?: number;
  /** 指定对手后，相邻两局共用种子并交换主教师席位。 */
  opponent?: TeacherProfile;
  signal?: AbortSignal;
}

/**
 * 共用规则和公开观察的热启动对弈；记录策略样本及真实终局，未决局不产生价值标签。
 * 一个命令一个取消边界，已有同步教师搜索不能在调用中途抢占；取消后不执行返回的旧命令。
 * emit 负责批量/流式写出并施加背压，默认不构建trace、不保留整局观察或写磁盘。
 */
export async function runSelfPlay(
  options: SelfPlayOptions,
  emit?: (row: unknown) => Promise<void>,
) {
  if (!Number.isSafeInteger(options.games) || options.games < 1)
    throw new Error('games 必须是正整数。');
  if (
    options.opponent &&
    (options.difficulty ?? 'easy') === options.opponent.difficulty &&
    options.simulations === options.opponent.simulations
  )
    throw new Error('相同教师配置会产生重复轨迹，请省略opponent逐局使用新种子。');
  const results: (TrainingStatus & {
    game: number;
    gameId: string;
    seed: number;
    primaryPlayer: Player;
    interrupted: string | null;
    error: string | null;
    elapsedMs: number;
    simulations: number;
    maxUnits: number;
    timing: { observationMs: number; teacherMs: number; stepMs: number };
  })[] = [];
  for (let game = 0; game < options.games; game++) {
    if (options.signal?.aborted) break;
    const ordinal = options.opponent ? Math.floor(game / 2) : game;
    const seed = (((options.seed ?? 20260907) - 1 + ordinal) % 0xffffffff) + 1;
    const primaryPlayer: Player = options.opponent && game % 2 ? 2 : 1;
    const primary: TeacherProfile = {
      difficulty: options.difficulty ?? 'easy',
      simulations: options.simulations,
    };
    const secondary = options.opponent ?? primary;
    const profiles: Record<Player, TeacherProfile> =
      primaryPlayer === 1 ? { 1: primary, 2: secondary } : { 1: secondary, 2: primary };
    const env = new TrainingEnvironment({ ...options, seed });
    const teachers = {
      1: new TrainingTeacher(profiles[1].difficulty, profiles[1].simulations),
      2: new TrainingTeacher(profiles[2].difficulty, profiles[2].simulations),
    };
    const profileId = (p: TeacherProfile) => `${p.difficulty}-${p.simulations ?? 'production'}`;
    const gameId = `${env.status().ruleset}:${options.rules ?? 'classic'}:${seed}:${profileId(profiles[1])}:${profileId(profiles[2])}`;
    let simulations = 0;
    let maxUnits = 0;
    const timing = { observationMs: 0, teacherMs: 0, stepMs: 0 };
    const start = performance.now();
    // 宿主复现元数据与样本分开；seed 不得成为网络输入。
    await emit?.({
      type: 'game',
      game,
      gameId,
      primaryPlayer,
      teachers: profiles,
      ruleset: env.status().ruleset,
      seed,
      rules: options.rules ?? 'classic',
      difficulty: options.difficulty ?? 'easy',
      budget: options.simulations ?? 'production-work',
    });
    let status = env.status(),
      interrupted: string | null = null,
      error: string | null = null;
    while (!status.terminated && !status.truncated) {
      if (options.signal?.aborted) {
        interrupted = 'cancelled';
        break;
      }
      const observeStart = performance.now();
      const observation = env.observation();
      timing.observationMs += performance.now() - observeStart;
      maxUnits = Math.max(maxUnits, observation.units.length);
      const actor = decisionOwner(observation);
      const teacherStart = performance.now();
      let decision;
      try {
        decision = teachers[actor].next(observation);
      } catch (e) {
        interrupted = 'teacher-error';
        error = e instanceof Error ? e.message : String(e);
        break;
      }
      const teacherMs = performance.now() - teacherStart;
      timing.teacherMs += teacherMs;
      await setImmediate();
      if (options.signal?.aborted) {
        interrupted = 'cancelled';
        break;
      }
      const stepStart = performance.now();
      try {
        status = env.step(actor, decision.command);
      } catch (e) {
        interrupted = 'rejected-command';
        error = e instanceof Error ? e.message : String(e);
        break;
      }
      const stepMs = performance.now() - stepStart;
      timing.stepMs += stepMs;
      simulations += decision.stats.simulations;
      await emit?.({
        type: 'sample',
        game,
        index: status.commands - 1,
        actor,
        observation,
        command: decision.command,
        teacherStats: decision.stats,
        timing: { teacherMs, stepMs },
        after: fingerprint(env.observation()),
      });
    }
    const result = {
      game,
      gameId,
      seed,
      primaryPlayer,
      ...status,
      interrupted,
      error,
      ...(interrupted ? { observation: env.observation() } : {}),
      elapsedMs: performance.now() - start,
      simulations,
      maxUnits,
      timing,
    };
    results.push(result);
    await emit?.({ type: 'outcome', ...result });
    if (options.signal?.aborted) break;
  }
  return { cancelled: options.signal?.aborted ?? false, results };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  const value = (name: string, fallback: string) => {
    const i = args.indexOf(`--${name}`);
    return i < 0 ? fallback : args[i + 1];
  };
  const output = value('output', '');
  const reportPath = value('report', '');
  if (reportPath && existsSync(reportPath)) throw new Error('报告已存在，请保留旧实验。');
  if (output) mkdirSync(dirname(output), { recursive: true });
  const stream = output ? createWriteStream(output, { flags: 'wx' }) : undefined;
  if (stream) await once(stream, 'open');
  const cancellation = new AbortController();
  process.once('SIGINT', () => cancellation.abort());
  const started = new Date().toISOString();
  const commit = execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
  const sourceSha256 = createHash('sha256')
    .update(encodingSourceHash())
    .update(readFileSync(fileURLToPath(import.meta.url), 'utf8').replaceAll('\r\n', '\n'))
    .digest('hex');
  const options: SelfPlayOptions = {
    games: Number(value('games', '2')),
    seed: Number(value('seed', '20260922')),
    rules: value('rules', 'classic') as 'classic' | 'shrine',
    difficulty: value('difficulty', 'easy') as Difficulty,
    maxPlies: Number(value('plies', '100')),
    maxCommands: Number(value('commands', '3000')),
    ...(args.includes('--nodes') ? { simulations: Number(value('nodes', '100')) } : {}),
    ...(args.includes('--opponent')
      ? {
          opponent: {
            difficulty: value('opponent', 'medium') as Difficulty,
            ...(args.includes('--opponent-nodes')
              ? { simulations: Number(value('opponent-nodes', '320')) }
              : {}),
          },
        }
      : {}),
    signal: cancellation.signal,
  };
  try {
    const report = await runSelfPlay(options, async (row) => {
      if (stream && !stream.write(JSON.stringify(row) + '\n')) await once(stream, 'drain');
      if ((row as { type: string }).type === 'outcome') {
        const { observation: _, ...brief } = row as Record<string, unknown>;
        console.error(JSON.stringify(brief));
      }
    });
    if (reportPath) {
      mkdirSync(dirname(reportPath), { recursive: true });
      const { signal: _, ...recordOptions } = options;
      writeFileSync(
        reportPath,
        JSON.stringify(
          {
            format: 'haojie-teacher-run-v1',
            started,
            options: recordOptions,
            sourceSha256,
            commit,
            runtime: { node: process.version, cpu: cpus()[0].model },
            ...report,
          },
          null,
          2,
        ) + '\n',
        { flag: 'wx' },
      );
    }
    console.log(JSON.stringify(report, null, 2));
    if (report.cancelled) process.exitCode = 130;
    else if (report.results.some((r) => r.interrupted)) process.exitCode = 1;
  } finally {
    if (stream) {
      stream.end();
      await once(stream, 'finish');
    }
  }
}
