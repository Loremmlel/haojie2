import { createWriteStream, mkdirSync } from 'node:fs';
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
import { decisionOwner } from '../../src/ai/observation';
import type { Difficulty } from '../../src/ai/types';

export interface SelfPlayOptions extends TrainingOptions {
  games: number;
  difficulty?: Difficulty;
  simulations?: number;
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
  const results: (TrainingStatus & {
    game: number;
    elapsedMs: number;
    simulations: number;
    maxUnits: number;
    timing: { observationMs: number; teacherMs: number; stepMs: number };
  })[] = [];
  for (let game = 0; game < options.games; game++) {
    if (options.signal?.aborted) break;
    const seed = (((options.seed ?? 20260907) - 1 + game) % 0xffffffff) + 1;
    const env = new TrainingEnvironment({ ...options, seed });
    const teacher = new TrainingTeacher(options.difficulty, options.simulations);
    let simulations = 0;
    let maxUnits = 0;
    const timing = { observationMs: 0, teacherMs: 0, stepMs: 0 };
    const start = performance.now();
    // 宿主复现元数据与样本分开；seed 不得成为网络输入。
    await emit?.({
      type: 'game',
      game,
      ruleset: env.status().ruleset,
      seed,
      rules: options.rules ?? 'classic',
      difficulty: options.difficulty ?? 'easy',
      budget: options.simulations ?? 'production-work',
    });
    let status = env.status();
    while (!status.terminated && !status.truncated) {
      if (options.signal?.aborted) return { cancelled: true, results };
      const observeStart = performance.now();
      const observation = env.observation();
      timing.observationMs += performance.now() - observeStart;
      maxUnits = Math.max(maxUnits, observation.units.length);
      const actor = decisionOwner(observation);
      const teacherStart = performance.now();
      const decision = teacher.next(observation);
      timing.teacherMs += performance.now() - teacherStart;
      await setImmediate();
      if (options.signal?.aborted) return { cancelled: true, results };
      const stepStart = performance.now();
      status = env.step(actor, decision.command);
      timing.stepMs += performance.now() - stepStart;
      simulations += decision.stats.simulations;
      await emit?.({
        type: 'sample',
        game,
        index: status.commands - 1,
        actor,
        observation,
        command: decision.command,
        teacherStats: decision.stats,
      });
    }
    const result = {
      game,
      ...status,
      elapsedMs: performance.now() - start,
      simulations,
      maxUnits,
      timing,
    };
    results.push(result);
    await emit?.({ type: 'outcome', ...result });
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
  if (output) mkdirSync(dirname(output), { recursive: true });
  const stream = output ? createWriteStream(output, { flags: 'wx' }) : undefined;
  if (stream) await once(stream, 'open');
  const cancellation = new AbortController();
  process.once('SIGINT', () => cancellation.abort());
  try {
    const report = await runSelfPlay(
      {
        games: Number(value('games', '2')),
        seed: Number(value('seed', '20260922')),
        rules: value('rules', 'classic') as 'classic' | 'shrine',
        difficulty: value('difficulty', 'easy') as Difficulty,
        maxPlies: Number(value('plies', '100')),
        maxCommands: Number(value('commands', '3000')),
        ...(args.includes('--nodes') ? { simulations: Number(value('nodes', '100')) } : {}),
        signal: cancellation.signal,
      },
      stream
        ? async (row) => {
            if (!stream.write(JSON.stringify(row) + '\n')) await once(stream, 'drain');
          }
        : undefined,
    );
    console.log(JSON.stringify(report, null, 2));
    if (report.cancelled) process.exitCode = 130;
  } finally {
    if (stream) {
      stream.end();
      await once(stream, 'finish');
    }
  }
}
