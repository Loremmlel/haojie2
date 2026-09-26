import { mkdirSync, writeFileSync, existsSync, readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { setImmediate } from 'node:timers/promises';
import { cpus, totalmem } from 'node:os';
import { execFileSync } from 'node:child_process';
import { TrainingEnvironment, type TrainingOptions } from '../../src/match/training';
import { TrainingTeacher } from '../../src/ai/training/teacher';
import {
  decodeCommand,
  DEFAULT_DECODE_NODES,
  type DecodeOptions,
  type PolicyEvaluator,
} from '../../src/ai/training/decoder';
import { decisionOwner, fingerprint } from '../../src/ai/observation';
import { ENCODING_SCHEMA } from '../../src/ai/training/encoding/schema';
import { encodeDecision } from '../../src/ai/training/encoding/decision';
import { TrainingActionTree } from '../../src/ai/training/action-tree';
import type { Difficulty } from '../../src/ai/types';
import { PythonPolicy } from './python-policy';
import { encodingSourceHash } from './encode';
import { summarizeMatches } from './neural-report';
import { recordHeader } from './records/replay';
import { withRecordOutput } from './records/io';

export interface NeuralMatchOptions extends TrainingOptions, DecodeOptions {
  games: number;
  difficulty?: Difficulty;
  simulations?: number;
  maxTurnCommands?: number;
}

/**
 * 模型与原手工教师对战；相邻两局使用同一种子并交换模型席位，双方共用权威引擎。
 * 不修补网络动作或按血量判胜；暂停/拒绝/重复公开局面单独记为中断，无终局标签。
 * emit逐条记录并施加背压，取消在提交前重新检查；宿主种子仅在game元数据中出现。
 */
export async function runNeuralMatches(
  options: NeuralMatchOptions,
  evaluate: PolicyEvaluator,
  emit: (row: any) => Promise<void>,
  decode: typeof decodeCommand = decodeCommand,
) {
  if (!Number.isSafeInteger(options.games) || options.games < 1)
    throw new Error('games必须是正整数');
  const maxTurnCommands = options.maxTurnCommands ?? 200;
  if (!Number.isSafeInteger(maxTurnCommands) || maxTurnCommands < 1)
    throw new Error('回合命令上限无效');
  const outcomes = [];
  for (let game = 0; game < options.games; game++) {
    if (options.signal?.aborted) break;
    const seed = (((options.seed ?? 2026092301) - 1 + Math.floor(game / 2)) % 0xffffffff) + 1;
    const networkPlayer = game % 2 === 0 ? 1 : 2;
    const env = new TrainingEnvironment({ ...options, seed });
    const teacher = new TrainingTeacher(options.difficulty, options.simulations);
    await emit({
      type: 'game',
      game,
      seed,
      networkPlayer,
      rules: options.rules ?? 'classic',
      ...recordHeader(env),
      source: 'neural',
      teacher: options.difficulty ?? 'easy',
      teacherBudget: options.simulations ?? 'production-work',
      maxNodes: options.maxNodes ?? DEFAULT_DECODE_NODES,
      maxEvaluations: options.maxEvaluations ?? 32,
      maxTurnCommands,
    });
    const started = performance.now();
    let status = env.status(),
      interrupted: string | null = null,
      error: string | null = null;
    let turn = status.ply,
      turnCommands = 0;
    const visits = new Map<string, number>();
    while (!status.terminated && !status.truncated) {
      await setImmediate();
      if (options.signal?.aborted) {
        interrupted = 'cancelled';
        break;
      }
      if (status.ply !== turn) {
        turn = status.ply;
        turnCommands = 0;
        visits.clear();
      }
      if (turnCommands >= maxTurnCommands) {
        interrupted = 'turn-command-limit';
        break;
      }
      const decisionStart = performance.now();
      const observation = env.observation();
      const actor = decisionOwner(observation),
        network = actor === networkPlayer;
      const before = fingerprint(observation),
        key = `${actor}:${before}`;
      const count = (visits.get(key) ?? 0) + 1;
      visits.set(key, count);
      if (count > 3) {
        interrupted = 'repeated-observation';
        break;
      }
      const observationMs = performance.now() - decisionStart;
      let decoded, command;
      try {
        if (network) {
          decoded = await decode(observation, actor, evaluate, options);
          if (decoded.status !== 'command') {
            interrupted = decoded.reason!;
            await emit({
              type: 'pause',
              game,
              index: status.commands,
              actor,
              before,
              decoded,
              decisionMs: performance.now() - decisionStart,
            });
            break;
          }
          command = decoded.command!;
        } else command = teacher.next(observation).command;
      } catch (e) {
        interrupted = network ? 'decoder-error' : 'teacher-error';
        error = e instanceof Error ? e.message : String(e);
        await emit({
          type: 'error',
          game,
          index: status.commands,
          actor,
          before,
          error,
        });
        break;
      }
      // 让取消信号有机会在同步教师搜索结束后送达，不能提交旧结果。
      await setImmediate();
      if (options.signal?.aborted) {
        interrupted = 'cancelled';
        break;
      }
      const decisionMs = performance.now() - decisionStart;
      const stepStart = performance.now();
      try {
        status = env.step(actor, command);
      } catch (e) {
        interrupted = 'rejected-command';
        error = e instanceof Error ? e.message : String(e);
        await emit({
          type: 'rejected',
          game,
          index: status.commands,
          actor,
          before,
          command,
          decoded,
          error,
        });
        break;
      }
      const stepMs = performance.now() - stepStart;
      turnCommands++;
      await emit({
        type: 'decision',
        game,
        index: status.commands - 1,
        actor,
        policy: network ? 'network' : 'teacher',
        ply: observation.ply,
        phase: observation.phase,
        reaction: observation.pending.length > 0,
        before,
        after: fingerprint(env.observation()),
        command,
        ...(decoded ? { decoded } : {}),
        timing: {
          observationMs,
          decisionMs,
          stepMs,
          totalMs: decisionMs + stepMs,
        },
      });
    }
    const outcome = {
      type: 'outcome',
      game,
      seed,
      networkPlayer,
      ...status,
      after: fingerprint(env.observation()),
      interrupted,
      error,
      elapsedMs: performance.now() - started,
    };
    outcomes.push(outcome);
    await emit(outcome);
    if (options.signal?.aborted) break;
  }
  return outcomes;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { values } = parseArgs({
    options: {
      checkpoint: { type: 'string' },
      'inference-module': { type: 'string' },
      decoder: { type: 'string', default: 'greedy' },
      'search-simulations': { type: 'string', default: '16' },
      'value-certificate': { type: 'string' },
      output: { type: 'string' },
      python: {
        type: 'string',
        default:
          process.platform === 'win32'
            ? 'training/.venv/Scripts/python.exe'
            : 'training/.venv/bin/python',
      },
      device: { type: 'string', default: 'cpu' },
      precision: { type: 'string', default: 'fp32' },
      threads: { type: 'string', default: '4' },
      games: { type: 'string', default: '2' },
      seed: { type: 'string', default: '2026092301' },
      rules: { type: 'string', default: 'classic' },
      difficulty: { type: 'string', default: 'easy' },
      nodes: { type: 'string', default: '40' },
      plies: { type: 'string', default: '100' },
      commands: { type: 'string', default: '1200' },
      'decode-nodes': { type: 'string', default: String(DEFAULT_DECODE_NODES) },
      evaluations: { type: 'string', default: '32' },
      'turn-commands': { type: 'string', default: '200' },
      'timeout-ms': { type: 'string', default: '60000' },
    },
  });
  if (!values.checkpoint || !values.output) throw new Error('请提供--checkpoint和新的--output目录');
  if (!['greedy', 'beam', 'mcts', 'gumbel', 'value-mcts', 'value-gumbel'].includes(values.decoder!))
    throw new Error('--decoder无效');
  const usesValue = values.decoder!.startsWith('value-');
  const valueCertificate = values['value-certificate']
    ? JSON.parse(readFileSync(values['value-certificate'], 'utf8'))
    : null;
  if (
    usesValue &&
    (!valueCertificate?.passed ||
      !valueCertificate.policy_unchanged ||
      valueCertificate.checkpoint_sha256 !==
        createHash('sha256').update(readFileSync(values.checkpoint)).digest('hex') ||
      !Array.isArray(valueCertificate.covered_phases) ||
      !valueCertificate.covered_phases.includes('play'))
  )
    throw new Error('研究价值搜索须提供与检查点匹配、通过分组校准的证书');
  if (existsSync(values.output)) throw new Error('输出目录已存在；请保留旧实验并使用新目录');
  const numeric = (key: keyof typeof values) => {
    const value = Number(values[key]);
    if (!Number.isSafeInteger(value) || value < 1) throw new Error(`--${key}必须是正整数`);
    return value;
  };
  const cancellation = new AbortController();
  process.once('SIGINT', () => cancellation.abort());
  const options: NeuralMatchOptions = {
    games: numeric('games'),
    seed: numeric('seed'),
    rules: values.rules as 'classic' | 'shrine',
    difficulty: values.difficulty as Difficulty,
    simulations: numeric('nodes'),
    maxPlies: numeric('plies'),
    maxCommands: numeric('commands'),
    maxNodes: numeric('decode-nodes'),
    maxEvaluations: numeric('evaluations'),
    maxTurnCommands: numeric('turn-commands'),
    signal: cancellation.signal,
  };
  mkdirSync(values.output, { recursive: true });
  const tracePath = resolve(values.output, 'games.jsonl.gz');
  let policy: PythonPolicy | undefined;
  const started = new Date().toISOString();
  const sourceSha256 = encodingSourceHash();
  const sources = createHash('sha256').update(sourceSha256);
  for (const path of [
    'scripts/training/neural-match.ts',
    'scripts/training/neural-report.ts',
    'scripts/training/python-policy.ts',
    'training/haojie_training/inference.py',
    'training/haojie_training/data.py',
    'training/haojie_training/model.py',
    'training/haojie_training/runtime.py',
  ])
    sources.update(path).update(readFileSync(path, 'utf8').replaceAll('\r\n', '\n'));
  const experimentSourceSha256 = sources.digest('hex');
  const decoderSources = Object.fromEntries(
    ['beam', 'search'].map((name) => {
      const path = `scripts/training/improvement/${name}.ts`;
      return [path, createHash('sha256').update(readFileSync(path)).digest('hex')];
    }),
  );
  const searchSummary = {
    decisions: 0,
    searched: 0,
    fallbacks: {} as Record<string, number>,
    simulations: 0,
    transitions: 0,
    terminalLeaves: 0,
    unknownLeaves: 0,
    valueCalls: 0,
    changedFromBeam: 0,
  };
  const decode: typeof decodeCommand =
    values.decoder === 'greedy'
      ? decodeCommand
      : values.decoder === 'beam'
        ? (await import('./improvement/beam')).beamDecode
        : (observation, actor, evaluate, limits) =>
            import('./improvement/search').then(({ policySearch }) =>
              policySearch(observation, actor, evaluate, {
                ...limits,
                simulations: numeric('search-simulations'),
                mode: values.decoder!.replace('value-', '') as 'mcts' | 'gumbel',
                ...(usesValue
                  ? {
                      valuePhases: valueCertificate.covered_phases,
                      leafValue: async (leaf, root) => {
                        const side = decisionOwner(leaf);
                        const node = new TrainingActionTree(leaf, side).node();
                        const predicted = await evaluate(
                          encodeDecision(leaf, side, node),
                          limits?.signal,
                        );
                        return predicted.value * (side === root ? 1 : -1);
                      },
                    }
                  : {}),
              }),
            );
  let failure: string | null = null;
  try {
    await withRecordOutput(tracePath, async (emit) => {
      policy = await PythonPolicy.start({
        python: values.python!,
        checkpoint: values.checkpoint!,
        module: values['inference-module'],
        device: values.device!,
        precision: values.precision!,
        threads: numeric('threads'),
        timeoutMs: numeric('timeout-ms'),
        signal: cancellation.signal,
      });
      await runNeuralMatches(
        options,
        policy.evaluate,
        async (row) => {
          await emit(row);
          if (row.type === 'decision' && row.decoded?.search) {
            const search = row.decoded.search;
            searchSummary.decisions++;
            if (search.fallback)
              searchSummary.fallbacks[search.fallback] =
                (searchSummary.fallbacks[search.fallback] ?? 0) + 1;
            else searchSummary.searched++;
            for (const key of [
              'simulations',
              'transitions',
              'terminalLeaves',
              'unknownLeaves',
              'valueCalls',
            ] as const)
              searchSummary[key] += search.stats[key];
            searchSummary.changedFromBeam += Number(search.changedFromBeam);
          }
          if (row.type === 'outcome') console.error(JSON.stringify(row));
        },
        decode,
      );
    });
  } catch (error) {
    failure = error instanceof Error ? error.message : String(error);
  } finally {
    policy?.close();
  }
  const { signal: _, ...recordOptions } = options;
  const report = {
    format: 'haojie-neural-match-v1',
    started,
    options: recordOptions,
    commit: execFileSync('git', ['rev-parse', 'HEAD'], {
      encoding: 'utf8',
    }).trim(),
    sourceSha256,
    experimentSourceSha256,
    decoder: values.decoder,
    decoderSources,
    searchSimulations: numeric('search-simulations'),
    searchSummary,
    valueCertificate: usesValue
      ? {
          path: values['value-certificate'],
          sha256: createHash('sha256')
            .update(readFileSync(values['value-certificate']!))
            .digest('hex'),
        }
      : null,
    inferenceModule: values['inference-module'] ?? 'haojie_training.inference',
    schema: ENCODING_SCHEMA,
    runtime: {
      node: process.version,
      cpu: cpus()[0].model,
      ramBytes: totalmem(),
    },
    model: policy?.ready ?? null,
    startupMs: policy?.startupMs ?? null,
    inference: policy?.totals ?? null,
    cancelled: cancellation.signal.aborted,
    failure,
    ...(await summarizeMatches(tracePath)),
  };
  writeFileSync(resolve(values.output, 'report.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(
    JSON.stringify(
      {
        output: values.output,
        games: report.games,
        terminated: report.terminatedGames,
        truncated: report.truncatedGames,
        interrupted: report.interruptedGames,
        rejected: report.rejectedCommands,
        network: report.network,
        evaluatedNetwork: report.evaluatedNetwork,
        failure,
      },
      null,
      2,
    ),
  );
  if (report.cancelled) process.exitCode = 130;
  else if (failure || report.interruptedGames || report.rejectedCommands) process.exitCode = 1;
}
