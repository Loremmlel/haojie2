import assert from 'node:assert/strict';
import { setImmediate } from 'node:timers/promises';
import { fingerprint } from '../../../src/ai/observation';
import { trainingPosition } from '../../../src/ai/training/queries';
import { allPieces, hasTrait } from '../../../src/engine/core/traits';
import { RuleError } from '../../../src/engine/core/state';
import { actorCommandError } from '../../../src/engine/online/authority';
import type { Command, Player } from '../../../src/engine/types';
import { TrainingEnvironment } from '../../../src/match/training';
import type { TrainingOptions, TrainingStatus } from '../../../src/match/training';
import type { Observation } from '../../../src/ai/types';
import { recordHeader } from '../records/replay';
import { TinyPolicy, randomStream } from './policy';
import { sampleCommand, sampleChoices, emptyMetrics, SamplingBudgetError } from './sample';

export interface Options {
  seconds: number;
  workers: number;
  worker: number;
  rules: 'classic' | 'shrine';
  policy: 'tiny' | 'uniform';
  seed: number;
  maxCommands: number;
  maxPlies: number;
  games?: number;
}

/** 选招与编码共用原采样循环；后端只负责权威规则执行和脱敏观察。 */
export interface SamplingEnvironment {
  limits(): { maxCommands: number; maxPlies: number };
  status(): TrainingStatus;
  observation(viewer?: Player): Observation | Promise<Observation>;
  view?(viewer?: Player): Observation | Promise<Observation>;
  step(actor: Player, input: unknown): TrainingStatus | Promise<TrainingStatus>;
  stepTyped?(actor: Player, command: Command): TrainingStatus | Promise<TrainingStatus>;
}

/**
 * 单进程真实自然开局；全部成功命令交给调用者，未知终局不填收益，异常不吞掉。
 * 默认输出可重放训练记录；commands 只供原生性能对照，显式改用诊断格式且不计算指纹。
 * 诊断命令必须经过共享引擎重放与正式记录读取器审核后，才可进入训练数据。
 */
export async function sampleWorker(
  options: Options,
  emit: (row: unknown) => Promise<void>,
  makeEnvironment: (
    options: TrainingOptions,
  ) => SamplingEnvironment | Promise<SamplingEnvironment> = (options) =>
    new TrainingEnvironment(options),
  recordMode: 'replayable' | 'commands' = 'replayable',
) {
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
    for (
      let localGame = 0;
      localGame < (options.games ?? Infinity) && !stopped && performance.now() < deadline;
      localGame++
    ) {
      const game = localGame * options.workers + options.worker;
      const seed = options.seed + game;
      const samplerSeed = (0x6ab921d3 + Math.imul(game + 1, 2654435761)) >>> 0;
      const random = randomStream(samplerSeed);
      const env = await makeEnvironment({
        seed,
        rules: options.rules,
        ...{
          maxCommands: options.maxCommands,
          maxPlies: options.maxPlies,
        },
      });
      const metrics = emptyMetrics();
      const observationFor = (viewer?: Player) =>
        env.view ? env.view(viewer) : env.observation(viewer);
      const gameStart = performance.now();
      const commandTypes: Record<string, number> = {};
      await emit({
        type: 'game',
        ...(recordMode === 'replayable'
          ? recordHeader(env)
          : { format: 'haojie-sampling-diagnostic-v1', limits: env.limits() }),
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
        let observation = await observationFor(actor);
        if (observation.phase === 'shrine-draft' && observation.shrineDraft?.committed[actor]) {
          actor = (3 - actor) as Player;
          observation = await observationFor(actor);
        }
        metrics.observationMs += performance.now() - start;
        try {
          const index = env.status().commands;
          const select = async (view: Observation, player: Player, optional = false) => {
            const choices = sampleChoices(view, player, policy, random, metrics, optional);
            let next = choices.next();
            while (!next.done) {
              const candidate = next.value;
              assert.equal(
                actorCommandError({ ...view, events: [], log: [] }, player, candidate.command),
                null,
              );
              const stepStart = performance.now();
              let accepted = true;
              try {
                if (env.stepTyped) await env.stepTyped(player, candidate.command);
                else await env.step(player, candidate.command);
              } catch (error) {
                if (!(error instanceof RuleError) || candidate.status !== 'uncertain') throw error;
                accepted = false;
                await emit({
                  type: 'rejected',
                  game,
                  index,
                  actor: player,
                  ...candidate,
                  before: recordMode === 'replayable' ? fingerprint(view) : undefined,
                  error: error.message,
                });
              }
              metrics.stepMs += performance.now() - stepStart;
              next = choices.next(accepted);
            }
            return next.value;
          };
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
            const interruptObservation = await observationFor(other);
            metrics.observationMs += performance.now() - start;
            selected = await select(interruptObservation, other, true);
            if (selected.command) {
              actor = other;
              observation = interruptObservation;
              metrics.offTurnCommands++;
              offerInterrupt = false;
            } else if (selected.passed) metrics.offTurnPasses++;
            else throw new Error(selected.error);
          }
          if (!selected?.command) {
            selected = await select(observation, actor);
            offerInterrupt = true;
          }
          assert.ok(selected.command, selected.error ?? '常规决策不能pass');
          start = performance.now();
          const before = recordMode === 'replayable' ? fingerprint(observation) : undefined;
          metrics.recordMs += performance.now() - start;
          commandTypes[selected.command.type] = (commandTypes[selected.command.type] ?? 0) + 1;
          start = performance.now();
          await emit({
            type: 'decision',
            game,
            index,
            actor,
            command: selected.command,
            before,
            after: recordMode === 'replayable' ? fingerprint(await env.observation()) : undefined,
            source: options.policy,
          });
          metrics.recordMs += performance.now() - start;
        } catch (reason) {
          error =
            reason instanceof SamplingBudgetError
              ? null
              : reason instanceof Error
                ? (reason.stack ?? reason.message)
                : String(reason);
          interrupted =
            reason instanceof SamplingBudgetError ? 'decode-budget' : 'implementation-error';
          await emit({
            type: error ? 'error' : 'pause',
            game,
            index: env.status().commands,
            actor,
            before:
              recordMode === 'replayable' ? fingerprint(await env.observation(actor)) : undefined,
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
        after: recordMode === 'replayable' ? fingerprint(await env.observation()) : undefined,
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
