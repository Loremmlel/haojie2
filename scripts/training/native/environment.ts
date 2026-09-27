import assert from 'node:assert/strict';
import { nativeClient } from './client';
import { actorCommandError, parseCommand } from '../../../src/engine/online/authority';
import { HAOJIE_RULESET } from '../../../src/engine/online/player-view';
import { decisionOwner } from '../../../src/ai/observation';
import type { Observation } from '../../../src/ai/types';
import type { Player } from '../../../src/engine/types';
import type { TrainingOptions, TrainingStatus } from '../../../src/match/training';
import type { SamplingEnvironment } from '../economics/run';

/** 实验训练后端只缓存公开观察；权威 RNG 留在 Rust。非法操作不更新计数或观察。 */
export async function nativeEnvironment(
  client: Awaited<ReturnType<typeof nativeClient>>,
  options: TrainingOptions,
): Promise<SamplingEnvironment> {
  const limits = {
    maxCommands: options.maxCommands ?? 10000,
    maxPlies: options.maxPlies ?? 500,
  };
  const created = await client.request({
    op: 'create',
    seed: options.seed ?? 20260907,
    rules: options.rules ?? 'classic',
    clearHistory: true,
  });
  let revision: number = created.revision;
  let observation: Observation = created.observation;
  const initialPly = observation.ply;
  let commands = 0;
  const status = (): TrainingStatus => {
    const terminated = observation.winner !== undefined;
    const truncation = terminated
      ? null
      : commands >= limits.maxCommands
        ? 'commands'
        : observation.ply - initialPly >= limits.maxPlies
          ? 'plies'
          : null;
    return {
      ruleset: HAOJIE_RULESET,
      commands,
      ply: observation.ply,
      phase: observation.phase,
      toPlay: terminated || truncation ? null : decisionOwner(observation),
      terminated,
      truncated: truncation !== null,
      truncation,
      winner: observation.winner ?? null,
      returns: terminated
        ? {
            1: observation.winner === 'draw' ? 0 : observation.winner === 1 ? 1 : -1,
            2: observation.winner === 'draw' ? 0 : observation.winner === 2 ? 1 : -1,
          }
        : null,
    };
  };
  return {
    limits: () => limits,
    status,
    async observation(viewer = decisionOwner(observation)) {
      assert.ok(viewer === 1 || viewer === 2);
      // 暗选时不能复用另一方的私有选择；其他阶段仍返回独立的公开对象。
      return viewer === decisionOwner(observation)
        ? structuredClone(observation)
        : await client.request({ op: 'observe', viewer });
    },
    async step(actor, input) {
      assert.ok(
        !status().terminated && !status().truncated,
        '训练对局已终止或截断，请显式 reset。',
      );
      const command = parseCommand(input);
      const error = actorCommandError({ ...observation, events: [], log: [] }, actor, command);
      assert.equal(error, null);
      const reply = await client.request({
        op: 'step',
        revision,
        commands: [command.type === 'choose-shrine' ? { ...command, player: actor } : command],
        clearHistory: true,
        observe: true,
      });
      assert.equal(reply.results[0].status, 'available', JSON.stringify(reply.results));
      revision = reply.revision;
      observation = reply.observation;
      commands++;
      return status();
    },
  };
}
