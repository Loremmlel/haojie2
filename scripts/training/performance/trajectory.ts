import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { decisionOwner, fingerprint, observe } from '../../../src/ai/observation';
import { trainingPosition } from '../../../src/ai/training/queries';
import { allPieces, hasTrait } from '../../../src/engine/core/traits';
import { RULESET_ID } from '../../../src/engine/catalog';
import { HAOJIE_RULESET } from '../../../src/engine/online/player-view';
import type { Player } from '../../../src/engine/types';
import { withRecordOutput } from '../records/io';
import { TRAINING_RECORD_FORMAT } from '../records/replay';
import type * as API from './api';

const { values } = parseArgs({
  options: {
    baseline: { type: 'string' },
    current: { type: 'string' },
    output: { type: 'string' },
    commands: { type: 'string', default: '1000' },
  },
});
assert.ok(values.baseline && values.current && values.output);
const baseline: typeof API = await import(pathToFileURL(resolve(values.baseline)).href);
const current: typeof API = await import(pathToFileURL(resolve(values.current)).href);
const maxCommands = Number(values.commands);
assert.ok(Number.isSafeInteger(maxCommands) && maxCommands > 0 && maxCommands <= 10000);
const output = resolve(values.output);

/**
 * 固定命令数的端到端对照；模型、独立采样流、自然开局和回合外巨大化调度完全相同。
 * 共享未修改的公开观察与记录接口，候选和结算分别使用冻结的整份引擎依赖图。
 * 只记录真实命令，命令上限不赋胜负；异常直接失败，不能用更短轨迹冒充加速。
 */
async function run(api: typeof API, rules: 'classic' | 'shrine', label: string) {
  const seed = 423470001;
  const samplerSeed = (0x6ab921d3 + 2654435761) >>> 0;
  const policy = new api.TinyPolicy(73129);
  const random = api.randomStream(samplerSeed);
  const metrics = api.emptyMetrics();
  let state = api.createGame(seed, rules);
  state.events = [];
  state.log = [];
  let offerInterrupt = true;
  let commands = 0;
  const started = performance.now();
  let outcome: any;
  await withRecordOutput(join(output, `${rules}-${label}.jsonl.gz`), async (emit) => {
    await emit({
      type: 'game',
      format: TRAINING_RECORD_FORMAT,
      ruleset: HAOJIE_RULESET,
      recordRuleset: RULESET_ID,
      limits: { maxCommands, maxPlies: 500 },
      game: 0,
      source: 'neural',
      seed,
      rules,
      policySeed: 73129,
      samplerSeed,
    });
    while (commands < maxCommands && !state.winner && state.ply - 1 < 500) {
      let actor = decisionOwner(state);
      let t = performance.now();
      let observation = observe(state, actor);
      if (observation.phase === 'shrine-draft' && observation.shrineDraft?.committed[actor]) {
        actor = (3 - actor) as Player;
        observation = observe(state, actor);
      }
      metrics.observationMs += performance.now() - t;
      let selected: ReturnType<typeof api.sampleCommand> | undefined;
      const other = (3 - actor) as Player;
      t = performance.now();
      const interrupt =
        offerInterrupt &&
        !observation.pending.length &&
        observation.phase !== 'shrine-draft' &&
        allPieces(trainingPosition(observation)).some(
          (u) => u.owner === other && hasTrait(u, 'u7'),
        );
      metrics.treeMs += performance.now() - t;
      if (interrupt) {
        t = performance.now();
        const otherObservation = observe(state, other);
        metrics.observationMs += performance.now() - t;
        selected = api.sampleCommand(otherObservation, other, policy, random, metrics, true);
        if (selected.command) {
          actor = other;
          observation = otherObservation;
          offerInterrupt = false;
          metrics.offTurnCommands++;
        } else {
          assert.ok(selected.passed, selected.error);
          metrics.offTurnPasses++;
        }
      }
      if (!selected?.command) {
        selected = api.sampleCommand(observation, actor, policy, random, metrics);
        offerInterrupt = true;
      }
      assert.ok(selected.command, selected.error);
      t = performance.now();
      state = api.applyPlayerCommand(state, actor, selected.command);
      metrics.stepMs += performance.now() - t;
      state.log = [];
      state.events = [];
      t = performance.now();
      await emit({
        type: 'decision',
        game: 0,
        index: commands++,
        actor,
        command: selected.command,
        before: fingerprint(observation),
        after: fingerprint(observe(state)),
      });
      metrics.recordMs += performance.now() - t;
    }
    const terminated = state.winner !== undefined;
    const truncation = terminated ? null : commands >= maxCommands ? 'commands' : 'plies';
    outcome = {
      type: 'outcome',
      game: 0,
      ruleset: HAOJIE_RULESET,
      commands,
      ply: state.ply,
      phase: state.phase,
      toPlay: null,
      terminated,
      truncated: !terminated,
      truncation,
      winner: state.winner ?? null,
      returns: terminated
        ? {
            1: state.winner === 'draw' ? 0 : state.winner === 1 ? 1 : -1,
            2: state.winner === 'draw' ? 0 : state.winner === 2 ? 1 : -1,
          }
        : null,
      after: fingerprint(observe(state)),
    };
    await emit(outcome);
  });
  const elapsedMs = performance.now() - started;
  return { elapsedMs, commands, metrics, outcome, state };
}

const results = [];
for (const rules of ['classic', 'shrine'] as const) {
  // 两种模式交换先后顺序；此短程实验只测性能，不估计终局率或棋力。
  let a: Awaited<ReturnType<typeof run>>, b: Awaited<ReturnType<typeof run>>;
  if (rules === 'classic') {
    a = await run(baseline, rules, 'before');
    b = await run(current, rules, 'after');
  } else {
    b = await run(current, rules, 'after');
    a = await run(baseline, rules, 'before');
  }
  assert.deepEqual(a.state, b.state, '完整终态不同');
  assert.deepEqual(a.outcome, b.outcome, '实际停止边界不同');
  const lines = async (label: string) => {
    const rows = [];
    for await (const row of current.readTrainingRecords(
      join(output, `${rules}-${label}.jsonl.gz`),
    )) {
      const { observation: _observation, ...record } = row;
      rows.push(record);
    }
    return rows;
  };
  assert.deepEqual(await lines('before'), await lines('after'), '命令轨迹不同');
  const { state: _a, ...before } = a;
  const { state: _b, ...after } = b;
  const result = { rules, before, after, speedup: a.elapsedMs / b.elapsedMs, recordsEqual: true };
  results.push(result);
  console.log(JSON.stringify({ rules, commands: a.commands, speedup: result.speedup }));
}
writeFileSync(join(output, 'trajectory-summary.json'), JSON.stringify(results, null, 2), {
  flag: 'wx',
});
