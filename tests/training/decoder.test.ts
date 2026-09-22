import test from 'node:test';
import assert from 'node:assert/strict';
import { applyCommand, type Command, type GameState, type Player } from '../../src/engine';
import { observe, decisionOwner } from '../../src/ai/observation';
import { TrainingActionTree } from '../../src/ai/training/action-tree';
import { decodeCommand, type PolicyEvaluator } from '../../src/ai/training/decoder';
import { COMMANDS } from '../../src/ai/training/encoding/schema';
import { TrainingEnvironment } from '../../src/match/training';
import { trainingGeometry } from '../../src/ai/training/queries';
import { add, card, fixture } from '../helpers';

const prefer =
  (...commands: Command['type'][]): PolicyEvaluator =>
  async (input) => ({
    logits: input.candidates.map((row) => {
      const rank = commands.findIndex((type) => row[COMMANDS.indexOf(type)] === 1);
      return rank < 0 ? -100 : 10 - rank;
    }),
    value: 0,
  });

async function decodeExpected(state: GameState, actor: Player, command: Command) {
  const observation = observe(state, actor),
    unchanged = structuredClone(observation);
  const trace = new TrainingActionTree(observation, actor).trace(command);
  const choices = trace.filter(({ node }) => node.choices.length > 1);
  let calls = 0;
  const result = await decodeCommand(observation, actor, async (input) => {
    assert.equal(input.candidates.length, choices[calls].node.choices.length);
    const selected = choices[calls++].selected;
    return { logits: input.candidates.map((_, i) => (i === selected ? 10 : -10)), value: 0 };
  });
  assert.equal(result.status, 'command');
  assert.equal(calls, choices.length);
  assert.deepEqual(applyCommand(state, result.command!), applyCommand(state, command));
  assert.deepEqual(observation, unchanged);
  return result;
}

test('网络选择完整路径和三材料合成，只有共享引擎提交才改变局面', async () => {
  const s = fixture();
  s.phase = 'synthesis';
  const ids = [2, 3, 4, 6].map((x) => add(s, 'u21', 1, x, 4).id);
  const synthesis: Command = { type: 'synthesize', recipeId: 'sage', materialIds: ids.slice(1) };
  Object.assign(synthesis, trainingGeometry(observe(s), synthesis).points![0]);
  await decodeExpected(s, 1, synthesis);

  const h = fixture(),
    carrier = add(h, 26, 1, 2, 5);
  add(h, 1, 2, 4, 6);
  const equipped = applyCommand(h, { type: 'equip', cardId: card(h, 'u28'), targetId: carrier.id });
  const result = await decodeExpected(equipped, 1, {
    type: 'attack',
    unitId: carrier.id,
    path: [
      { x: 2, y: 5 },
      { x: 3, y: 5 },
      { x: 3, y: 6 },
      { x: 4, y: 6 },
    ],
  });
  assert.equal(result.path.at(-1)!.key, 'commit-path');
});

test('最高分空分支回溯到原排序的合法end；固定预算不足显式暂停', async () => {
  const s = fixture();
  add(s, 1, 1, 2, 2);
  const o = observe(s),
    evaluator = prefer('clock', 'end');
  const result = await decodeCommand(o, 1, evaluator);
  assert.deepEqual(result.command, { type: 'end' });
  assert.ok(result.stats.emptyBranches > 0);
  assert.ok(result.stats.backtracks > 0);
  const limited = await decodeCommand(o, 1, evaluator, { maxNodes: 1 });
  assert.equal(limited.reason, 'node-budget');
  assert.equal(limited.command, undefined);
  assert.equal(limited.path.length, 1);
  const noInference = await decodeCommand(o, 1, evaluator, { maxEvaluations: 0 });
  assert.equal(noInference.reason, 'inference-budget');
  assert.equal(noInference.command, undefined);
});

test('强制节点不推理；反应属于对方时按反应方处理；随机召唤停在公开边界', async () => {
  const env = new TrainingEnvironment({ seed: 19 });
  const summon = await decodeCommand(env.observation(), 1, prefer('summon'));
  assert.equal(summon.command!.type, 'summon');
  assert.equal(summon.leafStatus, 'uncertain');
  assert.equal(summon.stats.evaluations, 0);
  assert.equal(env.status().commands, 0);
  env.step(1, summon.command);

  const s = fixture(),
    attacker = add(s, 26, 1, 3, 4),
    victim = add(s, 2, 2, 3, 5);
  victim.hp = 1;
  const pending = applyCommand(s, { type: 'attack', unitId: attacker.id, targetId: victim.id });
  assert.equal(pending.active, 1);
  assert.equal(decisionOwner(pending), 2);
  await decodeExpected(pending, 2, { type: 'react' });
});

test('取消后的迟到评分不能产生命令；非有限/错位评分不回退教师', async () => {
  const s = fixture();
  add(s, 1, 1, 3, 4);
  const o = observe(s),
    cancellation = new AbortController();
  let release!: () => void;
  const barrier = new Promise<void>((resolve) => {
    release = resolve;
  });
  const pending = decodeCommand(
    o,
    1,
    async (input) => {
      await barrier;
      return { logits: input.candidates.map(() => 0), value: 0 };
    },
    { signal: cancellation.signal },
  );
  cancellation.abort();
  release();
  const cancelled = await pending;
  assert.equal(cancelled.reason, 'cancelled');
  assert.equal(cancelled.command, undefined);
  for (const evaluator of [
    async () => ({ logits: [], value: 0 }),
    async (input: Parameters<PolicyEvaluator>[0]) => ({
      logits: input.candidates.map(() => NaN),
      value: 0,
    }),
  ])
    assert.equal((await decodeCommand(o, 1, evaluator)).reason, 'invalid-output');
  const failed = await decodeCommand(o, 1, async () => {
    throw new Error('管道断开');
  });
  assert.equal(failed.reason, 'inference-error');
  assert.equal(failed.error, '管道断开');
  assert.equal(failed.command, undefined);
});
