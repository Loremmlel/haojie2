import assert from 'node:assert/strict';
import { applyCommand, createGame } from '../../../../src/engine/commands/game';
import { observe } from '../../../../src/ai/observation';
import { RuleError } from '../../../../src/engine/core/state';
import { add, fixture } from '../../../../tests/helpers';
import type { Command, GameState } from '../../../../src/engine/types';
import type { nativeClient } from './client';

type Client = Awaited<ReturnType<typeof nativeClient>>;
const canonical = <T>(value: T): T => JSON.parse(JSON.stringify(value));
interface CommandWindow {
  state: GameState;
  commands: Command[];
}
/** 与训练环境一样逐步清理表现历史；trace 保留命令结果供差分。 */
function referenceWindow(window: CommandWindow, clearHistory: boolean) {
  let state = window.state;
  const results = [];
  for (const command of window.commands) {
    state = applyCommand(state, command);
    results.push({
      status: 'available',
      state: canonical(state),
    });
    if (clearHistory) {
      state.events = [];
      state.log = [];
    }
  }
  return { state, results };
}

/** 已成功的前缀可以提交；失败命令及其后续不执行，包含 RNG、序号和事件的整局保持不变。 */
export async function validateResidentProtocol(client: Client) {
  for (const seed of [0, 1, 19, -1, 0xffffffff]) {
    for (const rules of ['classic', 'shrine'] as const) {
      let expected = createGame(seed, rules);
      let created = await client.request({ op: 'create', seed, rules });
      assert.deepEqual((await client.request({ op: 'export' })).state, canonical(expected));
      for (const viewer of [1, 2] as const)
        assert.deepEqual(
          await client.request({ op: 'observe', viewer }),
          canonical(observe(expected, viewer)),
        );
      if (rules === 'shrine') {
        const command: Command = {
          type: 'choose-shrine',
          player: 1,
          shrineKind: expected.shrineDraft!.offers[1][0],
          parity: 'odd',
        };
        expected = applyCommand(expected, command);
        created = await client.request({
          op: 'step',
          revision: created.revision,
          commands: [command],
          observe: true,
        });
        assert.deepEqual(created.observation, canonical(observe(expected)));
        for (const viewer of [1, 2] as const)
          assert.deepEqual(
            await client.request({ op: 'observe', viewer }),
            canonical(observe(expected, viewer)),
          );
      }
    }
  }
  const state = fixture(),
    first = add(state, 26, 1, 4, 6),
    second = add(state, 26, 1, 5, 7),
    target = add(state, 14, 2, 4, 7);
  target.hp = target.maxHp = 1000;
  const commands: Command[] = [first, second].map((u) => ({
    type: 'attack',
    unitId: u.id,
    targetId: target.id,
  }));
  const window = {
    state,
    commands,
  };
  for (const clearHistory of [false, true]) {
    const { revision } = await client.request({ op: 'reset', state });
    const expected = referenceWindow(window, clearHistory);
    const result = await client.request({
      op: 'step',
      revision,
      commands,
      trace: true,
      clearHistory,
    });
    assert.equal(result.revision, revision + 2);
    assert.deepEqual(result.results, canonical(expected.results));
    const snapshot = {
      revision: result.revision,
      state: canonical(expected.state),
    };
    assert.deepEqual(await client.request({ op: 'export' }), snapshot);
    await assert.rejects(client.request({ op: 'step', revision, commands }), /stale revision/);
    assert.deepEqual(await client.request({ op: 'export' }), snapshot);
    await assert.rejects(client.request({ op: 'reset', state: { ...state, active: 3 } }));
    assert.deepEqual(await client.request({ op: 'export' }), snapshot);
  }
  const { revision } = await client.request({ op: 'reset', state });
  // 整个命令数组先解析，末项结构错误不能提交前一项。
  await assert.rejects(client.request({ op: 'step', revision, commands: [commands[0], {}] }));
  assert.deepEqual(await client.request({ op: 'export' }), {
    revision,
    state: canonical(state),
  });
  const invalid: Command = {
    type: 'attack',
    unitId: 'missing',
    targetId: target.id,
  };
  const afterFirst = applyCommand(state, commands[0]);
  let message = '';
  try {
    applyCommand(afterFirst, invalid);
  } catch (error) {
    assert.ok(error instanceof RuleError);
    message = error.message;
  }
  assert.ok(message);
  const partial = await client.request({
    op: 'step',
    revision,
    commands: [commands[0], invalid, commands[1]],
    trace: true,
  });
  assert.deepEqual(partial, {
    revision: revision + 1,
    results: [
      { status: 'available', state: canonical(afterFirst) },
      { status: 'invalid', message },
    ],
  });
  assert.deepEqual(await client.request({ op: 'export' }), {
    revision: revision + 1,
    state: canonical(afterFirst),
  });
  const resumed = await client.request({
    op: 'step',
    revision: partial.revision,
    commands: [commands[1]],
  });
  assert.equal(resumed.results[0].status, 'available');
  assert.deepEqual(
    (await client.request({ op: 'export' })).state,
    canonical(applyCommand(afterFirst, commands[1])),
  );

  // 暴击和策反均消耗正式随机数；驻留命令的状态与 TS 完整一致。
  const randomState = fixture(),
    attacker = add(randomState, 'u1', 1, 4, 6),
    victim = add(randomState, 14, 2, 4, 7);
  attacker.traits = ['s3'];
  victim.hp = victim.maxHp = 1e9;
  const attack: Command = {
    type: 'attack',
    unitId: attacker.id,
    targetId: victim.id,
  };
  assert.notEqual(applyCommand(randomState, attack).rng, randomState.rng);
  const loaded = await client.request({ op: 'reset', state: randomState });
  const resolved = await client.request({
    op: 'step',
    revision: loaded.revision,
    commands: [attack],
  });
  assert.deepEqual(resolved, {
    revision: loaded.revision + 1,
    results: [{ status: 'available' }],
  });
  assert.deepEqual(await client.request({ op: 'export' }), {
    revision: loaded.revision + 1,
    state: canonical(applyCommand(randomState, attack)),
  });
  // 召唤可能先支付人头，或老千K先掷候选数量，再发现自选不合法；失败必须全部回滚。
  for (const shrine of [false, true]) {
    const summoning = fixture();
    summoning.phase = 'summon';
    summoning.summonSlots = 2;
    if (shrine) {
      summoning.mode = 'shrine';
      summoning.regularSummons = 2;
      summoning.auras = { 1: [{ kind: 's13' }], 2: [] };
    }
    const command: Command = {
      type: 'summon',
      ultimate: true,
      chosenKind: 'u25',
    };
    assert.throws(() => applyCommand(summoning, command), /牢千K/);
    const { revision } = await client.request({
      op: 'reset',
      state: summoning,
    });
    const result = await client.request({
      op: 'step',
      revision,
      commands: [command, { type: 'summon' }],
    });
    assert.deepEqual(result, {
      revision,
      results: [
        {
          status: 'invalid',
          message: '本回合牢千K自选召唤已使用，或尚未获得光环。',
        },
      ],
    });
    assert.deepEqual(await client.request({ op: 'export' }), {
      revision,
      state: canonical(summoning),
    });
  }
  return {
    nativeCreationAndPrivateObservations: true,
    clearHistoryBothModes: true,
    staleRevisionRejected: true,
    malformedResetAndBatchAtomic: true,
    invalidStopsAtSuccessfulPrefix: true,
    resumeAfterFailure: true,
    randomAttackAndPassivesEquivalent: true,
    invalidSummonRollsBackPaymentAndRng: true,
  };
}
