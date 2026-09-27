import assert from 'node:assert/strict';
import { applyCommand } from '../../../src/engine/commands/game';
import { RuleError } from '../../../src/engine/core/state';
import { add, fixture } from '../../../tests/helpers';
import type { Command, GameState } from '../../../src/engine/types';
import type { nativeClient } from './client';

type Client = Awaited<ReturnType<typeof nativeClient>>;
const canonical = <T>(value: T): T => JSON.parse(JSON.stringify(value));
export interface CommandWindow {
  source: string;
  game: number;
  start: number;
  state: GameState;
  commands: Command[];
}
export const windowIdentity = ({ source, game, start, commands }: CommandWindow) => ({
  source,
  game,
  start,
  commands: commands.length,
});

/** 与 TrainingEnvironment 一样逐步清理表现历史；trace 保留清理前的命令结果供差分。 */
function referenceWindow(window: CommandWindow, trace: boolean, clearHistory = true) {
  let state = window.state;
  const results = [];
  for (const command of window.commands) {
    state = applyCommand(state, command);
    results.push({ status: 'available', ...(trace ? { state: canonical(state) } : {}) });
    if (clearHistory) {
      state.events = [];
      state.log = [];
    }
  }
  return { state, results };
}

/** 已成功的前缀可以提交；失败命令及其后续不执行，包含 RNG、序号和事件的整局保持不变。 */
export async function validateResidentProtocol(client: Client) {
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
  const window = { state, commands, source: 'resident-contract', game: 0, start: 0 };
  for (const clearHistory of [false, true]) {
    const { revision } = await client.request({ op: 'reset', state });
    const expected = referenceWindow(window, true, clearHistory);
    const result = await client.request({
      op: 'step',
      revision,
      commands,
      trace: true,
      clearHistory,
    });
    assert.equal(result.revision, revision + 2);
    assert.deepEqual(result.results, canonical(expected.results));
    const snapshot = { revision: result.revision, state: canonical(expected.state) };
    assert.deepEqual(await client.request({ op: 'export' }), snapshot);
    await assert.rejects(client.request({ op: 'step', revision, commands }), /stale revision/);
    assert.deepEqual(await client.request({ op: 'export' }), snapshot);
    await assert.rejects(client.request({ op: 'reset', state: { ...state, active: 3 } }));
    assert.deepEqual(await client.request({ op: 'export' }), snapshot);
  }
  const { revision } = await client.request({ op: 'reset', state });
  // 整个命令数组先解析，末项结构错误不能提交前一项。
  await assert.rejects(client.request({ op: 'step', revision, commands: [commands[0], {}] }));
  assert.deepEqual(await client.request({ op: 'export' }), { revision, state: canonical(state) });
  const invalid: Command = { type: 'attack', unitId: 'missing', targetId: target.id };
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

  // 暴击已消耗随机数并造成伤害后，未移植的策反被动拒绝整条命令。
  const randomState = fixture(),
    attacker = add(randomState, 'u1', 1, 4, 6),
    victim = add(randomState, 14, 2, 4, 7);
  attacker.traits = ['s3'];
  victim.hp = victim.maxHp = 1e9;
  const attack: Command = { type: 'attack', unitId: attacker.id, targetId: victim.id };
  assert.notEqual(applyCommand(randomState, attack).rng, randomState.rng);
  const loaded = await client.request({ op: 'reset', state: randomState });
  const refused = await client.request({
    op: 'step',
    revision: loaded.revision,
    commands: [attack, attack],
  });
  assert.deepEqual(refused, {
    revision: loaded.revision,
    results: [{ status: 'unsupported', reason: 'shrine-conversion' }],
  });
  assert.deepEqual(await client.request({ op: 'export' }), {
    revision: loaded.revision,
    state: canonical(randomState),
  });
  return {
    clearHistoryBothModes: true,
    staleRevisionRejected: true,
    malformedResetAndBatchAtomic: true,
    invalidStopsAtSuccessfulPrefix: true,
    resumeAfterFailure: true,
    unsupportedRollsBackRngAndDamage: true,
  };
}

/** 只验证真实记录中连续且全部受支持的命令段，不能跨过未移植命令拼接成整局。 */
export async function validateWindow(client: Client, window: CommandWindow) {
  const before = JSON.stringify(window);
  const { revision } = await client.request({ op: 'reset', state: window.state });
  const expected = referenceWindow(window, true);
  const actual = await client.request({
    op: 'step',
    revision,
    commands: window.commands,
    trace: true,
    clearHistory: true,
  });
  assert.deepEqual(actual.results, expected.results, JSON.stringify(windowIdentity(window)));
  assert.equal(actual.revision, revision + window.commands.length);
  assert.deepEqual(await client.request({ op: 'export' }), {
    revision: actual.revision,
    state: canonical(expected.state),
  });
  assert.equal(JSON.stringify(window), before);
}

/** 三次请求的总成本包含装载、命令批次和导出；另列批次耗时，不能将它冒充端到端。 */
export async function benchmarkWindows(client: Client, windows: CommandWindow[]) {
  const expected = windows.map((window) => canonical(referenceWindow(window, false)));
  const runTs = (json: boolean) =>
    windows.map((window) => {
      const input = json ? canonical(window) : window;
      const result = referenceWindow(input, false);
      return json ? canonical(result) : result;
    });
  const runRust = async () => {
    const results = [];
    let stepMs = 0;
    const start = performance.now();
    for (const window of windows) {
      const { revision } = await client.request({ op: 'reset', state: window.state });
      const stepStart = performance.now();
      const result = await client.request({
        op: 'step',
        revision,
        commands: window.commands,
        clearHistory: true,
      });
      stepMs += performance.now() - stepStart;
      results.push({
        state: (await client.request({ op: 'export' })).state,
        results: result.results,
      });
    }
    const totalMs = performance.now() - start;
    assert.deepEqual(results, expected);
    return { stepMs, totalMs };
  };
  const clock = (fn: () => unknown) => {
    const start = performance.now();
    fn();
    return performance.now() - start;
  };
  runTs(false);
  runTs(true);
  await runRust();
  const rounds: { tsPlainMs: number; tsJsonMs: number; rustStepMs: number; rustTotalMs: number }[] =
    [];
  for (let round = 0; round < 3; round++) {
    let tsPlainMs = 0,
      tsJsonMs = 0,
      rust!: Awaited<ReturnType<typeof runRust>>;
    const ts = () => {
      tsPlainMs = clock(() => runTs(false));
      tsJsonMs = clock(() => runTs(true));
    };
    if (round % 2 === 0) {
      ts();
      rust = await runRust();
    } else {
      rust = await runRust();
      ts();
    }
    rounds.push({ tsPlainMs, tsJsonMs, rustStepMs: rust.stepMs, rustTotalMs: rust.totalMs });
  }
  const median = (key: keyof (typeof rounds)[number]) =>
    rounds.map((r) => r[key]).sort((a, b) => a - b)[1];
  return {
    name: 'resident-command-windows',
    windows: windows.map(windowIdentity),
    commands: windows.reduce((n, w) => n + w.commands.length, 0),
    rounds,
    median: {
      tsPlainMs: median('tsPlainMs'),
      tsJsonMs: median('tsJsonMs'),
      rustStepMs: median('rustStepMs'),
      rustTotalMs: median('rustTotalMs'),
      totalVsPlain: median('tsPlainMs') / median('rustTotalMs'),
      totalVsJson: median('tsJsonMs') / median('rustTotalMs'),
    },
  };
}
