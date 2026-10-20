import assert from 'node:assert/strict';
import test from 'node:test';
import { createGame } from '../../src/engine';
import { applyRuntimeCommand } from '../../src/engine/commands/game';
import { importRuntimePosition } from '../../src/engine/runtime/position';
import { fingerprint, positionKey } from '../../src/ai/observation';
import { createStateKeys } from '../../src/ai/simulation/state-key';
import { distribution } from '../../src/ai/simulation/simulate';

test('搜索快照键保留公共字符串，独立快照与新的搜索上下文不共享旧结果', () => {
  const s = createGame(19),
    keys = createStateKeys();
  assert.equal(keys.positionKey(s), positionKey(s));
  assert.equal(keys.fingerprint(s), fingerprint(s));
  const next = structuredClone(s);
  next.bases[1]--;
  assert.notEqual(keys.positionKey(next), keys.positionKey(s));
  assert.equal(keys.fingerprint(next), fingerprint(next));
  // 权威状态可以修改，但不得再传给旧的只读缓存；新搜索必须创建自己的上下文。
  s.bases[2]--;
  const fresh = createStateKeys();
  assert.equal(fresh.positionKey(s), positionKey(s));
  assert.equal(fresh.fingerprint(s), fingerprint(s));
});

test('共享原字符串键不改变精确与抽样随机分布、权重或工作量', () => {
  const s = createGame(19),
    original = structuredClone(s);
  for (const limit of [2, 64]) {
    const expected = distribution(s, { type: 'summon' }, limit, 3);
    assert.ok(expected.outcomes.length > 0);
    const actual = distribution(
      s,
      { type: 'summon' },
      limit,
      3,
      0,
      undefined,
      createStateKeys().positionKey,
    );
    assert.deepEqual(actual, expected);
  }
  assert.deepEqual(s, original);
});

test('搜索内部运行分支保持精确与抽样概率、预算及后续分支隔离', () => {
  const original = createGame(19),
    saved = structuredClone(original),
    runtime = importRuntimePosition(original);
  for (const limit of [2, 64]) {
    const expected = distribution(original, { type: 'summon' }, limit, 3);
    const actual = distribution(
      runtime,
      { type: 'summon' },
      limit,
      3,
      0,
      undefined,
      positionKey,
      applyRuntimeCommand,
    );
    assert.deepEqual(actual, expected);
    const snapshots = structuredClone(actual.outcomes);
    for (const out of actual.outcomes)
      assert.deepEqual(
        distribution(
          out.state,
          { type: 'summon' },
          limit,
          3,
          0,
          undefined,
          positionKey,
          applyRuntimeCommand,
        ),
        distribution(out.state, { type: 'summon' }, limit, 3),
      );
    assert.deepEqual(actual.outcomes, snapshots);
    assert.deepEqual(runtime, saved);
  }
  assert.deepEqual(original, saved);
});
