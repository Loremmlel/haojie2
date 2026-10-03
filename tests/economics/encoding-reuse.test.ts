import assert from 'node:assert/strict';
import test from 'node:test';
import { observe } from '../../src/ai/observation';
import { TrainingActionTree } from '../../src/ai/training/action-tree';
import {
  createDecisionEncoder,
  createSamplingEncoder,
  encodeDecision,
} from '../../src/ai/training/encoding/decision';
import { createPositionEncoder, encodePosition } from '../../src/ai/training/encoding/state';
import { numeric } from '../../src/ai/training/encoding/schema';
import type { Command } from '../../src/engine/types';
import { add, fixture } from '../helpers';

test('预编译标量保留边界、非整数与负零的逐位结果', () => {
  for (const n of [
    ...Array.from({ length: 400 }, (_, i) => i - 90),
    -0,
    0.5,
    -0.25,
    Number.MIN_VALUE,
    -Number.MIN_VALUE,
    Number.MAX_VALUE,
  ])
    assert.ok(Object.is(numeric(n), (Math.sign(n) * Math.log1p(Math.abs(n))) / 8), String(n));
  assert.equal(numeric(true), 1);
  assert.equal(numeric(false), 0);
  for (const n of [NaN, Infinity, -Infinity]) assert.throws(() => numeric(n));
});

test('固定编码复用隔离前缀新增身份、返回张量、引用表和异常', () => {
  const s = fixture();
  const u = add(s, 1, 1, 4, 4);
  const observation = observe(s, 1);
  const before = structuredClone(observation);
  const encode = createPositionEncoder(observation, 1);
  const prefixes: (Command | undefined)[] = [
    {
      type: 'skill',
      unitId: u.id,
      secondId: 'only-first-prefix',
      deathId: 'dead-first',
    },
    {
      type: 'attack',
      unitId: u.id,
      targetId: 'only-second-prefix',
      path: [{ x: 4, y: 4 }],
    },
    undefined,
    { type: 'synthesize', recipeId: 'sage', materialIds: ['a', 'b', 'c'] },
    { type: 'choose-summons', offerIndices: [1, 2] },
  ];
  for (const prefix of [...prefixes, ...[...prefixes].reverse()]) {
    const expected = encodePosition(observation, 1, prefix);
    const actual = encode(prefix);
    const { reference: a, ...data } = actual;
    const { reference: b, ...expectedData } = expected;
    assert.deepEqual(data, expectedData);
    assert.equal(a('candidate-only'), b('candidate-only'));
    actual.entities[0][0] = 999;
    actual.entities.push([999]);
    actual.kinds[0] = 999;
    actual.globals[0] = 999;
    actual.indices.set('bad-output', 999);
    a('pollution');
  }
  assert.throws(() => encode({ type: 'attack', mode: 'unknown' }), /词表不支持/);
  assert.deepEqual(encode().entities, encodePosition(observation, 1).entities);
  assert.deepEqual(observation, before);
  const other = observe(s, 2);
  assert.deepEqual(createPositionEncoder(other, 2)().entities, encodePosition(other, 2).entities);
  add(s, 2, 2, 5, 5);
  assert.notDeepEqual(createPositionEncoder(observe(s, 1), 1)().entities, encode().entities);
});

test('批次编码仍拒绝私有字段、未知嵌套字段、非法观察方和空节点', () => {
  const s = fixture();
  add(s, 1, 1, 4, 4);
  const observation = observe(s, 1);
  assert.throws(() => createPositionEncoder(s, 1)(), /禁止携带/);
  assert.throws(() => createPositionEncoder(observation, 3 as 1)(), /观察方/);
  const bad = structuredClone(observation);
  Object.assign(bad.units[0], { unknown: 1 });
  const encodeBad = createPositionEncoder(bad, 1);
  assert.throws(() => encodeBad(), /未编码字段/);
  assert.throws(() => encodeBad(), /未编码字段/);
  const tree = new TrainingActionTree(observation, 1);
  const encode = createDecisionEncoder(observation, 1);
  assert.throws(() => encode({ cursor: [], stage: 'point', choices: [] }), /空动作分支/);
  assert.deepEqual(encode(tree.node()), encodeDecision(observation, 1, tree.node()));
});

test('同步编码工作区复用容量但清理前缀与掩码，拥有型出口继续隔离', () => {
  const s = fixture();
  add(s, 1, 1, 4, 4);
  add(s, 2, 2, 5, 4);
  const observation = observe(s, 1);
  const tree = new TrainingActionTree(observation, 1);
  const encode = createSamplingEncoder(observation, 1, true);
  const owned = createDecisionEncoder(observation, 1);
  const root = tree.node();
  const nodes = [
    root,
    ...root.choices
      .flatMap((c, i) => (c.next ? [tree.node([i])] : []))
      .filter((n) => n.choices.length),
    root,
  ];
  const snapshots = [];
  const stable = owned(root);
  const stableBefore = structuredClone(stable);
  for (const node of [...nodes, ...[...nodes].reverse()]) {
    const expected = owned(node);
    const input = encode(node);
    assert.deepEqual(input, expected);
    snapshots.push({ actual: structuredClone(input), expected });
    // 上层可追加回合外候选；下一节点必须清掉追加项和被使用的掩码。
    input.candidate_mask[0] = false;
    input.candidates.push(Array(64).fill(1));
    input.sources.push(-1);
    input.targets.push(-1);
  }
  assert.deepEqual(stable, stableBefore);
  for (const saved of snapshots) assert.deepEqual(saved.actual, saved.expected);
});
