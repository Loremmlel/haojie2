import assert from 'node:assert/strict';
import test from 'node:test';
import { observe } from '../../src/ai/observation';
import { TrainingActionTree } from '../../src/ai/training/action-tree';
import { createDecisionEncoder, encodeDecision } from '../../src/ai/training/encoding/decision';
import { createPositionEncoder, encodePosition } from '../../src/ai/training/encoding/state';
import type { Command } from '../../src/engine/types';
import { add, fixture } from '../helpers';

test('固定编码复用隔离前缀新增身份、返回张量、引用表和异常', () => {
  const s = fixture();
  const u = add(s, 1, 1, 4, 4);
  const observation = observe(s, 1);
  const before = structuredClone(observation);
  const encode = createPositionEncoder(observation, 1);
  const prefixes: (Command | undefined)[] = [
    { type: 'skill', unitId: u.id, secondId: 'only-first-prefix', deathId: 'dead-first' },
    { type: 'attack', unitId: u.id, targetId: 'only-second-prefix', path: [{ x: 4, y: 4 }] },
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
