import assert from 'node:assert/strict';
import test from 'node:test';
import { clonePosition } from '../../src/engine/core/clone';
import { applyCommand, createGame } from '../../src/engine/commands/game';
import { add, fixture } from '../helpers';

test('局面复制保留嵌套状态和引用关系，结果可独立修改', () => {
  const s = fixture();
  const u = add(s, 'u12', 1, 3, 5);
  u.abilityCharges = { u6: { charge: 2, readyCharge: 1, chargeType: 'skill', lastCharge: 1 } };
  u.effects.push({ type: 'burn', from: 1, until: 10, owner: 2, amount: 10 });
  s.pending.push({ kind: 'bounce', owner: 1, source: u, amount: 30 });
  s.summonOffer = { owner: 1, groups: [s.hands[1]], count: 2 };
  const sparse = new Array(3);
  sparse[2] = undefined;
  const extended = { ...s, optional: undefined, sparse, self: null as unknown };
  extended.self = extended;
  const before = structuredClone(extended);
  const copy = clonePosition(extended);
  assert.deepEqual(copy, before);
  assert.equal(copy.self, copy);
  assert.equal(copy.pending[0].source, copy.units[0]);
  assert.equal(copy.summonOffer!.groups[0], copy.hands[1]);
  copy.units[0].effects[0].amount = 999;
  copy.units[0].abilityCharges!.u6!.charge = 999;
  copy.hands[1].length = 0;
  assert.deepEqual(extended, before);
});

test('非普通扩展数据整份回退，不破坏共享引用', () => {
  const s = fixture();
  const shared = { value: 1 };
  const extended = { ...s, shared, nested: new Map([['shared', shared]]) };
  const copy = clonePosition(extended);
  assert.deepEqual(copy, structuredClone(extended));
  assert.equal(copy.nested.get('shared'), copy.shared);
  assert.notEqual(copy.shared, shared);
});

test('合法随机命令和失败命令都不修改输入，随机序列可重放', () => {
  const s = createGame(423470002);
  const before = structuredClone(s);
  const a = applyCommand(s, { type: 'summon' });
  const b = applyCommand(s, { type: 'summon' });
  assert.deepEqual(a, b);
  assert.deepEqual(s, before);
  assert.throws(() => applyCommand(s, { type: 'move', unitId: 'missing', x: 1, y: 1 }));
  assert.deepEqual(s, before);
});
