import assert from 'node:assert/strict';
import test from 'node:test';
import { clonePosition } from '../../src/engine/core/clone';
import { applyCommand, createGame, inspectCommand } from '../../src/engine/commands/game';
import { add, card, fixture, unit } from '../helpers';

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

test('正式结果是可原生序列化的独立快照，未修改的棋子与手牌也不共享可变数据', () => {
  const s = fixture();
  const attacker = add(s, 10, 1, 4, 6);
  const target = add(s, 14, 2, 4, 8);
  target.hp = target.maxHp = 1000;
  const idle = add(s, 14, 1, 1, 1);
  idle.effects.push({ type: 'burn', owner: 2, from: 1, until: 10, amount: 5 });
  card(s, 18);
  const before = structuredClone(s);
  const command = { type: 'attack' as const, unitId: attacker.id, targetId: target.id };
  assert.equal(inspectCommand(s, command).status, 'available');
  const a = applyCommand(s, command);
  const b = applyCommand(s, command);
  assert.deepEqual(a, structuredClone(a));
  assert.deepEqual(a, b);
  // 新建的投石标记当次引爆后须移除，不能因分支对象身份改变而残留。
  assert.equal(
    unit(a, target.id).effects.some((e) => e.type === 'mark'),
    false,
  );
  unit(a, idle.id).effects[0].amount = 999;
  a.hands[1][0].kind = 1;
  a.turns[1] = 99;
  a.log.push('仅修改导出快照');
  assert.equal(unit(b, idle.id).effects[0].amount, 5);
  assert.equal(b.hands[1][0].kind, 18);
  assert.deepEqual(s, before);
});

test('结算中途遇随机边界或异常时丢弃分支，已消费的标记与操作预算不泄露', () => {
  const s = fixture();
  const attacker = add(s, 1, 1, 4, 6);
  const target = add(s, 14, 2, 4, 7);
  target.effects.push({ type: 'mark', owner: 1, from: s.ply, until: s.ply + 2, global: true });
  const before = structuredClone(s);
  const command = { type: 'attack' as const, unitId: attacker.id, targetId: target.id };
  assert.equal(inspectCommand(s, command).status, 'uncertain');
  assert.deepEqual(s, before);
  const interrupted = new Error('模拟源中止');
  assert.throws(
    () =>
      applyCommand(s, command, () => {
        throw interrupted;
      }),
    (e) => e === interrupted,
  );
  assert.deepEqual(s, before);
  // 异常必须清除临时模拟源，之后正式执行仍使用原 PRNG。
  assert.deepEqual(applyCommand(s, command), applyCommand(before, command));
  assert.deepEqual(s, before);
});
