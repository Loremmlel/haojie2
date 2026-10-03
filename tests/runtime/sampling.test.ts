import assert from 'node:assert/strict';
import test from 'node:test';
import { add, fixture } from '../helpers';
import { TrainingEnvironment } from '../../src/match/training';
import {
  applyCommand,
  applyRuntimeCommand,
  inspectCommand,
  createCommandInspector,
} from '../../src/engine/commands/game';
import { observe } from '../../src/ai/observation';
import { TrainingActionTree } from '../../src/ai/training/action-tree';
import {
  createDecisionEncoder,
  createSamplingEncoder,
} from '../../src/ai/training/encoding/decision';
import { TinyPolicy } from '../../scripts/training/economics/policy';
import { RuleError, ensure } from '../../src/engine/core/state';

test('批次拒绝与随机边界不泄漏预检模式，新观察重建准备，正式错误保留调用栈', () => {
  const s = fixture(),
    u = add(s, 1, 1, 4, 6);
  const command = { type: 'move' as const, unitId: u.id, x: 4, y: 7 };
  const inspect = createCommandInspector(s);
  for (let i = 0; i < 3; i++) {
    assert.deepEqual(inspect(command), inspectCommand(s, command));
    assert.deepEqual(inspect({ ...command, x: 0 }), inspectCommand(s, { ...command, x: 0 }));
  }
  const next = applyRuntimeCommand(s, command);
  assert.deepEqual(createCommandInspector(next)(command), inspectCommand(next, command));
  const summoning = { ...s, phase: 'summon' as const, summonSlots: 1 };
  assert.equal(inspectCommand(summoning, { type: 'summon' }).status, 'uncertain');
  assert.throws(
    () => ensure(false, '正式拒绝'),
    (e: unknown) => e instanceof RuleError && !!e.stack?.includes('正式拒绝'),
  );
  assert.throws(
    () => applyCommand(next, { ...command, x: 0 }),
    (e: unknown) => e instanceof RuleError && !!e.stack?.includes('RuleError'),
  );
});

test('连续运行提交保留旧局面、观察隔离及正式错误原子性', () => {
  const initial = fixture();
  const attacker = add(initial, 26, 1, 3, 5),
    victim = add(initial, 2, 2, 3, 6);
  victim.hp = 1;
  const before = structuredClone(initial);
  const command = {
    type: 'attack' as const,
    unitId: attacker.id,
    targetId: victim.id,
  };
  const next = applyRuntimeCommand(initial, command);
  assert.deepEqual(next, applyCommand(initial, command));
  assert.deepEqual(initial, before);
  const saved = structuredClone(next);
  assert.throws(() =>
    applyRuntimeCommand(next, {
      type: 'move',
      unitId: attacker.id,
      x: 9,
      y: 13,
    }),
  );
  assert.deepEqual(next, saved);
  const completed = applyRuntimeCommand(next, { type: 'react' });
  assert.deepEqual(completed, applyCommand(next, { type: 'react' }));
  assert.deepEqual(next, saved, '后续提交不能修改冻结的死亡反应');
  const env = TrainingEnvironment.fromState(initial);
  env.step(1, command);
  const observed = env.observation(2),
    independent = structuredClone(observed);
  observed.units[0].hp = -999;
  observed.pending[0].source.hp = -999;
  assert.deepEqual(env.observation(2), independent);
  assert.deepEqual(initial, before);
});

test('外部别名资源走整图兼容入口，保留引用关系且不修改输入', () => {
  const s = fixture(),
    u = add(s, 1, 1, 4, 9),
    v = add(s, 1, 1, 5, 9);
  u.traits = [21];
  u.abilityCharges = {};
  v.abilityCharges = u.abilityCharges;
  const command = { type: 'charge' as const, unitId: u.id, ability: 21, mode: 'skill' as const };
  for (const apply of [applyCommand, applyRuntimeCommand]) {
    const next = apply(s, command);
    assert.equal(next.units[0].abilityCharges, next.units[1].abilityCharges);
    assert.equal(next.units[1].abilityCharges?.[21]?.charge, 1);
    assert.deepEqual(u.abilityCharges, {});
    assert.notEqual(next.units[0].abilityCharges, u.abilityCharges);
  }
  const env = TrainingEnvironment.fromState(s);
  env.step(1, command);
  assert.equal(env.observation(1).units[1].abilityCharges?.[21]?.charge, 1);
});

test('决策内固定行前向与独立张量前向逐位一致，前缀、掩码和新局面不串用', () => {
  const s = fixture();
  const u = add(s, 26, 1, 2, 5);
  add(s, 1, 2, 3, 5);
  u.equipment.push('u28');
  for (let i = 0; i < 200; i++)
    s.deaths.push({
      id: `grave-${i}`,
      kind: 1,
      owner: 2,
      ply: i,
      revived: false,
    });
  const policy = new TinyPolicy(73129);
  for (const viewer of [1, 2, 1] as const) {
    const o = observe(s, viewer),
      tree = new TrainingActionTree(o, viewer);
    const owned = createDecisionEncoder(o, viewer),
      borrowed = createSamplingEncoder(o, viewer),
      forward = policy.decision();
    const root = tree.node();
    const nodes = [root, ...root.choices.flatMap((c, i) => (c.next ? [tree.node([i])] : [])), root];
    for (const node of nodes.filter((n) => n.choices.length)) {
      const a = owned(node),
        b = borrowed(node);
      assert.deepEqual(b, a);
      assert.deepEqual(forward(b), policy.logits(a));
      b.entity_mask[0] = false;
      assert.deepEqual(forward(b), policy.logits(b), '池化必须重新读取掩码');
      a.entities[0][0] = 999;
      assert.notEqual(borrowed(node).entities[0][0], 999, '拥有型张量不能污染只读行');
    }
    s.units[0].hp--;
  }
});
