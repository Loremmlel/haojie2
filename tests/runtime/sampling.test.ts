import assert from 'node:assert/strict';
import test from 'node:test';
import { add, card, fixture } from '../helpers';
import { TrainingEnvironment } from '../../src/match/training';
import {
  applyCommand,
  applyRuntimeCommand,
  inspectCommand,
  createCommandInspector,
  createGame,
} from '../../src/engine/commands/game';
import { observe } from '../../src/ai/observation';
import { TrainingActionTree } from '../../src/ai/training/action-tree';
import {
  createDecisionEncoder,
  createSamplingEncoder,
} from '../../src/ai/training/encoding/decision';
import {
  RuleError,
  ensure,
  getStats,
  piercing,
  addEffect,
  withoutRulePresentation,
} from '../../src/engine/core/state';
import { createTrainingCommandInspector, trainingActionSpace } from '../../src/ai/training/queries';
import { ALL_CELLS, attackRoutes, targets } from '../../src/engine/core/geometry';
import type { Command, GameState } from '../../src/engine/types';

test('训练省展示仍保留随机流、连锁和完整规则状态，异常后恢复正式事件', () => {
  const cases: { state: GameState; command: Command }[] = [];
  for (const variant of ['death', 'random', 'retaliation', 'heal-mark', 'passives', 'spell']) {
    const state = fixture(),
      source = add(state, variant === 'random' ? 1 : variant === 'heal-mark' ? 2 : 26, 1, 4, 4),
      victim = add(
        state,
        variant === 'retaliation' ? 'u18' : variant === 'random' ? 20 : 2,
        2,
        4,
        5,
      );
    let command: Command = { type: 'attack', unitId: source.id, targetId: victim.id };
    if (variant === 'death' || variant === 'random') victim.hp = 1;
    if (variant === 'retaliation') source.hp = 1;
    if (variant === 'heal-mark') {
      source.hp = source.maxHp - 1;
      addEffect(state, source, 'mark', 2, 0, 4, undefined, victim.id);
      command = { ...command, targetId: source.id, mode: 'heal' };
    }
    if (variant === 'passives') source.traits = ['s3', 's12'];
    if (variant === 'spell') command = { type: 'cast', cardId: card(state, 8), x: 4, y: 5 };
    cases.push({ state, command });
  }
  for (const { state, command } of cases) {
    const before = structuredClone(state),
      expected = applyRuntimeCommand(state, command),
      actual = withoutRulePresentation(() => applyRuntimeCommand(state, command));
    assert.equal(actual.log.length, 0);
    assert.ok(actual.events.length > 0);
    assert.equal(actual.events.length, expected.events.length);
    assert.deepEqual(
      actual.events.map((e) => e.id),
      expected.events.map((e) => e.id),
    );
    expected.log = [];
    expected.events = [];
    actual.log = [];
    actual.events = [];
    assert.deepEqual(actual, expected);
    assert.deepEqual(state, before);
  }
  const { state, command } = cases[0];
  assert.throws(
    () =>
      withoutRulePresentation(() =>
        withoutRulePresentation(() => {
          throw new Error('退出展示范围');
        }),
      ),
    /退出展示范围/,
  );
  const restored = applyRuntimeCommand(state, command);
  assert.ok(restored.log.length > 0);
  assert.ok(restored.events.some((e) => e.actor));
});

test('根动作沿用预检后仍保持随机边界、操作者权限和新局面隔离', () => {
  const play = fixture();
  add(play, 15, 1, 4, 6);
  add(play, 1, 1, 5, 6).traits = [5, 21];
  const states = [
    createGame(731270001),
    createGame(741270001, 'shrine'),
    play,
    { ...play, phase: 'summon' as const, summonSlots: 1 },
    applyCommand(play, { type: 'end' }),
  ];
  const statuses = new Set<string>();
  for (const state of states)
    for (const actor of [1, 2] as const) {
      const observation = observe(state, actor);
      const before = structuredClone(observation);
      const tree = new TrainingActionTree(observation, actor);
      assert.deepEqual(tree.actions, trainingActionSpace(observation, actor).actions);
      const inspect = createTrainingCommandInspector(observation, actor);
      for (const choice of tree.node().choices) {
        if (choice.next) continue;
        const result = inspect(choice.command);
        assert.equal(choice.status, result.status);
        assert.notEqual(result.status, 'invalid');
        statuses.add(choice.status);
      }
      assert.deepEqual(observation, before);
    }
  assert.ok(statuses.has('available'));
  assert.ok(statuses.has('uncertain'));
});

test('目标域与路线首次匹配保留重复ID、休眠地标和基地别名的原候选顺序', () => {
  for (const variant of ['ordinary', 'base-unit', 'dormant-duplicate', 'base-landmark']) {
    const s = fixture(),
      source = add(s, 26, 1, 4, 4),
      victim = add(s, 26, 2, 4, 5);
    add(s, 2, 1, 3, 4);
    if (variant === 'base-unit') victim.id = 'base-2';
    if (variant === 'dormant-duplicate' || variant === 'base-landmark') {
      const active = add(s, 's1', 2, 4, 6);
      s.units.pop();
      active.id = variant === 'base-landmark' ? 'base-2' : 'duplicate';
      s.landmarks = [{ ...active, x: 8, dormantSince: s.ply }, active];
    }
    const observation = structuredClone(observe(s, 1)),
      before = structuredClone(observation),
      tree = new TrainingActionTree(observation, 1),
      inspect = createTrainingCommandInspector(observation, 1),
      root = tree.node(),
      attack = root.choices.findIndex(
        (choice) =>
          choice.command.type === 'attack' &&
          choice.command.unitId === source.id &&
          choice.command.mode === undefined,
      );
    assert.ok(attack >= 0, variant);
    const command = root.choices[attack].command,
      domain = targets(tree.position),
      choices = tree.node([attack]).choices;
    const expected = domain.flatMap((candidate) => {
      const target = domain.find((t) => t.id === candidate.id)!;
      const routes = attackRoutes(
        tree.position,
        source,
        target,
        getStats(tree.position, source).range,
        piercing(source),
      );
      const c = { ...command, targetId: candidate.id };
      return routes.length || inspect(c).status !== 'invalid'
        ? [{ id: candidate.id, next: routes.length > 0 }]
        : [];
    });
    assert.deepEqual(
      choices.map((choice) => ({ id: choice.command.targetId, next: !!choice.next })),
      expected,
      variant,
    );
    for (const [i, choice] of choices.entries()) {
      if (!choice.next) continue;
      const target = domain.find((t) => t.id === choice.command.targetId)!;
      const routes = attackRoutes(
        tree.position,
        source,
        target,
        getStats(tree.position, source).range,
        piercing(source),
      );
      const expectedKeys = [
        { key: 'auto', command: choice.command },
        ...routes.map((route) => ({
          key: route.direction,
          command: { ...choice.command, direction: route.direction },
        })),
      ]
        .filter(({ command }) => inspect(command).status !== 'invalid')
        .map(({ key }) => key);
      assert.deepEqual(
        tree.node([attack, i]).choices.map((choice) => choice.key),
        expectedKeys,
        variant,
      );
    }
    assert.deepEqual(observation, before, variant);
  }
});

test('公开根动作被改写后重新预检，根叶子命令保留独立副本', () => {
  const observation = observe(createGame(731270001));
  const tree = new TrainingActionTree(observation, 1);
  const summon = tree.actions.find((a) => a.command.type === 'summon')!;
  summon.command.type = 'end';
  assert.ok(!tree.node().choices.some((choice) => choice.command.type === 'end'));

  const offered = observe(createGame(731270002));
  offered.summonOffer = { owner: 1, groups: [[], [], []], count: 2 };
  const choices = new TrainingActionTree(offered, 1);
  choices.actions[0].command.offerIndices![1] = 0;
  assert.ok(
    choices.node().choices.every((choice) => new Set(choice.command.offerIndices).size === 2),
    '原地改写数组参数也必须使旧预检失效',
  );

  const original = new TrainingActionTree(observation, 1);
  const action = original.actions.find((a) => a.command.type === 'summon')!;
  const leaf = original.node().choices.find((choice) => choice.command.type === 'summon')!;
  leaf.command.type = 'end';
  assert.equal(action.command.type, 'summon', '修改返回的命令不能改写动作列表');
});

test('空参数前缀提前置零，原下标、友方治疗、随机攻击与完整合法叶子保持', () => {
  const s = fixture();
  const source = add(s, 1, 1, 4, 4);
  const friend = add(s, 26, 1, 4, 5);
  friend.hp -= 10;
  const enemy = add(s, 26, 2, 5, 4);
  const healer = add(s, 2, 1, 3, 5);
  const observation = observe(s, 1),
    before = structuredClone(observation);
  const tree = new TrainingActionTree(observation, 1);
  const root = tree.node();
  const attack = root.choices.findIndex(
    (c) => c.command.type === 'attack' && c.command.unitId === source.id,
  );
  const targets = tree.node([attack]);
  const invalid = targets.choices.findIndex((c) => c.command.targetId === friend.id);
  assert.ok(invalid >= 0, '保留原来存在的候选及其下标');
  assert.equal(targets.mask![invalid], false);
  assert.equal(createSamplingEncoder(observation, 1)(targets).candidate_mask[invalid], false);
  assert.equal(tree.node([attack, invalid]).choices.length, 0);
  const clock = root.choices.findIndex((c) => c.command.type === 'clock');
  assert.ok(clock >= 0);
  assert.equal(root.mask![clock], false, '没有可用时钟目标的根也应跳过');
  for (const command of [
    { type: 'attack' as const, unitId: source.id, targetId: enemy.id },
    { type: 'attack' as const, unitId: healer.id, targetId: friend.id },
  ])
    for (const { node, selected } of tree.trace(command))
      assert.equal(node.mask![selected], true, '随机攻击与合法友方治疗均保留');
  const leaves = (masked: boolean) => {
    const result: string[] = [];
    const visit = (cursor: number[]) => {
      const node = tree.node(cursor);
      for (const [i, choice] of node.choices.entries()) {
        if (masked && !node.mask![i]) continue;
        if (choice.next) visit([...cursor, i]);
        else result.push(JSON.stringify(choice.command));
      }
    };
    visit([]);
    return result.sort();
  };
  assert.deepEqual(leaves(true), leaves(false), '完整合法叶子及重复项均不减少');
  assert.deepEqual(observation, before);
});

test('移动与部署点域保留逐命令预检顺序和冲撞后果，公共阶段失败仍为空域', () => {
  const s = fixture();
  const normal = add(s, 1, 1, 2, 5);
  const giant = add(s, 5, 1, 6, 5);
  const runner = add(s, 'u12', 1, 4, 5);
  runner.mode = 'move';
  runner.moves = 1;
  add(s, 2, 2, 4, 6);
  s.hands[1].push({ id: 'domain-card', kind: 23, drawnAt: 0, summonedPly: s.ply });
  const commands: Command[] = [
    ...[normal, giant, runner].map((u) => ({ type: 'move' as const, unitId: u.id })),
    { type: 'deploy', cardId: 'domain-card' },
  ];
  for (const state of [s, { ...s, phase: 'summon' as const }]) {
    const inspect = createCommandInspector(state);
    for (const command of commands) {
      const expected = ALL_CELLS.flatMap((point) => {
        const inspection = inspectCommand(state, { ...command, ...point });
        return inspection.status === 'invalid' ? [] : [{ point, inspection }];
      });
      assert.deepEqual(inspect.points(command), expected);
    }
  }
  assert.equal(createTrainingCommandInspector(observe(s, 2), 2).points(commands[0])?.length, 0);
});

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
  const command = {
    type: 'charge' as const,
    unitId: u.id,
    ability: 21,
    mode: 'skill' as const,
  };
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

test('决策编码的固定行与拥有型张量一致，前缀、掩码和新局面不串用', () => {
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
  for (const viewer of [1, 2, 1] as const) {
    const o = observe(s, viewer),
      tree = new TrainingActionTree(o, viewer);
    const owned = createDecisionEncoder(o, viewer),
      borrowed = createSamplingEncoder(o, viewer);
    const root = tree.node();
    const nodes = [root, ...root.choices.flatMap((c, i) => (c.next ? [tree.node([i])] : [])), root];
    for (const node of nodes.filter((n) => n.choices.length)) {
      const a = owned(node),
        b = borrowed(node);
      assert.deepEqual(b, a);
      b.entity_mask[0] = false;
      assert.equal(borrowed(node).entity_mask[0], true, '重新编码必须恢复有效实体掩码');
      a.entities[0].ids[0] = 999n;
      assert.notEqual(borrowed(node).entities[0].ids[0], 999n, '拥有型张量不能污染只读行');
    }
    s.units[0].hp--;
  }
});
