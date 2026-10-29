import assert from 'node:assert/strict';
import test from 'node:test';
import {
  applyCommand,
  createCommandInspector,
  inspectCommand,
} from '../../src/engine/commands/game';
import {
  findTarget,
  inspectPreparedAttack,
  kill,
  prepareAttack,
  resolution,
} from '../../src/engine/commands/combat';
import { chargeAction, useSkill } from '../../src/engine/commands/abilities';
import { ALL_CELLS } from '../../src/engine/core/geometry';
import {
  addEffect,
  chooseMode,
  emit,
  getStats,
  RuleError,
  withRuleInspection,
} from '../../src/engine/core/state';
import { eventActor, withEventFacts } from '../../src/engine/core/event-facts';
import { imagined, observe } from '../../src/ai/observation';
import type { Command, GameState, Kind } from '../../src/engine/types';
import { add, card, fixture } from '../helpers';

test('嵌套预检退出后恢复事件快照和日志，保留原始事件与正式序号', () => {
  const s = fixture(),
    u = add(s, 1, 1, 4, 4);
  const serial = s.serial,
    log = [...s.log];
  let factsBuilt = 0;
  assert.throws(() =>
    withRuleInspection(() =>
      withRuleInspection(() =>
        withEventFacts(
          s,
          () => {
            factsBuilt++;
            return { actor: eventActor(u), action: 'attack' };
          },
          () => {
            emit(s, { type: 'damage', to: u, unitId: u.id, amount: 5 }, '预检');
            throw new Error('退出');
          },
        ),
      ),
    ),
  );
  assert.equal(s.serial, serial + 1);
  assert.equal(s.events.at(-1)?.unitId, u.id);
  assert.equal(s.events.at(-1)?.amount, 5);
  assert.equal(s.events.at(-1)?.actor, undefined);
  assert.equal(s.events.at(-1)?.action, undefined);
  assert.equal(factsBuilt, 0);
  assert.deepEqual(s.log, log);
  withEventFacts(
    s,
    () => {
      factsBuilt++;
      return { actor: eventActor(u), action: 'attack' };
    },
    () => {
      emit(s, { type: 'damage', to: u, unitId: u.id, amount: 5 }, '正式');
    },
  );
  assert.equal(factsBuilt, 1);
  u.x++;
  assert.equal(s.events.at(-1)?.actor?.x, 4);
  assert.equal(s.events.at(-1)?.subject?.x, 4);
  assert.equal(s.events.at(-1)?.to?.x, 4);
  assert.equal(s.events.at(-1)?.id, `e${serial + 1}`);
  assert.equal(s.log.at(-1), `${s.ply} · 正式`);
});

test('技能预检跳过纯展示事实，仍执行相同伤害、资源和原始事件序列', () => {
  const formal = fixture(),
    source = add(formal, 5, 1, 4, 4);
  add(formal, 26, 2, 6, 4);
  const preview = structuredClone(formal),
    command: Command = { type: 'skill', unitId: source.id };
  useSkill(formal, command, resolution());
  withRuleInspection(() => useSkill(preview, command, resolution()));
  assert.deepEqual({ ...preview, events: [], log: [] }, { ...formal, events: [], log: [] });
  const raw = (s: GameState) =>
    s.events.map(({ id, type, unitId, owner, amount, text }) => ({
      id,
      type,
      unitId,
      owner,
      amount,
      text,
    }));
  assert.deepEqual(raw(preview), raw(formal));
  assert.ok(formal.events.some((event) => event.action === 'quake' && event.actor && event.area));
  assert.ok(preview.events.every((event) => event.actor === undefined && event.area === undefined));
});

test('惰性事实不跳过冰痕提前坐标检查和蓄力来源检查', () => {
  for (const inspection of [false, true]) {
    const s = fixture(),
      source = add(s, 'u24', 1, 4, 4);
    source.silenced = true;
    s.phase = 'summon';
    const before = structuredClone(s);
    const run = (fn: () => unknown) => (inspection ? withRuleInspection(fn) : fn());
    assert.throws(
      () => run(() => useSkill(s, { type: 'skill', unitId: source.id }, resolution())),
      {
        name: 'RuleError',
        message: '请选择棋盘格。',
      },
    );
    assert.throws(
      () => run(() => chargeAction(s, { type: 'charge', unitId: 'missing', mode: 'move' })),
      {
        name: 'RuleError',
        message: '请选择仍在场上的随从。',
      },
    );
    assert.deepEqual(s, before);
  }
});

/** 使用正式命令入口作为参照；遇到随机请求即中断，不编造随机结果。 */
function compare(s: GameState, commands: Command[]) {
  const unknown = Symbol('随机边界');
  const position = imagined(observe(s, s.active));
  const inspect = createCommandInspector(position);
  const before = structuredClone(s),
    publicBefore = structuredClone(position);
  for (const c of commands) {
    let expected: ReturnType<typeof inspectCommand>;
    try {
      applyCommand(s, c, () => {
        throw unknown;
      });
      expected = { status: 'available' };
    } catch (e) {
      if (e === unknown) expected = { status: 'uncertain' };
      else if (e instanceof RuleError) expected = { status: 'invalid', message: e.message };
      else throw e;
    }
    assert.deepEqual(inspect(c), expected, JSON.stringify(c));
    assert.deepEqual(inspectCommand(position, c), expected, JSON.stringify(c));
  }
  assert.deepEqual(s, before);
  assert.deepEqual(position, publicBefore);
}

function attackInspectionGuard(s: GameState, c: Command, deterministic?: () => boolean) {
  const source = { ...s.units.find((u) => u.id === c.unitId)! };
  chooseMode(s, source, 'attack');
  const target = findTarget(s, c.targetId);
  return inspectPreparedAttack(
    s,
    source,
    target,
    prepareAttack(s, source, target, { mode: c.mode, path: c.path, direction: c.direction }),
    deterministic,
  );
}

test('未自选召唤保持费用拒绝与首次随机边界，自选仍由正式规则验证', () => {
  for (const shrine of [false, true])
    for (const heads of [0, 3])
      for (const slots of [0, 2]) {
        const s = fixture();
        s.phase = 'summon';
        s.summonSlots = slots;
        s.heads[1] = heads;
        if (shrine) {
          s.mode = 'shrine';
          s.auras = { 1: [{ kind: 's13' }, { kind: 'laoqian' }], 2: [] };
          s.regularSummons = 2;
        }
        compare(
          s,
          (['summon', 'extra-summon'] as const).flatMap((type) =>
            [false, true].flatMap((ultimate) =>
              [undefined, 1, 'u1'].map((chosenKind) => ({ type, ultimate, chosenKind }) as Command),
            ),
          ),
        );
      }
});

test('原生与继承复活的批量预检，和正式入口的资格/落点/错误顺序一致', () => {
  for (const kind of ['u19', 4] as const) {
    const s = fixture(),
      mage = add(s, kind, 1, 4, 4),
      victim = add(s, 1, 1, 5, 4);
    mage.traits = ['u19'];
    mage.charge = mage.readyCharge = 5;
    mage.chargeType = 'attack';
    kill(s, victim, { owner: 2, kind: 'spell' });
    s.ply += 2;
    compare(
      s,
      [s.deaths[0].id, 'missing'].flatMap((deathId) =>
        ALL_CELLS.map((p) => ({
          type: 'skill' as const,
          unitId: mage.id,
          ability: 'u19' as const,
          deathId,
          ...p,
        })),
      ),
    );
  }
});

test('复活批点域的共享模板与射程场保留双方、大体型、克隆和地标历史', () => {
  for (const owner of [1, 2] as const)
    for (const size of [1, 2])
      for (const range of [1.5, 4]) {
        const s = fixture();
        s.active = owner;
        const y = owner === 1 ? 4 : 8,
          mage = add(s, 'u19', owner, 4, y),
          opponent = owner === 1 ? 2 : 1;
        mage.size = size;
        mage.rangeBonus = range - 4;
        add(s, 'u25', owner, 6, y);
        add(s, 'u25', owner, 6, y);
        add(s, 'wall', opponent, 7, y);
        const landmark = add(s, 's1', opponent, 6, y + 1);
        s.units.pop();
        s.landmarks = [landmark];
        s.deaths = ([1, 5, 'u25', 's1'] as const).map((kind, index) => ({
          id: `record-${index}`,
          kind,
          owner,
          ply: s.ply - 1,
          revived: false,
        }));
        const before = structuredClone(s),
          inspect = createCommandInspector(s);
        for (const record of s.deaths) {
          const prefix: Command = { type: 'skill', unitId: mage.id, deathId: record.id };
          const expected = ALL_CELLS.flatMap((point) => {
            const inspection = inspectCommand(s, { ...prefix, ...point });
            return inspection.status === 'invalid' ? [] : [{ point, inspection }];
          });
          assert.deepEqual(
            inspect.points(prefix),
            expected,
            `${owner}:${size}:${range}:${record.kind}`,
          );
        }
        assert.deepEqual(s, before);
      }
});

test('小屋反应只读批次保持跳过、落点和新局面隔离', () => {
  const s = fixture(),
    hut = add(s, 'u22', 1, 4, 4);
  s.pending.push({ kind: 'hut-spawn', owner: 1, source: structuredClone(hut), amount: 0 });
  const commands: Command[] = [
    { type: 'react' },
    ...ALL_CELLS.map((p) => ({ type: 'react' as const, ...p })),
  ];
  compare(s, commands);
  add(s, 23, 1, 5, 5);
  compare(s, commands);
});

test('原生与继承牵引/献祭的参数预检和正式入口一致', () => {
  for (const ability of [7, 14] as const)
    for (const inherited of [false, true]) {
      const s = fixture(),
        u = add(s, inherited ? 4 : ability, 1, 4, 4);
      u.traits = [ability];
      u.charge = u.readyCharge = 5;
      u.chargeType = 'attack';
      add(s, 5, 2, 5, 4);
      add(s, 1, 2, 3, 4);
      add(s, 14, 1, 4, 5);
      const commands: Command[] = [];
      for (const t of s.units) {
        const base: Command = { type: 'skill', ability, unitId: u.id, targetId: t.id };
        if (ability === 7) for (const p of ALL_CELLS) commands.push({ ...base, ...p });
        else {
          commands.push({ ...base, mode: 'summon' });
          for (let column = 0; column <= 10; column++) commands.push({ ...base, column });
        }
      }
      compare(s, commands);
    }
});

test('攻击预检保留随机边界，不消耗正式随机数', () => {
  const s = fixture(),
    u = add(s, 'u1', 1, 4, 4),
    t = add(s, 'u18', 2, 4, 5);
  u.charge = u.readyCharge = 1;
  u.chargeType = 'attack';
  const c: Command = { type: 'attack', unitId: u.id, targetId: t.id };
  compare(s, [c]);
  assert.equal(inspectCommand(imagined(observe(s, 1)), c).status, 'uncertain');
});

test('攻击窄预检仅直接通过无有效标记和随机被动的友方治疗', () => {
  for (const variant of [
    'plain',
    'self',
    'expired-mark',
    'future-mark',
    'mark-owner',
    'mark-enemy',
    's3',
    's12',
    'silenced-passives',
    'enemy',
  ]) {
    const s = fixture(),
      source = add(s, 2, 1, 4, 4),
      victim = add(s, 26, variant === 'enemy' ? 2 : 1, 4, 5);
    victim.hp = 1;
    if (variant.includes('mark'))
      victim.effects.push({
        type: 'mark',
        owner: variant === 'mark-enemy' ? 2 : 1,
        from: variant === 'future-mark' ? s.ply + 1 : s.ply - 1,
        until: variant === 'expired-mark' ? s.ply : s.ply + 2,
        global: true,
      });
    if (variant === 's3' || variant === 's12') source.traits = [variant];
    if (variant === 'silenced-passives') {
      source.traits = ['s4', 's3', 's12'];
      source.silenced = true;
    }
    const command: Command = {
      type: 'attack',
      unitId: source.id,
      targetId: variant === 'self' ? source.id : victim.id,
      mode: 'heal',
    };
    const before = structuredClone(s);
    assert.equal(
      attackInspectionGuard(s, command, () => false),
      ['plain', 'self', 'expired-mark', 'future-mark', 'silenced-passives'].includes(variant)
        ? 'available'
        : undefined,
      variant,
    );
    compare(s, [command, { ...command, mode: 'invalid' }, { ...command, targetId: 'missing' }]);
    assert.deepEqual(s, before, variant);
  }
});

test('攻击窄预检仅在无处决的非穿透伤害分布首次抽样处直接停止', () => {
  for (const variant of [
    'normal',
    'ultimate',
    'vampire-guaranteed',
    'inherited',
    'expired-execute',
    'execute',
    'silenced',
    'piercing',
    's3-only',
    's12-only',
  ]) {
    const s = fixture(),
      source = add(
        s,
        variant === 'ultimate'
          ? 'u1'
          : variant === 'vampire-guaranteed'
            ? 'u8'
            : ['inherited', 's3-only', 's12-only'].includes(variant)
              ? 26
              : 1,
        1,
        4,
        4,
      ),
      victim = add(s, 26, 2, 4, 5);
    source.charge = source.readyCharge = 2;
    source.chargeType = 'attack';
    if (variant === 'vampire-guaranteed') source.kills = 5;
    if (variant === 'inherited') source.traits = ['u27', 1, 'u8'];
    if (variant === 's3-only') source.traits = ['s3'];
    if (variant === 's12-only') source.traits = ['s12'];
    if (variant.includes('execute'))
      source.effects.push({
        type: 'execute',
        owner: 1,
        from: s.ply - 1,
        until: variant === 'expired-execute' ? s.ply : s.ply + 2,
        global: true,
      });
    if (variant === 'silenced') source.silenced = true;
    if (variant === 'piercing') source.equipment = ['u28'];
    const command: Command = { type: 'attack', unitId: source.id, targetId: victim.id };
    const before = structuredClone(s);
    assert.equal(
      attackInspectionGuard(s, command, () => false),
      ['normal', 'ultimate', 'vampire-guaranteed', 'inherited', 'expired-execute'].includes(variant)
        ? 'uncertain'
        : undefined,
      variant,
    );
    compare(s, [command, { ...command, mode: 'invalid' }, { ...command, targetId: 'missing' }]);
    assert.deepEqual(s, before, variant);
  }
});

test('全局确定性攻击预检保留死亡回血、标记、反伤和击杀继承的正式后果', () => {
  for (const variant of ['death-heal', 'mark', 'reflect', 'steal']) {
    const s = fixture(),
      source = add(s, 26, 1, 4, 4),
      victim = add(s, 26, 2, 4, 5),
      friend = add(s, 26, 2, 5, 5);
    source.hp = friend.hp = 1;
    victim.hp = 1;
    if (variant === 'death-heal') victim.traits = ['sage'];
    if (variant === 'mark') {
      victim.hp = victim.maxHp = 200;
      victim.effects.push({
        type: 'mark',
        owner: 1,
        from: s.ply - 1,
        until: s.ply + 2,
        global: true,
      });
    }
    if (variant === 'reflect') {
      victim.traits = ['slayer'];
      victim.hp = victim.maxHp = 200;
    }
    if (variant === 'steal') {
      source.traits = ['s5'];
      victim.traits = ['u20'];
      victim.silenced = true;
    }
    const command: Command = { type: 'attack', unitId: source.id, targetId: victim.id };
    assert.equal(attackInspectionGuard(s, command), 'available', variant);
    compare(s, [command, { ...command, mode: 'invalid' }, { ...command, targetId: 'missing' }]);
    const after = applyCommand(s, command),
      attacker = after.units.find((unit) => unit.id === source.id);
    if (variant === 'death-heal') {
      assert.ok(attacker && attacker.hp > source.hp);
      assert.equal(after.units.find((unit) => unit.id === friend.id)?.hp, friend.maxHp);
    }
    if (variant === 'mark') {
      assert.ok(after.events.some((event) => event.action === 'mark' && event.stage === 'trigger'));
      assert.equal(after.units.find((unit) => unit.id === victim.id)?.effects.length, 0);
    }
    if (variant === 'reflect') assert.equal(attacker, undefined);
    if (variant === 'steal') assert.ok(attacker?.traits?.includes('u20'));
  }
});

test('全局确定性攻击预检遇到任一沉默随机后果本体或继承能力仍退回模拟', () => {
  const excluded: Kind[] = ['s3', 's12', '17p', 20, 'u18'];
  for (const kind of excluded)
    for (const location of ['native', 'trait', 'landmark']) {
      const s = fixture(),
        source = add(s, 26, 1, 4, 4),
        victim = add(s, 26, 2, 4, 5),
        remote = add(
          s,
          location === 'native' ? kind : location === 'landmark' ? 's1' : 26,
          2,
          8,
          10,
        );
      remote.silenced = true;
      if (location !== 'native') remote.traits = [kind];
      if (location === 'landmark') {
        s.units.pop();
        s.landmarks = [{ ...remote, hp: 0, dormantSince: s.ply }];
      }
      const command: Command = { type: 'attack', unitId: source.id, targetId: victim.id };
      assert.equal(attackInspectionGuard(s, command), undefined, `${kind}/${location}`);
      compare(s, [command]);
    }
});

test('全局确定性攻击预检允许其他棋子持有暴击，强夺后也不重复本次抽样', () => {
  for (const kind of [1, 'u1', 'u8'] as const)
    for (const location of ['native', 'trait', 'landmark', 'stolen']) {
      const s = fixture(),
        source = add(s, 26, 1, 4, 4),
        victim = add(s, 26, 2, 4, 5),
        remote = add(
          s,
          location === 'native' ? kind : location === 'landmark' ? 's1' : 26,
          2,
          8,
          10,
        );
      if (location !== 'native') remote.traits = [kind];
      if (location === 'landmark') {
        s.units.pop();
        s.landmarks = [{ ...remote, hp: 0, dormantSince: s.ply }];
      }
      if (location === 'stolen') {
        source.traits = ['s5'];
        victim.traits = [kind];
        victim.hp = 1;
      }
      const command: Command = { type: 'attack', unitId: source.id, targetId: victim.id };
      assert.equal(attackInspectionGuard(s, command), 'available', `${kind}/${location}`);
      compare(s, [command, { ...command, mode: 'invalid' }]);
      const after = applyCommand(s, command);
      assert.equal(after.rng, s.rng, `${kind}/${location}`);
      if (location === 'stolen')
        assert.ok(after.units.find((unit) => unit.id === source.id)?.traits?.includes(kind));
    }
});

test('全局确定性攻击预检仍排除来源暴击本体或继承，沉默来源也回退模拟', () => {
  for (const kind of [1, 'u1', 'u8'] as const)
    for (const inherited of [false, true])
      for (const silenced of [false, true]) {
        const s = fixture(),
          source = add(s, inherited ? 26 : kind, 1, 4, 4),
          victim = add(s, 26, 2, 4, 5);
        source.charge = source.readyCharge = 5;
        source.chargeType = 'attack';
        source.silenced = silenced;
        if (inherited) source.traits = [kind];
        const command: Command = { type: 'attack', unitId: source.id, targetId: victim.id };
        assert.equal(
          attackInspectionGuard(s, command),
          silenced ? undefined : 'uncertain',
          `${kind}/${inherited}/${silenced}`,
        );
        compare(s, [command, { ...command, targetId: 'missing' }]);
      }
});

test('攻击预检复用原始属性，保留普通首末次、额外攻击和错误顺序', () => {
  for (const stage of ['first', 'last', 'bonus'] as const) {
    const s = fixture(),
      source = add(s, 9, 1, 4, 4),
      victim = add(s, 26, 2, 4, 5),
      actions = getStats(s, source).actions;
    assert.ok(actions > 1);
    if (stage === 'last') {
      source.mode = 'attack';
      source.shots = actions - 1;
    } else source.shots = 9;
    if (stage === 'bonus') {
      source.operations = 1;
      source.bonusAttacks = 1;
    }
    const command: Command = { type: 'attack', unitId: source.id, targetId: victim.id };
    compare(s, [
      command,
      { ...command, mode: 'invalid' },
      {
        ...command,
        targetId: 'missing',
        path: [
          { x: 4, y: 4 },
          { x: 4, y: 5 },
        ],
      },
    ]);
    const after = applyCommand(s, command),
      updated = after.units.find((unit) => unit.id === source.id)!;
    assert.deepEqual(updated.attacked, [victim.id]);
    if (stage === 'last') {
      assert.equal(updated.mode, 'none');
      assert.equal(updated.shots, 0);
      assert.equal(updated.operations, 1);
    } else {
      assert.equal(updated.mode, 'attack');
      assert.equal(updated.shots, 1);
      assert.equal(getStats(after, updated).remaining, actions - 1);
      if (stage === 'bonus') {
        assert.equal(updated.bonusAttacks, 0);
        assert.equal(updated.bonusSequence, true);
        assert.equal(updated.operations, 1);
      }
    }
  }
});

test('攻击预检完整保留反击死亡与蓄力穿透，输入和原始事件序号独立', () => {
  const retaliation = fixture(),
    fragile = add(retaliation, 26, 1, 4, 4),
    king = add(retaliation, 'u18', 2, 4, 5);
  fragile.hp = 1;
  const counter: Command = { type: 'attack', unitId: fragile.id, targetId: king.id };
  compare(retaliation, [counter]);
  const countered = applyCommand(retaliation, counter);
  assert.equal(
    countered.units.some((unit) => unit.id === fragile.id),
    false,
  );
  assert.equal(countered.events.filter((event) => event.type === 'attack').length, 2);
  assert.ok(
    countered.events.some((event) => event.type === 'death' && event.unitId === fragile.id),
  );

  const piercing = fixture(),
    cannon = add(piercing, 15, 1, 4, 4),
    front = add(piercing, 26, 2, 4, 5),
    back = add(piercing, 26, 2, 4, 6);
  cannon.equipment = ['u28'];
  cannon.charge = cannon.readyCharge = 4;
  cannon.chargeType = 'attack';
  front.hp = front.maxHp = back.hp = back.maxHp = 1000;
  const shot: Command = {
    type: 'attack',
    unitId: cannon.id,
    targetId: back.id,
    path: [
      { x: 4, y: 4 },
      { x: 4, y: 5 },
      { x: 4, y: 6 },
    ],
  };
  compare(piercing, [
    shot,
    {
      ...shot,
      path: [
        { x: 4, y: 4 },
        { x: 4, y: 6 },
      ],
    },
  ]);
  const after = applyCommand(piercing, shot),
    updated = after.units.find((unit) => unit.id === cannon.id)!;
  assert.equal(updated.charge, 0);
  assert.equal(updated.readyCharge, 0);
  assert.deepEqual(updated.attacked, [back.id, front.id]);
  assert.ok(after.units.find((unit) => unit.id === front.id)!.hp < 1000);
  assert.ok(after.units.find((unit) => unit.id === back.id)!.hp < 1000);
  assert.deepEqual(
    after.events.map((event) => event.id),
    after.events.map((_, index) => `e${piercing.serial + index}`),
  );
});

test('攻击预检保留策反与先攻的首次随机边界，不因省略计数收尾而越过', () => {
  for (const ability of ['s3', 's12'] as const) {
    const s = fixture(),
      source = add(s, 26, 1, 4, 4),
      victim = add(s, 26, 2, 4, 5);
    source.traits = [ability];
    const before = structuredClone(s),
      command: Command = { type: 'attack', unitId: source.id, targetId: victim.id },
      boundary = Symbol('首次随机边界');
    let requests = 0;
    assert.throws(
      () =>
        applyCommand(s, command, () => {
          requests++;
          throw boundary;
        }),
      (error) => error === boundary,
    );
    assert.equal(requests, 1);
    compare(s, [command]);
    assert.deepEqual(createCommandInspector(s)(command), { status: 'uncertain' });
    assert.deepEqual(s, before);
  }
});

test('攻击投影属性只属于原预检批次，击杀成长和失败模拟不污染其他目标或新局面', () => {
  const s = fixture(),
    attacker = add(s, 26, 1, 4, 4),
    victim = add(s, 1, 2, 4, 5),
    runner = add(s, 20, 2, 5, 4);
  attacker.kills = 1;
  attacker.extraOperations = 1;
  victim.hp = 1;
  runner.hp = getStats(s, attacker).attack + 1;
  const kill: Command = { type: 'attack', unitId: attacker.id, targetId: victim.id },
    probe: Command = { ...kill, targetId: runner.id };
  let attributeReads = 0;
  const before = structuredClone(s),
    inspect = createCommandInspector(s, (position, unit) => {
      attributeReads++;
      assert.equal(position, s);
      assert.equal(unit, attacker, '共享缓存只读取原实体，不能用模式投影副本作键');
      return getStats(position, unit);
    });
  compare(s, [kill, probe, { ...probe, direction: 'up' }, { ...kill, targetId: 'missing' }, probe]);
  for (const command of [
    { ...probe, mode: 'invalid' },
    { ...kill, targetId: 'missing' },
    { ...probe, direction: 'up' as const },
  ])
    assert.deepEqual(inspect(command), inspectCommand(s, command));
  assert.equal(inspect(probe).status, 'available');
  assert.equal(inspect(kill).status, 'available');
  assert.equal(inspect(probe).status, 'available');
  assert.equal(attributeReads, 1, '原实体资格与模式准备只读取一次属性');
  const next = applyCommand(s, kill);
  assert.equal(next.units.find((u) => u.id === attacker.id)!.attackBonus, attacker.attackBonus + 5);
  let nextReads = 0;
  const nextInspect = createCommandInspector(next, (position, unit) => {
    nextReads++;
    assert.equal(position, next);
    assert.notEqual(unit, attacker);
    return getStats(position, unit);
  });
  assert.equal(nextInspect(probe).status, 'uncertain');
  assert.equal(nextReads, 1);
  assert.equal(inspect(probe).status, 'available');
  assert.equal(attributeReads, 1);
  assert.deepEqual(s, before);
});

test('区域法术所有落点保留空地、全栈与2×2覆盖的完整结算预检结果', () => {
  const s = fixture();
  add(s, 5, 1, 2, 5);
  add(s, 5, 2, 6, 7);
  add(s, 'u12', 2, 6, 7).hp = 1;
  const bomb = card(s, 8),
    storm = card(s, 'u9');
  compare(s, [
    ...ALL_CELLS.map((p) => ({ type: 'cast' as const, cardId: bomb, ...p })),
    ...Array.from({ length: 15 }, (_, row) => ({
      type: 'cast' as const,
      cardId: storm,
      mode: 'row',
      row,
    })),
    ...Array.from({ length: 11 }, (_, column) => ({
      type: 'cast' as const,
      cardId: storm,
      mode: 'column',
      column,
    })),
    { type: 'cast', cardId: bomb, x: 1, y: 4, targetId: 'missing' },
    { type: 'cast', cardId: 'missing', x: 1, y: 4 },
  ]);
  assert.equal(inspectCommand(s, { type: 'cast', cardId: bomb, x: 1, y: 4 }).status, 'available');
  const pointInspector = createCommandInspector(s);
  assert.deepEqual(
    pointInspector.points({ type: 'cast', cardId: bomb }),
    ALL_CELLS.flatMap((point) => {
      const inspection = pointInspector({ type: 'cast', cardId: bomb, ...point });
      return inspection.status === 'invalid' ? [] : [{ point, inspection }];
    }),
  );
});

test('施法参数先于反制随机边界，沉默与冻结的原生/继承反制资格一致', () => {
  for (const kind of ['u3', 'archmage', 's5'] as const) {
    const s = fixture(),
      mage = add(s, kind, 2, 8, 10),
      friend = add(s, 1, 1, 3, 4),
      big = add(s, 5, 1, 5, 4);
    if (kind === 's5') mage.traits = ['archmage'];
    const bomb = card(s, 8),
      ward = card(s, 17),
      conversion = card(s, 22),
      buff = card(s, 25);
    const commands: Command[] = [
      { type: 'cast', cardId: bomb, x: 1, y: 4 },
      { type: 'cast', cardId: bomb, x: 9, y: 13 },
      { type: 'cast', cardId: ward, targetId: friend.id },
      { type: 'cast', cardId: conversion, targetId: big.id },
      {
        type: 'cast',
        cardId: buff,
        mode: 'double',
        sacrificeIds: [friend.id, friend.id],
      },
    ];
    compare(s, commands);
    assert.equal(inspectCommand(s, commands[0]).status, 'uncertain');
    mage.silenced = true;
    compare(s, commands);
    assert.equal(inspectCommand(s, commands[0]).status, 'available');
    mage.silenced = false;
    addEffect(s, mage, 'freeze', 1, 0, 2);
    compare(s, commands);
    assert.equal(inspectCommand(s, commands[0]).status, kind === 'u3' ? 'uncertain' : 'available');
  }
});

test('纯附魔、随机抽牌和阶段切换的快速预检沿用完整规则结果', () => {
  const s = fixture(),
    friend = add(s, 1, 1, 3, 4),
    enemy = add(s, 1, 2, 5, 6);
  const spells = [17, 18, 22, 25].map((kind) => card(s, kind as 17 | 18 | 22 | 25));
  compare(
    s,
    spells.flatMap((cardId) => [
      { type: 'cast', cardId, targetId: friend.id },
      { type: 'cast', cardId, targetId: enemy.id },
      { type: 'cast', cardId, targetId: 'missing' },
    ]),
  );
  for (const phase of ['summon', 'synthesis', 'play'] as const) {
    s.phase = phase;
    for (const slots of [0, 1]) {
      s.summonSlots = slots;
      compare(s, [{ type: 'begin' }, { type: 'skip-synthesis' }]);
    }
  }
});

test('纯增益和冰痕技能的原生与继承预检保留完整资格及落点', () => {
  for (const ability of [6, 19, 'u21', 'u24'] as const)
    for (const inherited of [false, true]) {
      const s = fixture(),
        u = add(s, inherited ? 4 : ability, 1, 4, 4),
        friend = add(s, 1, 1, 5, 4),
        enemy = add(s, 1, 2, 5, 6);
      if (inherited) u.traits = [ability];
      const base: Command = { type: 'skill', unitId: u.id, ability };
      const commands =
        ability === 'u24' || ability === 19
          ? ALL_CELLS.map((p) => ({ ...base, ...p }))
          : ability === 'u21'
            ? [friend.id, enemy.id, 'missing'].map((targetId) => ({ ...base, targetId }))
            : [base];
      compare(s, commands);
      u.operations = 1;
      compare(s, commands);
    }
});

test('复活仅在支付后必定存活时省去模拟，最后生命保留继承死亡随机边界', () => {
  const s = fixture(),
    u = add(s, 'u19', 1, 4, 4);
  u.traits = [20];
  s.deaths = [{ id: 'victim', kind: 1, owner: 1, ply: s.ply - 1, revived: false }];
  const command: Command = { type: 'skill', unitId: u.id, deathId: 'victim', x: 5, y: 4 };
  for (const maxHp of [65, 25, 20, 5]) {
    u.hp = u.maxHp = maxHp;
    compare(s, [command]);
    assert.equal(inspectCommand(s, command).status, maxHp > 20 ? 'available' : 'uncertain');
  }
  u.hp = u.maxHp = 65;
  u.bannerHp = 60;
  compare(s, [command]);
  assert.equal(inspectCommand(s, command).status, 'uncertain');
});

test('装备只读准备与完整结算的资格、低上限及错误顺序一致', () => {
  const s = fixture(),
    plain = add(s, 1, 1, 3, 4),
    mage = add(s, 6, 1, 4, 4),
    enemy = add(s, 1, 2, 6, 6),
    loner = add(s, 10, 1, 2, 5),
    grave = add(s, 'grave', 1, 6, 5);
  plain.hp = plain.maxHp = 1;
  const cards = ['u5', 'u11', 'u28', 's15', 's16'].map((kind) => card(s, kind as 'u5'));
  const commands: Command[] = cards.flatMap((cardId) =>
    [plain.id, mage.id, enemy.id, loner.id, grave.id, 'missing'].map((targetId) => ({
      type: 'equip',
      cardId,
      targetId,
    })),
  );
  compare(s, commands);
  plain.chargedOnDeploy = true;
  plain.deployedAt = s.ply;
  compare(s, commands);
});
