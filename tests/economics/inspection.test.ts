import assert from 'node:assert/strict';
import test from 'node:test';
import {
  applyCommand,
  createCommandInspector,
  inspectCommand,
} from '../../src/engine/commands/game';
import { kill } from '../../src/engine/commands/combat';
import { ALL_CELLS } from '../../src/engine/core/geometry';
import { emit, RuleError, withRuleInspection } from '../../src/engine/core/state';
import { eventActor, withEventFacts } from '../../src/engine/core/event-facts';
import { observe } from '../../src/ai/observation';
import { trainingPosition } from '../../src/ai/training/queries';
import type { Command, GameState } from '../../src/engine/types';
import { add, fixture } from '../helpers';

test('嵌套预检退出后恢复事件快照和日志，保留原始事件与正式序号', () => {
  const s = fixture(),
    u = add(s, 1, 1, 4, 4);
  const serial = s.serial,
    log = [...s.log];
  assert.throws(() =>
    withRuleInspection(() =>
      withRuleInspection(() =>
        withEventFacts(s, { actor: eventActor(u), action: 'attack' }, () => {
          emit(s, { type: 'damage', to: u, unitId: u.id, amount: 5 }, '预检');
          throw new Error('退出');
        }),
      ),
    ),
  );
  assert.equal(s.serial, serial + 1);
  assert.equal(s.events.at(-1)?.unitId, u.id);
  assert.equal(s.events.at(-1)?.amount, 5);
  assert.deepEqual(s.log, log);
  withEventFacts(s, { actor: eventActor(u), action: 'attack' }, () => {
    emit(s, { type: 'damage', to: u, unitId: u.id, amount: 5 }, '正式');
  });
  u.x++;
  assert.equal(s.events.at(-1)?.actor?.x, 4);
  assert.equal(s.events.at(-1)?.subject?.x, 4);
  assert.equal(s.events.at(-1)?.to?.x, 4);
  assert.equal(s.events.at(-1)?.id, `e${serial + 1}`);
  assert.equal(s.log.at(-1), `${s.ply} · 正式`);
});

/** 使用正式命令入口作为参照；遇到随机请求即中断，不编造随机结果。 */
function compare(s: GameState, commands: Command[]) {
  const unknown = Symbol('随机边界');
  const position = trainingPosition(observe(s, s.active));
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
  assert.equal(inspectCommand(trainingPosition(observe(s, 1)), c).status, 'uncertain');
});
