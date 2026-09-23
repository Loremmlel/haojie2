import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { applyCommand, createGame, parseSession } from '../../src/engine';
import { TrainingEnvironment } from '../../src/match/training';
import { observe, fingerprint } from '../../src/ai/observation';
import {
  trainingActionSpace,
  trainingGeometry,
  inspectTrainingCommand,
} from '../../src/ai/training/queries';
import { sampleTrainingTransition, trainingDistribution } from '../../src/ai/training/simulation';
import { add, card, fixture } from '../helpers';
import type { Command } from '../../src/engine';

test('训练环境逐步复现当前79条正式命令，公开指纹、身份与实际随机结果不漂移', () => {
  const [header, ...rows] = readFileSync(
    'docs/playtests/current/cli-feedback5-20260923.jsonl',
    'utf8',
  )
    .trim()
    .split('\n')
    .map((line) => JSON.parse(line));
  let state = parseSession(JSON.stringify(header.initial)).present;
  const env = TrainingEnvironment.fromState(state);
  for (const row of rows) {
    assert.equal(fingerprint(env.observation()), row.before);
    env.step(row.owner, row.command);
    state = applyCommand(state, row.command);
    assert.equal(fingerprint(env.observation()), row.after);
    assert.deepEqual(env.observation(1), observe(state, 1));
    assert.deepEqual(env.observation(2), observe(state, 2));
  }
  assert.equal(env.status().commands, 79);
});

test('非法训练命令不付费、不耗随机数、不增加步数；观察副本不能修改实局', () => {
  const env = new TrainingEnvironment({ seed: 7 });
  const before = env.frame();
  assert.throws(() => env.step(2, { type: 'summon' }));
  assert.throws(() => env.step(1, { type: 'summon', ultimate: true }));
  assert.throws(() => env.step(1, { type: 'move', x: 999 }));
  assert.deepEqual(env.frame(), before);
  const copy = env.observation();
  copy.bases[1] = 1;
  assert.equal(env.observation().bases[1], 300);
  env.step(1, { type: 'summon' });
  assert.deepEqual(env.observation(), observe(applyCommand(createGame(7), { type: 'summon' })));
});

test('训练截断不当成平局或血量胜负，终局优先于步数上限', () => {
  const limited = new TrainingEnvironment({ maxCommands: 1 });
  const status = limited.step(1, { type: 'summon' });
  assert.equal(status.truncated, true);
  assert.equal(status.terminated, false);
  assert.equal(status.returns, null);
  assert.equal(status.toPlay, null);
  assert.throws(() => limited.step(1, { type: 'summon' }), /截断/);
  const s = fixture();
  const cannon = add(s, 9, 1, 5, 10);
  s.bases[2] = 1;
  const env = TrainingEnvironment.fromState(s, { maxCommands: 1 });
  const ended = env.step(1, { type: 'attack', unitId: cannon.id, targetId: 'base-2' });
  assert.equal(ended.terminated, true);
  assert.equal(ended.truncated, false);
  assert.deepEqual(ended.returns, { 1: 1, 2: -1 });
});

test('暗选观察按席位脱敏，公开预检不依赖对方提交，模拟明确拒绝补造秘密', () => {
  const env = new TrainingEnvironment({ seed: 90, rules: 'shrine' });
  const menu = trainingActionSpace(env.observation(1), 1);
  assert.equal(menu.actions.filter((a) => a.command.shrineKind === 's9').length, 2);
  const choice = { type: 'choose-shrine', shrineKind: 's9', parity: 'even' };
  assert.equal(inspectTrainingCommand(env.observation(1), 1, choice).status, 'available');
  assert.equal(
    inspectTrainingCommand(env.observation(1), 1, { ...choice, player: 2 }).status,
    'invalid',
  );
  env.step(1, { type: 'choose-shrine', shrineKind: 's9', parity: 'even' });
  const o = env.observation(2);
  assert.equal(o.shrineDraft!.choices[1], undefined);
  assert.equal(env.observation(1).shrineDraft!.choices[1]!.parity, 'even');
  const command = {
    type: 'choose-shrine',
    player: 2,
    shrineKind: o.shrineDraft!.offers[2][0],
  } as const;
  assert.equal(inspectTrainingCommand(o, 2, command).status, 'available');
  assert.throws(() => sampleTrainingTransition(o, 2, command, 12), /暗选/);
  env.step(2, command);
  assert.equal(env.observation(2).shrineDraft!.revealed, true);
});

test('反应由实际拥有者操作；回合外巨大化不被主回合锁阻拦', () => {
  const s = fixture();
  const attacker = add(s, 26, 1, 3, 4);
  const victim = add(s, 2, 2, 3, 5);
  victim.hp = 1;
  const env = TrainingEnvironment.fromState(s);
  env.step(1, { type: 'attack', unitId: attacker.id, targetId: victim.id });
  assert.equal(env.status().toPlay, 2);
  assert.equal(env.observation().active, 1);
  assert.throws(() => env.step(1, { type: 'react' }));
  env.step(2, { type: 'react' });
  const giantState = fixture();
  const giant = add(giantState, 'u7', 2, 8, 8);
  const target = add(giantState, 1, 1, 3, 5);
  const g = TrainingEnvironment.fromState(giantState);
  const cmd: Command = { type: 'skill', unitId: giant.id, targetId: target.id, x: 3, y: 5 };
  assert.ok(trainingActionSpace(g.observation(2), 2).actions.some((a) => a.id === 'giant'));
  g.step(2, cmd);
  assert.equal(g.observation().units.find((u) => u.id === target.id)!.size, 2);
});

test('未裁剪材料集合及手绘路径能通过训练接口，不依赖旧教师候选', () => {
  const s = fixture();
  s.phase = 'synthesis';
  const materialIds = [2, 3, 4, 6, 7].map((x) => add(s, 'u21', 1, x, 4).id);
  const env = TrainingEnvironment.fromState(s);
  const action = trainingActionSpace(env.observation(), 1).actions.find(
    (a) => a.command.recipeId === 'sage',
  )!;
  assert.deepEqual(action.materialIds, materialIds);
  const command: Command = {
    type: 'synthesize',
    recipeId: 'sage',
    materialIds: materialIds.slice(2),
  };
  const geometry = trainingGeometry(env.observation(), command);
  assert.ok(geometry.points?.length);
  env.step(1, { ...command, ...geometry.points[0] });
  assert.ok(env.observation().units.some((u) => u.kind === 'sage'));

  const heartState = fixture();
  const carrier = add(heartState, 26, 1, 2, 5);
  const first = add(heartState, 1, 2, 3, 5);
  const last = add(heartState, 1, 2, 4, 6);
  const outside = add(heartState, 1, 2, 5, 6);
  const weapon = card(heartState, 'u28');
  const heart = TrainingEnvironment.fromState(heartState);
  heart.step(1, { type: 'equip', cardId: weapon, targetId: carrier.id });
  assert.ok(
    trainingActionSpace(heart.observation(), 1).actions.some((a) =>
      a.steps.some((v) => v.kind === 'path'),
    ),
  );
  heart.step(1, {
    type: 'attack',
    unitId: carrier.id,
    path: [
      { x: 2, y: 5 },
      { x: 3, y: 5 },
      { x: 3, y: 6 },
      { x: 4, y: 6 },
    ],
  });
  const units = heart.observation().units;
  assert.equal(units.find((u) => u.id === first.id)!.hp, 30);
  assert.equal(units.find((u) => u.id === last.id)!.hp, 30);
  assert.equal(units.find((u) => u.id === outside.id)!.hp, 50);
});

test('独立模拟可复现真实概率边界，不能读取实局随机数或修改观察', () => {
  const s = fixture();
  const attacker = add(s, 1, 1, 3, 4);
  const victim = add(s, 1, 2, 3, 5);
  victim.hp = 30;
  const command: Command = { type: 'attack', unitId: attacker.id, targetId: victim.id };
  const o = observe(s);
  const before = structuredClone(o);
  const result = trainingDistribution(o, 1, command, 7);
  const kill = result.outcomes
    .filter((v) => !v.observation.units.some((u) => u.id === victim.id))
    .reduce((n, v) => n + v.weight, 0);
  assert.equal(result.sampled, false);
  assert.ok(Math.abs(kill - 1 / 3) < 1e-9);
  const sample = sampleTrainingTransition(o, 1, command, 7);
  s.rng = 123456;
  s.seed = 54321;
  assert.deepEqual(sampleTrainingTransition(observe(s), 1, command, 7), sample);
  assert.deepEqual(o, before);
  assert.equal('rng' in sample, false);
  assert.throws(() => sampleTrainingTransition(s, 1, command, 7), /禁止携带/);
});
