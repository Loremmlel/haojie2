import test from 'node:test';
import assert from 'node:assert/strict';
import { positions } from '../../scripts/training/search/positions';
import { search } from '../../scripts/training/search/puct';
import { terminalRollout } from '../../scripts/training/search/leaf/rollout';
import { sampleTrainingTransition } from '../../src/ai/training/simulation';

test('延迟首次枚举保持零估值搜索的命令、机会采样和访问分配，计时不影响选招', () => {
  for (const p of positions()) {
    const before = JSON.stringify(p.observation);
    const options = { simulations: 32, horizon: 2, sampleSeed: 2026092731 };
    const eager = search(p.observation, options);
    const profile = { enumerationMs: 0, transitionMs: 0, leafMs: 0, leafCalls: 0 };
    const deferred = search(p.observation, { ...options, deferExpansion: true, profile });
    assert.equal(eager.status, 'command');
    assert.equal(deferred.status, 'command');
    if (eager.status !== 'command' || deferred.status !== 'command') throw new Error(p.name);
    assert.deepEqual(deferred.command, eager.command);
    assert.deepEqual(deferred.edges, eager.edges);
    assert.equal(deferred.stats.transitions, eager.stats.transitions);
    assert.equal(deferred.stats.terminalLeaves, eager.stats.terminalLeaves);
    assert.ok(deferred.stats.actionNodes <= eager.stats.actionNodes);
    assert.ok(Object.values(profile).every((n) => Number.isFinite(n) && n >= 0));
    assert.equal(JSON.stringify(p.observation), before);
  }
});

test('叶端估计接收公开局面及固定根视角，拒绝非有限值，不改变终局真实回报', () => {
  const p = positions().find((p) => p.family === 'opponent-death-reaction')!;
  let calls = 0;
  const result = search(p.observation, {
    simulations: 128,
    horizon: 2,
    sampleSeed: 42,
    deferExpansion: true,
    leafValue: (o, actor) => {
      assert.equal(actor, p.actor);
      assert.ok(!('seed' in o) && !('rng' in o));
      calls++;
      return 0;
    },
  });
  assert.equal(result.status, 'command');
  assert.ok(calls > 0 && result.stats.terminalLeaves > 0);
  for (const value of [NaN, Infinity, 2])
    assert.equal(
      search(p.observation, { simulations: 32, horizon: 2, sampleSeed: 42, leafValue: () => value })
        .status,
      'paused',
    );
});

test('父节点初值避免高正估计把小预算全部锁在首次抽到的结束回合', () => {
  const p = positions().find((p) => p.name === 'same-player-two-attacks-p1-x5')!;
  const options = {
    simulations: 16,
    horizon: 2,
    sampleSeed: 2026092731,
    deferExpansion: true,
    leafValue: () => 0.97,
  };
  const zero = search(p.observation, options);
  const parent = search(p.observation, { ...options, firstPlayValue: 'parent' });
  if (zero.status !== 'command' || parent.status !== 'command') throw new Error('搜索未完成');
  assert.deepEqual(zero.command, { type: 'end' });
  assert.equal(zero.edges.filter((e) => e.visits).length, 1);
  assert.ok(parent.edges.filter((e) => e.visits).length > 1);
  assert.ok(parent.edges.some((e) => e.visits && e.command.type === 'attack'));
});

test('短续演按反应操作者取真实胜负，深度耗尽明确未知，随机流可复现且输入不变', () => {
  for (const p of positions().filter((p) => p.family === 'reaction-to-play' && p.x === 5)) {
    const before = structuredClone(p.observation);
    const rollout = terminalRollout(2026092731);
    assert.equal(rollout.leafValue(p.observation, p.actor, 1), 1);
    assert.equal(rollout.leafValue(p.observation, p.actor === 1 ? 2 : 1, 1), -1);
    assert.equal(rollout.leafValue(p.observation, p.actor, 0), 0);
    assert.equal(rollout.stats.terminal, 2);
    assert.equal(rollout.stats.unknown, 1);
    assert.equal(rollout.stats.reactionCalls, 2);
    assert.deepEqual(p.observation, before);
  }
  const p = positions().find((p) => p.family === 'chance-win')!;
  const a = terminalRollout(71),
    b = terminalRollout(71);
  const av = Array.from({ length: 60 }, () => a.leafValue(p.observation, p.actor, 1));
  assert.deepEqual(
    av,
    Array.from({ length: 60 }, () => b.leafValue(p.observation, p.actor, 1)),
  );
  assert.ok(av.includes(0) && av.includes(1));
  const endedWindow = sampleTrainingTransition(p.observation, p.actor, { type: 'end' }, 0);
  const calls = a.stats.teacherCalls;
  assert.equal(a.leafValue(endedWindow, p.actor, 1), 0);
  assert.equal(a.stats.teacherCalls, calls);
});
