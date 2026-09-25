import test from 'node:test';
import assert from 'node:assert/strict';
import { createGame } from '../../src/engine';
import { observe } from '../../src/ai/observation';
import { bootstrapDecision } from '../../scripts/training/search/bootstrap/policy';
import { positions } from '../../scripts/training/search/positions';
import { searchExamples } from '../../scripts/training/search/bootstrap/encoding';
import { search } from '../../scripts/training/search/puct';

test('教师辅助搜索在宽根之前先检验已知候选，真实访问分布归一且决策可复现', () => {
  for (const p of positions().filter((p) => p.family === 'immediate-win')) {
    const before = structuredClone(p.observation);
    const d = bootstrapDecision(p.observation, 93);
    assert.equal(d.mode, 'search');
    assert.ok(d.policy!.every((e) => e.visits > 0));
    assert.equal(
      d.policy!.reduce((n, e) => n + e.probability, 0),
      1,
    );
    assert.deepEqual(d, bootstrapDecision(p.observation, 93));
    assert.deepEqual(p.observation, before);
  }
});

test('召唤完整概率分支允许超出请求40，记录真实work且回退不伪造搜索分布', () => {
  const s = createGame(19);
  s.phase = 'summon';
  s.heads[1] = 40;
  const d = bootstrapDecision(observe(s), 93);
  assert.equal(d.mode, 'teacher-fallback');
  assert.equal(d.policy, null);
  assert.ok(d.stats.teacherWork > 40 && d.stats.teacherWork <= 256);
  assert.equal(d.stats.searchSimulations, 0);
});

test('访问标签沿实际解码路径条件归一，不把其它动作前缀伪造成价值样本', () => {
  const p = positions().find((p) => p.family === 'immediate-win')!;
  const d = bootstrapDecision(p.observation, 93);
  const rows = searchExamples(p.observation, p.actor, d.command, [
    { command: d.command, visits: 12, probability: 0.75 },
    { command: { type: 'end' }, visits: 4, probability: 0.25 },
  ]);
  assert.deepEqual(rows[0].policy.filter((p) => p > 0).sort(), [0.25, 0.75]);
  assert.equal(rows.filter((r) => r.step === 0).length, 1);
  for (const row of rows.slice(1)) {
    assert.equal(row.prefixMass, 0.75);
    assert.equal(row.policy[row.selected], 1);
  }
  assert.throws(() =>
    searchExamples(p.observation, p.actor, d.command, [
      { command: d.command, visits: 16, probability: 0.5 },
    ]),
  );
});

test('外部搜索候选经过合法性校验，根覆盖不改变默认全域入口', () => {
  const p = positions()[0];
  const result = search(p.observation, {
    simulations: 16,
    horizon: 2,
    sampleSeed: 9,
    candidateCommands: () => [{ type: 'attack', unitId: 'missing', targetId: 'base-2' }],
    coverRoot: true,
  });
  assert.equal(result.status, 'paused');
});
