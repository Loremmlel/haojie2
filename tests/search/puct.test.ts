import test from 'node:test';
import assert from 'node:assert/strict';
import { positions } from '../../scripts/training/search/positions';
import { search, commands } from '../../scripts/training/search/puct';
import { reference } from '../../scripts/training/search/reference';
import { sampleTrainingTransition } from '../../src/ai/training/simulation';
import { decisionOwner } from '../../src/ai/observation';
import { add, fixture } from '../helpers';
import { observe } from '../../src/ai/observation';

test('完整命令PUCT保留同方连击、对手反应最小化及机会概率，不向输入写状态', () => {
  for (const p of positions().filter((p) => p.x === 5)) {
    const before = structuredClone(p.observation);
    const oracle = reference(p.observation, 2);
    const expected =
      p.family === 'opponent-death-reaction' ? 0 : p.family === 'chance-win' ? 1 / 3 : 1;
    assert.ok(Math.abs(oracle.best - expected) < 1e-9, p.name);
    const options = { simulations: 128, horizon: 2, sampleSeed: 42 };
    const result = search(p.observation, options);
    if (result.status !== 'command') throw new Error(result.reason);
    const selected = oracle.rootValues.find(
      (r) => JSON.stringify(r.command) === JSON.stringify(result.command),
    )!;
    assert.ok(Math.abs(selected.value - oracle.best) < 1e-9, p.name);
    assert.equal(result.stats.simulations, 128);
    assert.equal(
      result.edges.reduce((sum, e) => sum + e.visits, 0),
      128,
    );
    assert.ok(result.stats.transitions >= 128 && result.stats.transitions <= 256);
    assert.equal(result.stats.networkCalls, 0);
    assert.deepEqual(search(p.observation, options), result);
    assert.deepEqual(p.observation, before);
    if (p.family === 'same-player-two-attacks') {
      const next = sampleTrainingTransition(p.observation, p.actor, result.command, 0);
      assert.equal(decisionOwner(next), p.actor);
      assert.equal(next.winner, undefined);
    }
    if (p.family === 'opponent-death-reaction') {
      assert.ok(
        oracle.rootValues.some((r) => r.value === -1),
        '参照必须看到对方反应必败，而非逐边误取负',
      );
      assert.ok(result.stats.changedActorEdges > 0);
    }
    if (p.family === 'chance-win') {
      assert.ok(
        result.edges.some((e) => e.chanceOutcomesSeen > 1),
        '机会结果必须反复采样',
      );
    }
    if (p.family === 'reaction-to-play') assert.notEqual(p.actor, p.observation.active);
  }
});

test('零模拟基线、取消、枚举上限和信息边界显式处理，不返回半个命令', () => {
  const o = positions()[0].observation;
  const options = { simulations: 32, horizon: 2, sampleSeed: 17 };
  const zero = search(o, { ...options, simulations: 0 });
  assert.equal(zero.status, 'command');
  assert.equal(zero.stats.transitions, 0);
  if (zero.status === 'command') assert.deepEqual(zero.command, zero.baseline);
  const cancelled = new AbortController();
  cancelled.abort();
  for (const result of [
    search(o, { ...options, signal: cancelled.signal }),
    search(o, { ...options, maxActionNodes: 1 }),
    search({ ...o, rng: 7 } as typeof o, options),
  ]) {
    assert.equal(result.status, 'paused');
    assert.equal('command' in result, false);
  }
  assert.throws(() => commands(o, 1), /预算/);
  const giant = fixture();
  add(giant, 'u7', 2, 5, 7);
  assert.equal(search(observe(giant), options).status, 'paused');
  assert.equal(search({ ...o, mode: 'shrine' }, options).status, 'paused');
  assert.throws(() => search(o, { ...options, simulations: -1 }), /非负/);
});
