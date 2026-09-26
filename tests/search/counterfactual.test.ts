import assert from 'node:assert/strict';
import test from 'node:test';
import { createGame } from '../../src/engine';
import { observe } from '../../src/ai/observation';
import { rollout, pairedRanking } from '../../scripts/training/counterfactual/rollout';
import { positions } from '../../scripts/training/search/positions';
import { decide } from '../../src/ai/planning/search';

test('完整窗口跨召唤和对手行动，未知不混入排名，已完成配对不因后来截断而丢失', () => {
  const s = createGame(3);
  Object.assign(s, { phase: 'play', summonSlots: 0, hands: { 1: [], 2: [] } });
  const observation = observe(s);
  const job = {
    observation,
    command: { type: 'end' } as const,
    sample: 23,
    maxCommands: 200,
    maxWork: 8000,
  };
  const row = rollout(job);
  assert.equal(row.stop, 'boundary');
  assert.equal(row.ply, observation.ply + 2);
  assert.equal(row.pending, 0);
  assert.ok(row.steps.some((step) => step.command.type === 'summon'));
  const again = rollout(job);
  assert.deepEqual(again.steps, row.steps);
  assert.equal(again.final, row.final);
  const unknown = rollout({ ...job, maxCommands: 1 });
  assert.equal(unknown.stop, 'command-limit');
  assert.equal(unknown.heuristic, null);
  assert.equal(unknown.value, null);
  assert.deepEqual(
    pairedRanking([
      [{ complete: true, heuristic: 2 }, unknown],
      [{ complete: true, heuristic: 4 }, row],
    ]),
    { paired: [0], scores: [2, 4], best: 1, fallback: false },
  );
  const funded = structuredClone(observation);
  funded.heads = { 1: 100, 2: 100 };
  const atomic = rollout({ ...job, observation: funded });
  assert.equal(atomic.stop, 'boundary', '召唤完整枚举可以越过请求预算，必须按实际工作量记账');
  assert.ok(atomic.steps.some((step) => step.work > 40));
});

test('战术护栏的当前教师首步在新续弈中兑现终局，实际反应方仍由引擎决定', () => {
  for (const p of positions().filter((p) =>
    ['immediate-win', 'same-player-two-attacks', 'reaction-to-play'].includes(p.family),
  )) {
    const d = decide(p.observation, p.actor, 'easy', { simulations: 40, mode: 'work' });
    assert.ok(d.command);
    const row = rollout({
      observation: p.observation,
      command: d.command,
      sample: 41,
      maxCommands: 200,
      maxWork: 8000,
    });
    assert.equal(row.stop, 'terminal', p.name);
    assert.equal(row.value, 1, p.name);
  }
});
