import assert from 'node:assert/strict';
import test from 'node:test';
import { positions } from '../../scripts/training/search/positions';
import { parallelRechecks } from '../../scripts/training/search/rechecks';
import { reference } from '../../scripts/training/search/reference';
import { search } from '../../scripts/training/search/puct';

test('独立局面复核同时启动四个进程，结果与串行同种子同预算一致', async () => {
  const observations = positions()
    .slice(0, 5)
    .map((p) => p.observation);
  const protocol = {
    horizon: 2,
    referenceMaxActions: 512,
    maxActionNodes: 512,
    searchSeeds: [2026092731],
  };
  const result = await parallelRechecks(observations, protocol);
  assert.equal(result.workers, 4);
  assert.equal(result.maxActive, 4);
  assert.equal(result.results.length, 5);
  for (const [index, o] of observations.entries()) {
    assert.deepEqual(
      result.results[index].oracle,
      reference(o, 2, { maxActions: 512, maxActionNodes: 512 }),
    );
    assert.deepEqual(
      result.results[index].repeated,
      search(o, { simulations: 128, horizon: 2, sampleSeed: 2026092731, maxActionNodes: 512 }),
    );
  }
  await assert.rejects(
    parallelRechecks(observations.slice(0, 1), { ...protocol, searchSeeds: [-1] }, 1),
    /sampleSeed 必须是 uint32/,
  );
});
