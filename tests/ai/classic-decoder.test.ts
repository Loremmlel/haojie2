import assert from 'node:assert/strict';
import test from 'node:test';
import { decoderFrontier } from '../../src/ai/classic/decoder';
import { observe } from '../../src/ai/observation';
import { preparationFixtures } from '../../scripts/native/engine/fixtures/preparation';

test('同一改判来源可以选择任意一张当回合手牌', () => {
  const { job } = preparationFixtures().find((f) => f.name.endsWith('reroll-multiple-cards'))!;
  const observation = observe(job.state),
    root = decoderFrontier(observation, []),
    source = root.choices!.findIndex((c) => c.unitId === job.probes[0].unitId);
  assert.ok(source >= 0);
  assert.deepEqual(root.choices![source], { type: '', unitId: job.probes[0].unitId });
  const actions = decoderFrontier(observation, [source]);
  for (const command of job.probes) {
    const action = actions.choices!.findIndex((c) => c.cardId === command.cardId);
    assert.ok(action >= 0);
    assert.deepEqual(decoderFrontier(observation, [source, action]).command, command);
  }
});
