import assert from 'node:assert/strict';
import test from 'node:test';
import { decoderFrontier } from '../../src/ai/classic/decoder';
import { observe } from '../../src/ai/observation';
import { preparationFixtures } from '../../scripts/native/engine/fixtures/preparation';
import { fixtures } from '../../scripts/native/engine/fixtures/movement';
import { applyCommand, inspectCommand } from '../../src/engine/commands/game';
import type { Command } from '../../src/engine/types';

test('冲撞解码排除随机免疫后无处落脚的命令，保留有余步的冲撞', () => {
  for (const moves of [1, 2]) {
    const { job } = fixtures().find((f) => f.name === `rush-random-immunity-${moves}`)!;
    const before = structuredClone(job.state);
    const command: Command = { type: 'move', unitId: job.state.units[0].id, x: 4, y: 7 };
    assert.equal(inspectCommand(job.state, command).status, 'uncertain');
    assert.doesNotThrow(() => applyCommand(job.state, command, () => 0.99));
    if (moves === 1) assert.throws(() => applyCommand(job.state, command, () => 0), /剩余移动次数/);
    else assert.doesNotThrow(() => applyCommand(job.state, command, () => 0));
    const observation = observe(job.state);
    const root = decoderFrontier(observation, []);
    const source = root.choices!.findIndex((c) => c.unitId === command.unitId);
    const actions = decoderFrontier(observation, [source]);
    const action = actions.choices!.findIndex((c) => c.type === 'move');
    const points = decoderFrontier(observation, [source, action]);
    assert.equal(
      points.choices!.some((c) => c.x === 4 && c.y === 7),
      moves === 2,
    );
    assert.deepEqual(job.state, before);
  }
});

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
