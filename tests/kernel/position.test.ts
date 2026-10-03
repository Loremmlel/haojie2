import test from 'node:test';
import assert from 'node:assert/strict';
import { fixtures } from '../../scripts/training/native/fixtures';
import { combatFixtures } from '../../scripts/training/native/combat-fixtures';
import { completeFixtures } from '../../scripts/training/native/complete-fixtures';
import { preparationFixtures } from '../../scripts/training/native/preparation-fixtures';
import {
  applyCommand,
  applyRuntimeCommand,
  createCommandInspector,
} from '../../src/engine/commands/game';
import {
  importRuntimePosition,
  forkRuntimePosition,
  entityAt,
  entityHandle,
} from '../../src/engine/runtime/position';
import { publicPositionView, observe } from '../../src/ai/observation';
import { TrainingActionTree } from '../../src/ai/training/action-tree';
import { createDecisionEncoder } from '../../src/ai/training/encoding/decision';
import { createPlacementQuery, movementPath, ALL_CELLS } from '../../src/engine/core/geometry';
import { allPieces } from '../../src/engine/core/traits';
import type { GameState } from '../../src/engine/types';

const cases = [...fixtures(), ...combatFixtures(), ...completeFixtures(), ...preparationFixtures()];
const plain = <T>(v: T): T => JSON.parse(JSON.stringify(v));
const outcome = (fn: () => GameState) => {
  try {
    return { state: plain(fn()) };
  } catch (e) {
    return { error: (e as Error).message };
  }
};

test('实体句柄跨分支、同长度替换、删除与重排均指向当前实体', () => {
  const source = cases.find((c) => c.job.state.units.length >= 3)!.job.state;
  const s = importRuntimePosition(source),
    [a, b, c] = s.units;
  const handles = [a, b, c].map((u) => entityHandle(s, u.id)!);
  const next = forkRuntimePosition(s);
  next.units[0] = { ...next.units[0], hp: next.units[0].hp + 5 };
  assert.equal(entityAt(next, handles[0])!.hp, a.hp + 5);
  assert.equal(entityAt(s, handles[0])!.hp, a.hp);
  next.units.splice(0, 1, { ...next.units[0], id: 'kernel-replacement' });
  assert.equal(entityHandle(next, 'absent'), undefined);
  assert.equal(entityHandle(next, 'kernel-replacement') === undefined, false);
  assert.equal(entityAt(next, handles[0]), undefined);
  const replacement = entityHandle(next, 'kernel-replacement')!;
  assert.notEqual(replacement, handles[0]);
  next.units.reverse();
  assert.equal(entityAt(next, handles[1])!.id, b.id);
  assert.equal(entityAt(next, replacement)!.id, 'kernel-replacement');
  assert.equal(entityAt(s, handles[2]), c);
});

test('新运行分支在全部规则夹具保留成功、失败原子性和公开候选/编码', () => {
  for (const { name, job } of cases) {
    const original = plain(job.state);
    const runtime = importRuntimePosition(job.state);
    const before = plain(runtime);
    const oldInspect = createCommandInspector(job.state);
    const inspect = createCommandInspector(runtime);
    for (const c of job.probes) assert.deepEqual(inspect(c), oldInspect(c), name);
    if (job.command)
      assert.deepEqual(
        outcome(() => applyRuntimeCommand(runtime, job.command!)),
        outcome(() => applyCommand(job.state, job.command!)),
        name,
      );
    assert.deepEqual(plain(runtime), before, name);
    assert.deepEqual(plain(job.state), original, name);
    for (const actor of [1, 2] as const) {
      const exported = observe(runtime, actor);
      const view = publicPositionView(runtime, actor);
      assert.deepEqual(plain(view), plain(exported), name);
      assert.ok(!('rng' in view) && !('seed' in view) && !('log' in view));
      const reference = new TrainingActionTree(exported, actor).node([]);
      const actual = new TrainingActionTree(view, actor).node([]);
      assert.deepEqual(plain(actual), plain(reference), name);
      if (actual.choices.length)
        assert.deepEqual(
          createDecisionEncoder(view, actor)(actual),
          createDecisionEncoder(exported, actor)(reference),
          name,
        );
      assert.throws(() => {
        view.bases[1] = 0;
      }, TypeError);
    }
    assert.deepEqual(plain(runtime), before, name);
  }
});

test('移动可达场保留所有落点的精确路径、顺序和分数距离边界', () => {
  for (const { name, job } of cases) {
    const query = createPlacementQuery(job.state);
    for (const u of allPieces(job.state).slice(0, 3)) {
      for (const [limit, straight] of [
        [0, false],
        [1.5, false],
        [3, false],
        [5, false],
        [0, true],
      ] as const) {
        const field = query.movementField(u, limit, straight);
        for (const to of ALL_CELLS)
          assert.deepEqual(
            field(to),
            movementPath(job.state, u, to, limit, straight),
            `${name}:${u.id}:${to.x},${to.y}:${limit}:${straight}`,
          );
      }
    }
  }
});
