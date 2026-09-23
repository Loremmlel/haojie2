import assert from 'node:assert/strict';
import test from 'node:test';
import {
  ALL_CELLS,
  canPlace,
  createPlacementQuery,
  movementPath,
} from '../../src/engine/core/geometry';
import { fixture, add } from '../helpers';
import { template } from '../../src/engine/core/state';

test('批量查询与权威查询保持占位、克隆、地标、禁区和忽略列表一致', () => {
  const s = fixture();
  add(s, 'u25', 1, 3, 5);
  add(s, 'u25', 1, 3, 5);
  add(s, 23, 1, 5, 6);
  add(s, 'grave', 1, 7, 6);
  add(s, 'u7', 2, 1, 8);
  const land = add(s, 's1', 1, 4, 7);
  s.units.pop();
  s.landmarks = [land];
  add(s, 2, 1, 4, 7);
  const before = structuredClone(s),
    q = createPlacementQuery(s);
  const units = [
    ...s.units,
    template('u27', 1, 0, { x: 1, y: 1 }),
    template('s1', 2, 0, { x: 1, y: 1 }),
  ];
  for (const u of units)
    for (const p of ALL_CELLS)
      for (const deployment of [false, true])
        for (const ignore of [[], [s.units[0].id], [s.units[0].id, s.units[1].id]])
          assert.equal(q.canPlace(u, p, deployment, ignore), canPlace(s, u, p, deployment, ignore));
  assert.deepEqual(s, before);
});

test('移动索引保留完整路径和等长路径顺序', () => {
  const s = fixture(),
    u = add(s, 13, 1, 4, 5);
  add(s, 1, 2, 4, 6);
  add(s, 23, 1, 6, 6);
  const q = createPlacementQuery(s);
  for (const p of ALL_CELLS)
    for (const straight of [false, true])
      assert.deepEqual(q.movementPath(u, p, 3, straight), movementPath(s, u, p, 3, straight));
});

test('创建新批次可反映同一对象的修改，默认权威查询无旧缓存', () => {
  const s = fixture(),
    u = add(s, 1, 1, 3, 9),
    v = add(s, 1, 1, 5, 8);
  const ghost = template(1, 1, 0, { x: 7, y: 9 });
  assert.equal(createPlacementQuery(s).canPlace(ghost, ghost, true), false);
  v.y = 9;
  assert.equal(canPlace(s, ghost, ghost, true), true);
  assert.equal(createPlacementQuery(s).canPlace(ghost, ghost, true), true);
  u.owner = 2;
  assert.equal(createPlacementQuery(s).canPlace(ghost, ghost, true), false);
});
