import test from 'node:test';
import assert from 'node:assert/strict';
import {
  cells,
  occupants,
  occupant,
  canPlace,
  deploymentRows,
} from '../../src/engine/core/geometry';
import { add, fixture } from '../helpers';

test('占位查询保持逐格匹配、叠放栈顶和独立坐标，不把格间点当作占位', () => {
  const s = fixture();
  const first = add(s, 'u25', 1, 4, 7);
  add(s, 'u25', 1, 4, 7);
  const large = add(s, 1, 2, 6, 8);
  large.size = 2;
  const before = structuredClone(s);
  for (let y = 0; y <= 14; y += 0.5)
    for (let x = 0; x <= 10; x += 0.5) {
      const expected = s.units.filter((u) => {
        for (let dy = 0; dy < u.size; dy++)
          for (let dx = 0; dx < u.size; dx++) if (u.x + dx === x && u.y + dy === y) return true;
        return false;
      });
      assert.deepEqual(occupants(s, { x, y }), expected);
      assert.equal(occupant(s, { x, y }), expected[0]);
    }
  assert.equal(occupant(s, first), first);
  assert.deepEqual(cells(large), [
    { x: 6, y: 8 },
    { x: 7, y: 8 },
    { x: 6, y: 9 },
    { x: 7, y: 9 },
  ]);
  for (const u of [first, large]) {
    const points = cells(u);
    points[0].x = -1;
    assert.equal(cells(u)[0].x, u.x);
  }
  assert.deepEqual(s, before);
  assert.equal(canPlace(s, first, { x: 7, y: 9 }), false);
  assert.equal(canPlace(s, first, { x: 8, y: 9 }), true);
});

test('大体型部署行逐行计一枚，实时移动和中立墓地不产生旧缓存行权', () => {
  const s = fixture();
  const large = add(s, 1, 1, 2, 9);
  large.size = 2;
  const support = add(s, 1, 1, 5, 9);
  add(s, 'grave', 1, 6, 10);
  assert.deepEqual(deploymentRows(s, 1), [1, 2, 3, 4, 5, 6, 7, 8, 9]);
  support.y = 10;
  assert.deepEqual(deploymentRows(s, 1), [1, 2, 3, 4, 5, 6, 7, 8, 10]);
  support.owner = 2;
  assert.deepEqual(deploymentRows(s, 1), [1, 2, 3, 4, 5, 6, 7, 8]);
});
