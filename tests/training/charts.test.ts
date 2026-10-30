import assert from 'node:assert/strict';
import test from 'node:test';
import { curve, smooth } from '../../training/ui/charts';

test('训练趋势按时间平滑，缺测与运行边界断开', () => {
  const values = smooth(
    [
      { id: 1, t: 0, loss: 0 },
      { id: 2, t: 5, loss: 10 },
      { id: 3, t: 10, loss: 0 },
      { id: 4, t: 40, loss: 4 },
      { id: 5, t: 45, gap: true },
      { id: 6, t: 50, loss: 2 },
    ],
    'loss',
    60,
  );
  assert.ok(values[1]!.value > 0 && values[1]!.value < 1);
  assert.equal(values[3], null);
  assert.equal(values[4]!.value, 4);
  assert.equal(values[5], null);
  assert.equal(values[6]!.value, 2);
  assert.equal(values[1]!.raw, 10);
});

test('训练曲线的控制点不越过单调区间，单点仍可见', () => {
  assert.equal(curve([[3, 4]]), 'M3,4h0.01');
  const path = curve([
    [0, 0],
    [10, 100],
    [20, 101],
    [30, 101],
  ]);
  const controls = path
    .split('C')
    .slice(1)
    .map((segment) => segment.split(' ').map((pair) => pair.split(',').map(Number)));
  const intervals = [
    [0, 100],
    [100, 101],
    [101, 101],
  ];
  controls.forEach((points, i) =>
    points.forEach((p) => assert.ok(p[1] >= intervals[i][0] && p[1] <= intervals[i][1])),
  );
});
