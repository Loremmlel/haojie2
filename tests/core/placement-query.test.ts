import assert from 'node:assert/strict';
import test from 'node:test';
import {
  ALL_CELLS,
  canPlace,
  createPlacementQuery,
  createAttackQuery,
  createPointAttackQuery,
  attackPath,
  attackRoutes,
  targets,
  movementPath,
} from '../../src/engine/core/geometry';
import { fixture, add } from '../helpers';
import { asTarget, template } from '../../src/engine/core/state';
import { createCommandInspector, inspectCommand } from '../../src/engine/commands/game';

test('决策内攻击路径复用保留自动正面偏好、方向顺序、重叠和穿透边界', () => {
  for (const owner of [1, 2] as const)
    for (const overlap of [false, true]) {
      const s = fixture();
      s.active = owner;
      const source = add(s, 26, owner, 4, owner === 1 ? 4 : 8);
      const victim = add(s, 24, owner === 1 ? 2 : 1, 4, 6);
      if (overlap) {
        source.size = 2;
        victim.x = source.x + 1;
        victim.y = source.y;
      }
      add(s, 'wall', owner === 1 ? 2 : 1, 5, 7);
      add(s, 'u25', owner === 1 ? 2 : 1, victim.x, victim.y);
      add(s, 5, owner === 1 ? 2 : 1, 6, 6);
      add(s, 'u25', owner, 6, 6);
      add(s, 'grave', owner, 3, 6);
      add(s, 26, owner === 1 ? 2 : 1, 5, owner === 1 ? 13 : 1);
      const landmark = add(s, 's1', owner === 1 ? 2 : 1, victim.x, victim.y);
      s.units.pop();
      s.landmarks = [landmark];
      const before = structuredClone(s),
        query = createAttackQuery(s),
        inspect = createCommandInspector(s);
      for (const target of [...targets(s), asTarget(source)])
        for (const limit of [0, 1.5, 4, 7, 20])
          for (const pierce of [true, false]) {
            const expected = attackRoutes(s, source, target, limit, pierce);
            const directions = expected.map((route) => route.direction);
            assert.deepEqual(query.directions(source, target, limit, pierce), directions);
            assert.deepEqual(query.routes(source, target, limit, pierce), expected);
            for (const direction of [undefined, 'down', 'up', 'right', 'left'] as const) {
              const path = attackPath(s, source, target, limit, direction, pierce);
              assert.deepEqual(query.path(source, target, limit, direction, pierce), path);
              assert.deepEqual(
                query.path(
                  { ...source, mode: 'attack', shots: 0 },
                  target,
                  limit,
                  direction,
                  pierce,
                ),
                path,
              );
              const command = {
                type: 'attack' as const,
                unitId: source.id,
                targetId: target.id,
                ...(direction ? { direction } : {}),
              };
              assert.deepEqual(inspect(command), inspectCommand(s, command));
            }
            const returned = query.routes(source, target, limit, pierce);
            if (returned[0]) {
              returned[0].path[0].x = -1;
              returned[0].direction = returned[0].direction === 'left' ? 'right' : 'left';
            }
            assert.deepEqual(query.routes(source, target, limit, pierce), expected);
            const returnedDirections = query.directions(source, target, limit, pierce);
            returnedDirections.reverse();
            returnedDirections.push('up');
            assert.deepEqual(query.directions(source, target, limit, pierce), directions);
            assert.deepEqual(query.routes(source, target, limit, pierce), expected);
            const auto = query.path(source, target, limit, undefined, pierce);
            if (auto?.[0]) auto[0].x = -1;
            assert.deepEqual(
              query.path(source, target, limit, undefined, pierce),
              attackPath(s, source, target, limit, undefined, pierce),
            );
          }
      assert.deepEqual(s, before);
      const moved = { ...source, x: 1, y: 2 };
      assert.deepEqual(
        query.path(moved, asTarget(victim), 7),
        attackPath(s, moved, asTarget(victim), 7),
      );
      for (const [from, target] of [
        [moved, asTarget(victim)],
        [source, asTarget({ ...victim, x: 1, y: 3 })],
      ] as const)
        assert.deepEqual(
          query.directions(from, target, 7),
          attackRoutes(s, from, target, 7).map((route) => route.direction),
        );
    }
});

test('基地与实体同名的兼容局面仍按原目标ID排除整枚阻挡实体', () => {
  const s = fixture(),
    source = add(s, 26, 1, 5, 10),
    blocker = add(s, 26, 2, 5, 11);
  blocker.id = 'base-2';
  const base = targets(s).find((target) => target.id === 'base-2' && !target.unit)!,
    query = createAttackQuery(s);
  const expected = attackPath(s, source, base, 4);
  assert.ok(expected?.some((point) => point.x === blocker.x && point.y === blocker.y));
  assert.deepEqual(query.path(source, base, 4), expected);
  assert.deepEqual(query.routes(source, base, 4), attackRoutes(s, source, base, 4));
  assert.deepEqual(
    query.directions(source, base, 4),
    attackRoutes(s, source, base, 4).map((route) => route.direction),
  );
});

test('实体目标使用敌基地ID时仍保留该ID取消基地封锁的兼容路径', () => {
  const s = fixture(),
    source = add(s, 26, 1, 5, 12),
    victim = add(s, 26, 2, 6, 13);
  victim.id = 'base-2';
  const target = asTarget(victim),
    query = createAttackQuery(s),
    expected = attackPath(s, source, target, 4);
  assert.ok(expected?.some((point) => point.x === 5 && point.y === 13));
  assert.deepEqual(query.path(source, target, 4), expected);
  assert.deepEqual(query.routes(source, target, 4), attackRoutes(s, source, target, 4));
  assert.deepEqual(
    query.directions(source, target, 4),
    attackRoutes(s, source, target, 4).map((route) => route.direction),
  );
});

test('单来源纯落点射程场保留多格起点、敌方终点、基地封锁与分数距离', () => {
  for (const owner of [1, 2] as const)
    for (const size of [1, 2]) {
      const s = fixture(),
        source = add(s, size === 1 ? 26 : 5, owner, 3, 4);
      const opponent = owner === 1 ? 2 : 1;
      add(s, 'wall', opponent, 5, 5);
      add(s, 5, opponent, 6, 6);
      add(s, 'u25', owner, 5, 7);
      add(s, 'u25', owner, 5, 7);
      const landmark = add(s, 's1', opponent, 5, 6);
      s.units.pop();
      s.landmarks = [landmark];
      const before = structuredClone(s);
      for (const limit of [-1, 0, 0.5, 1.5, 3, 5.5, 20]) {
        const query = createPointAttackQuery(s, source, limit);
        for (const point of [...ALL_CELLS, { x: 0, y: 1 }, { x: 1.5, y: 2 }])
          assert.equal(
            query.has(point),
            attackPath(s, source, point, limit) !== null,
            `${owner}:${size}:${limit}:${point.x},${point.y}`,
          );
      }
      assert.deepEqual(s, before);
    }
});

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
