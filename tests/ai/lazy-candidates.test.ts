import assert from 'node:assert/strict';
import test from 'node:test';
import {
  candidateGroups,
  iterateCandidateGroups,
  attackCandidates,
} from '../../src/ai/planning/candidates';
import { fixture, add, card } from '../helpers';
import { targets } from '../../src/engine/core/geometry';

test('延迟落点评分保留组顺序、优先级、完整候选及同分顺序', () => {
  const s = fixture();
  add(s, 9, 1, 3, 5);
  add(s, 24, 2, 5, 7);
  add(s, 'u23', 1, 7, 4);
  card(s, 1);
  card(s, 7);
  card(s, 'u25');
  card(s, 25);
  card(s, 8);
  const before = structuredClone(s);
  for (const level of ['easy', 'medium', 'hard'] as const) {
    const eager = candidateGroups(s, level);
    const lazy = [...iterateCandidateGroups(s, level, true)].map((g) => ({ ...g }));
    assert.deepEqual(lazy, eager);
  }
  assert.deepEqual(s, before);
});

test('单目标续招等价于原全目标生成后的过滤，含侧面、基地和不存在目标', () => {
  const s = fixture(),
    u = add(s, 'u20', 1, 4, 6);
  add(s, 24, 2, 5, 7);
  add(s, 2, 1, 4, 7);
  add(s, 'u25', 2, 6, 7);
  add(s, 'u25', 2, 6, 7);
  const original = attackCandidates(s, u.id);
  for (const id of [...targets(s).map((t) => t.id), 'missing'])
    assert.deepEqual(
      attackCandidates(s, u.id, id),
      original.filter((c) => c.targetId === id),
    );
});
