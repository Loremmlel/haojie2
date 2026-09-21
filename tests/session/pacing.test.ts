import test from 'node:test';
import assert from 'node:assert/strict';
import type { GameEvent } from '../../src/engine';
import { actionDelay, waitForPresentation } from '../../src/ui/opponent/pacing';

const empty = { first: false, previous: null, events: [] };
test('AI starts at a readable pace, including already-cached commands', () => {
  const move = { type: 'move' as const, unitId: 'runner' };
  const regular = actionDelay(move, empty);
  const repeated = actionDelay(move, { ...empty, previous: move });
  const start = actionDelay(move, { ...empty, first: true });
  assert.ok(start >= 1400 && start >= regular);
  assert.ok(regular >= 1000);
  assert.ok(repeated >= 700 && repeated < regular);
  assert.ok(actionDelay({ type: 'cast' }, empty) > regular);
  assert.ok(actionDelay({ type: 'begin' }, empty) < repeated);
});
test('even housekeeping lets previous hit numbers and summon reveals finish', () => {
  const events: GameEvent[] = [{ id: 'hit', type: 'damage', amount: 20 }];
  const before = structuredClone(events);
  assert.ok(actionDelay({ type: 'finish-mode' }, { ...empty, events }) >= 1400);
  assert.ok(
    actionDelay({ type: 'begin' }, { ...empty, events: [{ id: 'draw', type: 'summon' }] }) >= 1500,
  );
  assert.deepEqual(events, before, 'Pacing must not alter event data or any game state');
});
test('presentation waiting is abortable, both before and after scheduling', async () => {
  const abort = new AbortController();
  const waiting = waitForPresentation(5000, abort.signal);
  abort.abort();
  await assert.rejects(waiting, { name: 'AbortError' });
  await assert.rejects(waitForPresentation(0, abort.signal), { name: 'AbortError' });
});
test('an elapsed presentation minimum does not add another think-time wait', async () => {
  const abort = new AbortController();
  // 剩余时间为负表示真实搜索已经覆盖展示间隔。
  await waitForPresentation(-1000, abort.signal);
  abort.abort(); // 已完成等待不保留能再次拒绝 Promise 的取消监听器。
});
