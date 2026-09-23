import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  createGame,
  createSession,
  dispatch,
  undo,
  redo,
  parseSession,
  serializeSession,
  sessionSave,
  type RecordedSave,
  type Session,
} from '../../src/engine';
import { rewindMatch } from '../../src/match/history';
import { replayTranscript } from '../../scripts/ai/cli/transcript';
import { add, fixture } from '../helpers';

function currentReplay() {
  const [header, ...rows] = readFileSync(
    'docs/playtests/current/cli-feedback5-20260923.jsonl',
    'utf8',
  )
    .trim()
    .split('\n')
    .map((line) => JSON.parse(line));
  let session = parseSession(JSON.stringify(header.initial));
  for (const row of rows) session = dispatch(session, row.command);
  return session;
}

test('完整79步路线跨越60步缓存，紧凑存档重放、续玩和CLI共用结果', () => {
  const session = currentReplay();
  const text = serializeSession(session);
  const save = JSON.parse(text);
  assert.equal(save.format, 'haojie-record-v1');
  assert.equal(save.commands.length, 79);
  assert.equal(save.past, undefined);
  assert.equal(save.future, undefined);
  assert.equal(save.humanAnchor, undefined);
  assert.ok(Buffer.byteLength(text) < Buffer.byteLength(JSON.stringify(session)) / 5);
  const restored = parseSession(text);
  assert.deepEqual(restored.present, session.present);
  assert.deepEqual(restored.past, session.past);
  assert.equal(restored.past.length, 60);
  assert.equal(serializeSession(restored), text);
  assert.deepEqual(replayTranscript(text), { state: session.present, commands: 79 });
  assert.deepEqual(redo(undo(restored)).present, session.present);
});

test('悔棋导出只保存当前路线，另走丢弃旧分支，命令参数不共享外部引用', () => {
  const state = fixture();
  const unit = add(state, 1, 1, 3, 4);
  const start = createSession(state);
  const command = { type: 'move' as const, unitId: unit.id, x: 3, y: 5 };
  const first = dispatch(start, command);
  command.y = 6;
  const back = undo(first);
  assert.equal(sessionSave(back).commands.length, 0);
  assert.equal(parseSession(serializeSession(back)).future.length, 0);
  assert.deepEqual(redo(back).present, first.present);
  const branch = dispatch(back, { type: 'move', unitId: unit.id, x: 4, y: 4 });
  assert.equal(branch.future.length, 0);
  assert.equal(sessionSave(branch).commands.length, 1);
  assert.equal(sessionSave(branch).commands[0].x, 4);
  assert.deepEqual(parseSession(serializeSession(branch)).present, branch.present);
  const original = serializeSession(branch);
  assert.throws(() => dispatch(branch, { type: 'attack', unitId: 'missing', targetId: 'missing' }));
  assert.equal(serializeSession(branch), original);
});

test('超过60步的AI响应从增量文件恢复后仍可整段悔棋和重做', () => {
  const state = fixture();
  state.bonus[2] = 310;
  let session = dispatch(createSession(state, { mode: 'ai', human: 1, difficulty: 'medium' }), {
    type: 'end',
  });
  for (let i = 0; i < 300; i++) session = dispatch(session, { type: 'summon' });
  const restored = parseSession(serializeSession(session));
  assert.equal(restored.record!.cursor, 301);
  assert.equal(restored.past.length, 60);
  const back = rewindMatch(restored);
  assert.deepEqual(back.present, state);
  assert.equal(sessionSave(back).commands.length, 0);
  const forward = rewindMatch(back, true);
  assert.deepEqual(forward.present, restored.present);
  assert.equal(serializeSession(forward), serializeSession(restored));
  assert.deepEqual(rewindMatch(forward).present, state);
});

test('旧v2继续迁移和悔棋，导出明确以已知局面起录', () => {
  const first = dispatch(createSession(createGame(7)), { type: 'summon' });
  const { record: _, ...old } = first;
  const restored = parseSession(JSON.stringify(old));
  assert.deepEqual(undo(restored).present, first.past[0]);
  const save = sessionSave(restored);
  assert.equal(save.origin, 'position');
  assert.equal(save.commands.length, 0);
  assert.deepEqual(save.initial, restored.present);
  const next = dispatch(restored, { type: 'summon' });
  assert.equal(sessionSave(next).commands.length, 1);
  assert.equal(sessionSave(next).origin, 'position');
  assert.deepEqual(parseSession(serializeSession(next)).present, next.present);
});

test('拒绝版本不符、伪造起点、非法命令和终态随机数损坏', () => {
  const session = dispatch(createSession(createGame(7)), { type: 'summon' });
  assert.equal(sessionSave(session).origin, 'opening');
  for (const corrupt of [
    (s: RecordedSave) => {
      s.ruleset = 'older';
    },
    (s: RecordedSave) => {
      s.initial.rng++;
    },
    (s: RecordedSave) => {
      s.commands[0] = { type: 'attack', unitId: 'missing' };
    },
    (s: RecordedSave) => {
      s.present.rng++;
    },
    (s: RecordedSave) => {
      s.commands = [];
    },
  ]) {
    const bad = JSON.parse(serializeSession(session));
    corrupt(bad);
    assert.throws(() => parseSession(JSON.stringify(bad)));
  }
  const malformed: Session = structuredClone(session);
  malformed.record!.cursor = -1;
  assert.throws(() => parseSession(JSON.stringify(malformed)));
});

test('旧人机档在记录起点之前的锚点仍能恢复，重置路线避免生成错误增量', () => {
  const state = fixture();
  state.bonus[2] = 100;
  const played = dispatch(createSession(state, { mode: 'ai', human: 1, difficulty: 'easy' }), {
    type: 'end',
  });
  const { record: _, humanAnchorCursor: __, ...legacy } = played;
  let session = parseSession(JSON.stringify(legacy));
  for (let i = 0; i < 70; i++) session = dispatch(session, { type: 'summon' });
  const back = rewindMatch(session);
  assert.deepEqual(back.present, state);
  assert.deepEqual(parseSession(serializeSession(back)).present, state);
  assert.equal(sessionSave(back).commands.length, 0);
  assert.deepEqual(rewindMatch(back, true).present, session.present);
});
