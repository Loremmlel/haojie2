import test from 'node:test';
import assert from 'node:assert/strict';
import { applyCommand, createGame, createSession, parseSession } from '../../src/engine';
import { fixture, add, card, round } from '../helpers';
import {
  DEFAULT_STORAGE_KEY,
  readStoredSession,
  writeStoredSession,
} from '../../src/ui/session/storage';

test('horn remains usable in its eighth owning turn and expires at the ninth', () => {
  let s = fixture();
  const u = add(s, 9, 1, 2, 4),
    id = card(s, 'u17');
  for (let i = 0; i < 7; i++) s = round(s);
  assert.ok(s.hands[1].some((c) => c.id === id));
  assert.doesNotThrow(() => applyCommand(s, { type: 'cast', cardId: id, targetId: u.id }));
  s = round(s);
  assert.equal(
    s.hands[1].some((c) => c.id === id),
    false,
  );
  assert.throws(() => applyCommand(s, { type: 'cast', cardId: id, targetId: u.id }));
});
test('older v2 unlimited horns gain the confirmed expiry in all undo/redo snapshots', () => {
  const s = fixture(),
    id = card(s, 'u17');
  delete s.hands[1][0].expiresAt;
  const expired = structuredClone(s);
  expired.turns[1] += 8;
  const before = { ...createSession(s), past: [structuredClone(s)], future: [expired] };
  const text = JSON.stringify(before),
    after = parseSession(text);
  assert.equal(after.present.hands[1][0].expiresAt, s.turns[1] + 8);
  assert.equal(after.past[0].hands[1][0].id, id);
  assert.equal(after.past[0].hands[1][0].expiresAt, s.turns[1] + 8);
  assert.deepEqual(after.future[0].hands[1], []);
  assert.equal(after.present.rng, s.rng);
  assert.equal(before.present.hands[1][0].expiresAt, undefined);
  assert.deepEqual(parseSession(JSON.stringify(after)), after);
});
test('reload uses the stable schema key and retains both current state and history', () => {
  const data = new Map<string, string>();
  const storage = {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => {
      data.set(key, value);
    },
  };
  const session = createSession(fixture());
  session.past.push(createGame(9));
  assert.equal(writeStoredSession(storage, DEFAULT_STORAGE_KEY, session), 'saved');
  const loaded = readStoredSession(storage, DEFAULT_STORAGE_KEY, () => {
    throw new Error('must not start a new game');
  });
  assert.deepEqual(loaded.session, session);
  assert.equal(loaded.writable, true);
  assert.equal(data.size, 1);
});
test('corrupt saved data remains untouched and pauses writes until explicit replacement', () => {
  const raw = '{corrupt';
  let writes = 0;
  const storage = {
    getItem: () => raw,
    setItem: () => {
      writes++;
    },
  };
  const result = readStoredSession(storage, DEFAULT_STORAGE_KEY, () => createSession(fixture()));
  assert.equal(result.writable, false);
  assert.equal(writes, 0);
  assert.match(result.notice, /暂停/);
});
test('quota exhaustion can save a current-only snapshot; denied storage reports failure', () => {
  const session = createSession(fixture());
  session.past.push(createGame(5));
  let value = '';
  const quota = {
    getItem: () => null,
    setItem: (_key: string, raw: string) => {
      if (JSON.parse(raw).past.length) throw new Error('quota');
      value = raw;
    },
  };
  assert.equal(writeStoredSession(quota, DEFAULT_STORAGE_KEY, session), 'snapshot');
  assert.deepEqual(JSON.parse(value).present, session.present);
  assert.equal(session.past.length, 1);
  const denied = {
    getItem: () => null,
    setItem: () => {
      throw new Error('denied');
    },
  };
  assert.equal(writeStoredSession(denied, DEFAULT_STORAGE_KEY, session), 'unavailable');
});
