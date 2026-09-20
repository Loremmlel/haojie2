import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { createSession, validState, type GameState, type Session } from '../../src/engine';
import { installEquipment } from '../../src/engine/shrines';
import { add, fixture } from '../helpers';
const root = 'artifacts/feedback3-fixtures';
mkdirSync(root, { recursive: true });
function save(name: string, state: GameState, session: Session = createSession(state)) {
  assert.ok(validState(state), name);
  writeFileSync(`${root}/${name}.json`, JSON.stringify(session));
}
{
  const s = fixture();
  add(s, 'u7', 1, 2, 3);
  add(s, 1, 2, 6, 7);
  save('giant', s);
  s.active = 2;
  const session = createSession(s, { mode: 'ai', human: 1, difficulty: 'easy' });
  // A genuine undo boundary holds automatic play until a new human command clears redo.
  session.future = [structuredClone(s)];
  save('interrupt', s, session);
}
{
  const s = fixture();
  add(s, 'u4', 1, 3, 5).size = 2;
  save('large', s);
}
{
  const s = fixture(),
    carrier = add(s, 26, 1, 2, 5);
  installEquipment(carrier, 'u28');
  add(s, 1, 2, 3, 5);
  add(s, 1, 2, 4, 6);
  add(s, 1, 2, 5, 6);
  add(s, 1, 1, 3, 6);
  save('heart', s);
}
{
  const s = fixture();
  add(s, 2, 1, 3, 5).hp = 5;
  save('healer', s);
}
