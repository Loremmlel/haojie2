import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { createSession, validState, type GameState } from '../../../src/engine';
import { refreshDeployment } from '../../../src/engine/core/geometry';
import { addEffect } from '../../../src/engine/core/state';
import { add, card, fixture } from '../../helpers';
const root = 'artifacts/feedback4-fixtures';
mkdirSync(root, { recursive: true });
function save(name: string, s: GameState) {
  assert.ok(validState(s), name);
  writeFileSync(`${root}/${name}.json`, JSON.stringify(createSession(s)));
}
for (const owner of [1, 2] as const) {
  const s = fixture();
  s.active = owner;
  const y = owner === 1 ? 10 : 4;
  add(s, 14, owner, 5, y);
  add(s, 1, owner, 4, y);
  save(`cannon-${owner}`, s);
}
for (const advantage of [false, true]) {
  const s = fixture();
  add(s, 9, 1, 1, 9);
  add(s, 9, advantage ? 1 : 2, 3, 9);
  refreshDeployment(s, 1);
  card(s, 9);
  save(advantage ? 'row-two-zero' : 'row-one-one', s);
}
{
  const s = fixture();
  add(s, 9, 1, 1, 9);
  add(s, 9, 1, 3, 8);
  card(s, 9);
  save('live-deployment', s);
}
{
  const s = fixture();
  add(s, 'grave', 1, 3, 5);
  add(s, 1, 1, 5, 5);
  add(s, 1, 1, 7, 5);
  card(s, 25);
  save('neutral-grave', s);
}
{
  const s = fixture();
  add(s, 7, 1, 3, 4);
  add(s, 5, 2, 3, 6);
  save('hook', s);
}
{
  const s = fixture();
  const arch = add(s, 'archmage', 1, 3, 4);
  addEffect(s, arch, 'freeze', 2, 0, 4);
  addEffect(s, arch, 'attack', 2, 0, 2, -15);
  addEffect(s, arch, 'attack', 1, 0, 2, 10);
  save('counter-status', s);
}
