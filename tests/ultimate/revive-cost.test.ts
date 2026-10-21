import test from 'node:test';
import assert from 'node:assert/strict';
import {
  applyCommand,
  createSession,
  dispatch,
  getStats,
  redo,
  undo,
  unitActions,
} from '../../src/engine';
import { kill } from '../../src/engine/commands/combat';
import { applyRuntimeCommand } from '../../src/engine/commands/game';
import { forkRuntimePosition, importRuntimePosition } from '../../src/engine/runtime/position';
import { add, fixture, round, unit } from '../helpers';

test('U19 can spend its last five max HP to complete a second revival without giving the enemy a head', () => {
  let s = fixture();
  const mage = add(s, 'u19', 1, 3, 4),
    first = add(s, 1, 1, 5, 4),
    second = add(s, 9, 1, 5, 5);
  kill(s, first, { owner: 2, kind: 'spell' });
  kill(s, second, { owner: 2, kind: 'spell' });
  s = round(s);
  s = applyCommand(s, { type: 'skill', unitId: mage.id, deathId: s.deaths[0].id, x: 4, y: 4 });
  assert.equal(unit(s, mage.id).maxHp, 5);
  s = round(s);
  assert.ok(unitActions(s, unit(s, mage.id)).some((a) => a.id === 'revive'));
  const before = structuredClone(s);
  const played = dispatch(createSession(s), {
    type: 'skill',
    unitId: mage.id,
    deathId: s.deaths[1].id,
    x: 4,
    y: 5,
  });
  const after = played.present;
  assert.equal(
    after.units.some((u) => u.id === mage.id),
    false,
  );
  const revived = after.units.find((u) => u.kind === 9)!;
  assert.ok(revived);
  assert.equal(revived.hp, revived.maxHp);
  assert.equal(getStats(after, revived).sleeping, true);
  assert.equal(after.deaths[1].revived, true);
  assert.equal(after.deaths.filter((d) => d.kind === 'u19').length, 1);
  assert.deepEqual(after.heads, before.heads);
  assert.deepEqual(after.rng, before.rng);
  assert.deepEqual(s, before);
  assert.deepEqual(undo(played).present, before);
  assert.deepEqual(redo(undo(played)).present, after);
});

test('U19 final revival still validates record and destination before spending its remaining life', () => {
  const s = fixture(),
    mage = add(s, 'u19', 1, 3, 4);
  mage.hp = mage.maxHp = 5;
  s.deaths.push({ id: 'recent', kind: 1, owner: 1, ply: s.ply - 1, revived: false });
  s.deaths.push({ id: 'used', kind: 1, owner: 1, ply: s.ply - 1, revived: true });
  s.deaths.push({ id: 'old', kind: 1, owner: 1, ply: s.ply - 5, revived: false });
  s.deaths.push({ id: 'enemy', kind: 1, owner: 2, ply: s.ply - 1, revived: false });
  s.deaths.push({ id: 'current', kind: 1, owner: 1, ply: s.ply, revived: false });
  for (const choice of [
    { deathId: 'recent', x: 9, y: 12 },
    { deathId: 'recent', x: 3, y: 4 },
    ...['used', 'old', 'enemy', 'current', 'missing'].map((deathId) => ({ deathId, x: 4, y: 4 })),
  ]) {
    const before = structuredClone(s);
    assert.throws(() => applyCommand(s, { type: 'skill', unitId: mage.id, ...choice }));
    assert.deepEqual(s, before);
  }
});

test('满死亡历史的复活与施法者死亡隔离原局面、兄弟分支和外部快照', () => {
  const s = fixture(),
    mage = add(s, 'u19', 1, 3, 4);
  mage.hp = mage.maxHp = 5;
  s.deaths = Array.from({ length: 240 }, (_, i) => ({
    id: `recent-${i}`,
    kind: 1,
    owner: 1,
    ply: s.ply - 1,
    revived: false,
  }));
  const record = Object.assign(s.deaths[239], {
    group: 'history-group',
    sourceNote: { label: '原记录扩展' },
  });
  const command = { type: 'skill' as const, unitId: mage.id, deathId: 'recent-239', x: 4, y: 4 };
  const runtime = importRuntimePosition(s),
    sibling = forkRuntimePosition(runtime),
    before = structuredClone(runtime);
  assert.throws(() => applyRuntimeCommand(runtime, { ...command, x: 3 }));
  const next = applyRuntimeCommand(runtime, command);
  assert.equal(next.deaths.length, 240);
  assert.equal(next.deaths[0].id, 'recent-1');
  assert.equal(next.deaths.at(-2)!.revived, true);
  assert.deepEqual(next.deaths.at(-2), { ...record, revived: true });
  assert.equal(next.deaths.at(-1)!.kind, 'u19');
  assert.equal(next.deaths.at(-1)!.owner, 1);
  assert.equal(next.deaths.at(-1)!.ply, s.ply);
  assert.deepEqual(next.heads, s.heads);
  assert.deepEqual(runtime, before);
  assert.deepEqual(sibling, before);
  assert.deepEqual(s, before);

  const exported = applyCommand(runtime, command);
  exported.deaths[0].revived = true;
  exported.deaths.at(-2)!.kind = 9;
  assert.equal(runtime.deaths[1].revived, false);
  assert.equal(next.deaths.at(-2)!.kind, 1);

  // 外部宿主允许附加别名；兼容入口仍保持同一记录的更新关系。
  const aliased = Object.assign(structuredClone(s), { deathAlias: s.deaths.at(-1)! });
  aliased.deaths[239] = aliased.deathAlias;
  const result = applyCommand(aliased, command) as typeof aliased;
  assert.equal(result.deathAlias, result.deaths.at(-2));
  assert.equal(result.deathAlias.revived, true);
  assert.equal(aliased.deathAlias.revived, false);
});
