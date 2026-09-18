import test from 'node:test';
import assert from 'node:assert/strict';
import {
  applyCommand,
  availableSyntheses,
  SYNTHESIS_RECIPES,
  getStats,
  definition,
  createSession,
  dispatch,
  undo,
  redo,
  parseSession,
  isLegal,
  synthesisDestinations,
  ULTIMATE_POOL,
  SUMMON_POOL,
  type GameState,
  type Command,
} from '../../src/engine';
import { addEffect } from '../../src/engine/state';
import { add, card, fixture } from '../helpers';

export function fusionFixture(recipeId: string) {
  const s = fixture();
  const recipe = SYNTHESIS_RECIPES.find((r) => r.id === recipeId)!;
  const ids = Array.from({ length: 3 }, (_, i) =>
    recipe.source === 'board' ? add(s, recipe.material, 1, 2 + i, 3).id : card(s, recipe.material),
  );
  s.phase = 'synthesis';
  s.summonSlots = 2;
  return { s, recipe, ids };
}
const fuse = (recipeId: string, materialIds: string[], x = 2, y = 3): Command => ({
  type: 'synthesize',
  recipeId,
  materialIds,
  x,
  y,
});

test('2.5 seven recipes remove exactly three materials atomically, deploy fresh units, and leave both random pools unchanged', () => {
  assert.equal(SUMMON_POOL.length, 26);
  assert.equal(ULTIMATE_POOL.length, 28);
  for (const { id } of SYNTHESIS_RECIPES.filter((r) => !definition(r.result).aura)) {
    const { s, recipe, ids } = fusionFixture(id);
    if (recipe.source === 'board') {
      s.units[0].hp = 1;
      s.units[0].equipment = ['u16'];
      s.units[0].attackBonus = 30;
      s.units[0].kills = 8;
    }
    const before = structuredClone(s);
    const to = recipe.source === 'board' ? { x: 2, y: 3 } : { x: 2, y: 4 };
    const n = applyCommand(s, fuse(id, ids, to.x, to.y));
    assert.deepEqual(s, before, 'input state is immutable');
    const result = n.units.find((u) => u.kind === recipe.result)!;
    assert.ok(result, id);
    assert.equal(result.hp, definition(recipe.result).health);
    assert.equal(result.attackBonus, 0);
    assert.equal(result.kills, 0);
    assert.deepEqual(result.equipment, []);
    assert.deepEqual(result.effects, []);
    assert.equal(n.units.filter((u) => ids.includes(u.id)).length, 0);
    assert.equal(n.hands[1].filter((c) => ids.includes(c.id)).length, 0);
    assert.equal(n.deaths.length, 0);
    assert.equal(n.pending.length, 0);
    assert.deepEqual(n.heads, before.heads);
    assert.equal(n.rng, before.rng);
    assert.equal(n.summonSlots, 2);
    assert.equal(n.phase, 'summon');
    assert.equal(getStats(n, result).sleeping, recipe.result !== 'u12');
    assert.equal(result.deployedAt, s.ply);
  }
});

test('2.5 fusion window opens only at an actual own start after expiry, and skipping closes it without spending materials', () => {
  let s = fixture();
  for (let i = 0; i < 3; i++) add(s, 'u21', 1, 2 + i, 3);
  assert.equal(availableSyntheses(s).length, 1);
  assert.equal(
    isLegal(
      s,
      fuse(
        'sage',
        s.units.map((u) => u.id),
      ),
    ),
    false,
  );
  s.active = 2;
  s.ply = 4;
  s.turns = { 1: 2, 2: 2 };
  const started = applyCommand(s, { type: 'end' });
  assert.equal(started.active, 1);
  assert.equal(started.phase, 'synthesis');
  assert.equal(started.summonSlots, 2);
  for (const c of [{ type: 'summon' }, { type: 'begin' }, { type: 'end' }] as Command[])
    assert.equal(isLegal(started, c), false);
  const skipped = applyCommand(started, { type: 'skip-synthesis' });
  assert.deepEqual(skipped.units, started.units);
  assert.equal(skipped.phase, 'summon');
  assert.equal(
    isLegal(
      skipped,
      fuse(
        'sage',
        skipped.units.map((u) => u.id),
      ),
    ),
    false,
  );
  const expired = structuredClone(s);
  expired.units[0].expiresAt = 5;
  // Expiry can trigger the original U21 heal but cannot leave three synthesis materials.
  assert.equal(applyCommand(expired, { type: 'end' }).phase, 'summon');
  s = fixture();
  for (let i = 0; i < 3; i++) card(s, 'u28');
  for (const c of s.hands[1]) c.expiresAt = 4;
  s.active = 2;
  s.ply = 6;
  s.turns = { 1: 3, 2: 3 };
  assert.equal(applyCommand(s, { type: 'end' }).phase, 'summon');
});

test('2.5 material identity, allegiance and destination are validated before removal; no friendly death effects or links leak', () => {
  const { s, ids } = fusionFixture('sage');
  const friend = add(s, 26, 1, 6, 3);
  friend.hp = 1;
  add(s, 'citadel', 1, 4, 4);
  s.siphons.push({ id: 'link', sourceId: friend.id, fromId: ids[0], toId: friend.id, owner: 1 });
  const before = structuredClone(s);
  for (const c of [
    fuse('sage', [ids[0], ids[0], ids[2]]),
    fuse('sage', ids.slice(0, 2)),
    fuse('sage', [...ids, friend.id]),
    fuse('slayer', ids),
    fuse('sage', ids, 5, 1),
    fuse('sage', ids, 2, 13),
    fuse('sage', ids, 6, 3),
  ]) {
    assert.throws(() => applyCommand(s, c));
    assert.deepEqual(s, before);
  }
  const n = applyCommand(s, fuse('sage', ids));
  assert.equal(n.units.find((u) => u.id === friend.id)!.hp, 1, 'no U21 death-heal');
  assert.equal(n.pending.length, 0, 'no citadel spawn');
  assert.equal(n.siphons.length, 0);
  const frozen = structuredClone(s);
  addEffect(frozen, frozen.units[0], 'freeze', 2, 0, 4);
  assert.equal(availableSyntheses(frozen).length, 0);
  assert.equal(isLegal(frozen, fuse('sage', ids)), false);
  const enemy = structuredClone(s);
  enemy.units[0].owner = 2;
  assert.equal(isLegal(enemy, fuse('sage', ids)), false);
  const silenced = structuredClone(s);
  silenced.units[0].silenced = true;
  assert.equal(
    isLegal(silenced, fuse('sage', ids)),
    true,
    'identity remains eligible under silence',
  );
});

test('2.5 repeated synthesis and in-progress selection state survive saves, undo and redo without rerolling', () => {
  const { s, ids } = fusionFixture('archmage');
  const extra = Array.from({ length: 3 }, (_, i) => add(s, 'u3', 1, 2 + i, 5).id);
  const first = dispatch(createSession(s), fuse('archmage', ids));
  assert.equal(first.present.phase, 'synthesis');
  const decoded = parseSession(JSON.stringify(first));
  const second = dispatch(decoded, fuse('archmage', extra, 2, 5));
  assert.equal(second.present.phase, 'summon');
  assert.equal(second.present.units.filter((u) => u.kind === 'archmage').length, 2);
  assert.deepEqual(undo(second).present, first.present);
  assert.deepEqual(redo(undo(second)).present, second.present);
  assert.equal(second.present.rng, s.rng);
  assert.ok(
    synthesisDestinations(s, SYNTHESIS_RECIPES.find((r) => r.id === 'archmage')!, ids).some(
      (p) => p.x === 2 && p.y === 3,
    ),
  );
});
