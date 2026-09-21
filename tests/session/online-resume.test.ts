import assert from 'node:assert/strict';
import test from 'node:test';
import {
  applyPlayerCommand, createGame, getPlayerView, HAOJIE_RULESET, validState,
  type Command, type GameState, type Player,
} from '../../src/engine';
import { selectOnlineUpdate, updateError, type OnlineUpdate } from '../../src/ui/online/types';
import { add, fixture } from '../helpers';
const wire = <T>(v: T): T => JSON.parse(JSON.stringify(v)) as T;

function resume(s: GameState, actor: Player, command: Command) {
  // The server owns this envelope. Neither seed/rng nor the authoritative save goes to a browser.
  const saved = wire({ ruleset: HAOJIE_RULESET, revision: 42, state: s });
  assert.ok(validState(saved.state));
  const next = applyPlayerCommand(saved.state, actor, wire(command));
  assert.deepEqual(wire(next), wire(applyPlayerCommand(s, actor, command)));
  assert.equal(next.rng, applyPlayerCommand(s, actor, command).rng);
  for (const viewer of [1, 2] as const) {
    const snapshot: OnlineUpdate = { matchId: 'resume-fixture', revision: saved.revision, kind: 'snapshot', view: getPlayerView(saved.state, viewer) };
    assert.equal(updateError(snapshot), null);
    assert.equal('rng' in snapshot.view.state, false);
    const update: OnlineUpdate = { ...snapshot, revision: 43, kind: 'update', view: getPlayerView(next, viewer) };
    assert.equal(selectOnlineUpdate(snapshot, update), update);
    assert.equal(selectOnlineUpdate(update, snapshot), update);
  }
  return saved.state;
}

test('online resume preserves RNG and next random summon in both modes', () => {
  resume(createGame(20260921), 1, { type: 'summon' });
  const s = createGame(20260921, 'shrine');
  const partial = applyPlayerCommand(s, 2, { type: 'choose-shrine', shrineKind: s.shrineDraft!.offers[2][0], parity: 'even' });
  const loaded = resume(partial, 1, { type: 'choose-shrine', shrineKind: partial.shrineDraft!.offers[1][0], parity: 'odd' });
  assert.deepEqual(loaded.shrineDraft!.choices[2], partial.shrineDraft!.choices[2]);
  assert.equal(getPlayerView(loaded, 1).state.shrineDraft!.choices[2], undefined);
  assert.deepEqual(getPlayerView(loaded, 2).state.shrineDraft!.choices[2], partial.shrineDraft!.choices[2]);
});

test('online resume preserves reaction owner, fixed victim and charged resources mid-turn', () => {
  const s = fixture(); s.active = 2;
  const hook = add(s, 'formless', 2, 3, 9);
  const target = add(s, 1, 1, 3, 4);
  hook.charge = hook.readyCharge = 1; hook.chargeType = 'attack';
  const hit = applyPlayerCommand(s, 2, { type: 'attack', unitId: hook.id, targetId: target.id });
  assert.equal(hit.pending[0].kind, 'hit-pull');
  resume(hit, 2, { type: 'react', mode: 'pull' });
  assert.throws(() => applyPlayerCommand(wire(hit), 1, { type: 'react', mode: 'pull' }));
});

test('online resume rejects mismatched rulesets instead of silently interpreting an old snapshot', () => {
  const update: OnlineUpdate = { matchId: 'saved', revision: 42, kind: 'snapshot', view: getPlayerView(createGame(1), 1) };
  const stale = wire(update);
  Object.assign(stale.view, { ruleset: '3.0-feedback3' });
  assert.match(updateError(stale) ?? '', /版本不一致/);
});
