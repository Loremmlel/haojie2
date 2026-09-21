import assert from 'node:assert/strict';
import test from 'node:test';
import {
  ALL_CELLS,
  actorCommandError,
  applyCommand,
  applyPlayerCommand,
  canAttemptCommand,
  canDeployKind,
  canRebasePlayerCommand,
  commandError,
  createGame,
  expansionAnchors,
  getPlayerView,
  inspectCommand,
  parseCommand,
  queryCommandError,
  template,
  type Command,
  type GameState,
  type Player,
} from '../../src/engine';
import { observe } from '../../src/ai/observation';
import { selectOnlineUpdate, updateError } from '../../src/ui/online/types';
import { DemoRoom } from '../../examples/online/room';
import { add, fixture, unit } from '../helpers';
const wire = <T>(v: T): T => JSON.parse(JSON.stringify(v)) as T;

function assertNoPrivate(value: unknown) {
  if (!value || typeof value !== 'object') return;
  for (const [key, child] of Object.entries(value)) {
    assert.ok(!['seed', 'rng', 'past', 'future', 'secret'].includes(key), `leaked ${key}`);
    assertNoPrivate(child);
  }
}

test('foreign command parsing rejects malformed parameters without touching authority', () => {
  const s = fixture(),
    before = structuredClone(s);
  const invalid: unknown[] = [
    null,
    [],
    {},
    { type: 'unknown' },
    { type: 'end', rng: 1 },
    { type: 'move', x: NaN },
    { type: 'move', x: '3' },
    { type: 'move', x: 0 },
    { type: 'attack', path: [{ x: 1, y: 2, secret: 'x' }] },
    { type: 'attack', path: Array(257).fill({ x: 1, y: 1 }) },
    { type: 'summon', ultimate: 1 },
    { type: 'choose-shrine', shrineKind: 'not-a-kind' },
    { type: 'skill', unitId: '' },
    { type: 'synthesize', materialIds: ['a', 'a'] },
    JSON.parse('{"type":"end","__proto__":{"admin":true}}'),
  ];
  for (const input of invalid) assert.throws(() => applyPlayerCommand(s, 1, input));
  assert.throws(() => applyPlayerCommand(s, 2, { type: 'end' }));
  assert.throws(() => applyPlayerCommand(s, 1, { type: 'end', player: 2 }));
  assert.deepEqual(s, before);
  assert.deepEqual(
    parseCommand({
      type: 'attack',
      unitId: 'u1',
      path: [
        { x: 1, y: 1 },
        { x: 1, y: 2 },
      ],
    }).path,
    [
      { x: 1, y: 1 },
      { x: 1, y: 2 },
    ],
  );
});

test('authenticated seat cannot operate an enemy piece even when the rule is out-of-turn', () => {
  const s = fixture(),
    enemy = add(s, 'u7', 2, 2, 8),
    target = add(s, 1, 1, 5, 8);
  const anchor = expansionAnchors(s, target)[0];
  assert.ok(anchor);
  const c: Command = { type: 'skill', unitId: enemy.id, targetId: target.id, ...anchor };
  const before = structuredClone(s);
  assert.ok(actorCommandError(s, 1, c));
  assert.throws(() => applyPlayerCommand(s, 1, c));
  assert.equal(actorCommandError(s, 2, c), null);
  const next = applyPlayerCommand(s, 2, c);
  assert.equal(unit(next, target.id).size, 2);
  assert.equal(unit(next, target.id).owner, 1);
  assert.equal(unit(next, enemy.id).hp, enemy.hp - 10);
  assert.deepEqual(next, applyCommand(s, c));
  assert.deepEqual(s, before);
  assert.throws(() => applyPlayerCommand(next, 2, c));
});

test('reaction owner can differ from active and does not grant arbitrary other commands', () => {
  const s = fixture();
  s.pending = [
    {
      kind: 'death-shot',
      owner: 2,
      source: template(8, 2, 0, { x: 5, y: 8 }, 'dead-source'),
      amount: 10,
    },
  ];
  const before = structuredClone(s);
  assert.throws(() => applyPlayerCommand(s, 1, { type: 'react' }));
  assert.throws(() => applyPlayerCommand(s, 2, { type: 'end' }));
  assert.equal(applyPlayerCommand(s, 2, { type: 'react' }).pending.length, 0);
  assert.deepEqual(s, before);
});

test('both shrine seats may choose independently; spoofed player is rejected and selection reveals together', () => {
  const initial = createGame(90, 'shrine');
  const first = {
    type: 'choose-shrine' as const,
    shrineKind: initial.shrineDraft!.offers[2][0],
    parity: 'even' as const,
  };
  assert.throws(() => applyPlayerCommand(initial, 2, { ...first, player: 1 }));
  assert.equal(canRebasePlayerCommand(initial, 2, first), true);
  const s = applyPlayerCommand(initial, 2, first);
  assert.equal(s.shrineDraft!.committed[2], true);
  assert.equal(s.shrineDraft!.revealed, false);
  assert.equal(getPlayerView(s, 1).state.shrineDraft!.choices[2], undefined);
  assert.deepEqual(getPlayerView(s, 2).state.shrineDraft!.choices[2], s.shrineDraft!.choices[2]);
  assert.equal(canRebasePlayerCommand(s, 2, first), false);
  const final = applyPlayerCommand(s, 1, {
    type: 'choose-shrine',
    shrineKind: s.shrineDraft!.offers[1][0],
    parity: 'odd',
  });
  assert.equal(final.phase, 'shrine-setup');
  assert.equal(final.shrineDraft!.revealed, true);
  assert.deepEqual(
    getPlayerView(final, 1).state.shrineDraft,
    getPlayerView(final, 2).state.shrineDraft,
  );
});

test('unrevealed Jade parity stays private through observation, public logs and public events', () => {
  const initial = createGame(123, 'shrine');
  initial.shrineDraft!.offers[2] = ['s9', 's1', 's2'];
  const s = applyPlayerCommand(initial, 2, {
    type: 'choose-shrine',
    shrineKind: 's9',
    parity: 'even',
  });
  s.log.push('SECRET-PARITY-even');
  s.events.push({ id: 'private-debug', type: 'skill', text: 'SECRET-PARITY-even', ability: 's9' });
  const opponent = getPlayerView(s, 1),
    own = getPlayerView(s, 2);
  assert.equal(opponent.state.shrineDraft!.choices[2], undefined);
  assert.equal(own.state.shrineDraft!.choices[2]!.parity, 'even');
  assert.deepEqual(opponent.state.shrineDraft!.offers[2], ['s9', 's1', 's2']);
  assert.equal(JSON.stringify(opponent).includes('SECRET-PARITY'), false);
  assert.equal(observe(s, 1).shrineDraft!.choices[2], undefined);
  assertNoPrivate(opponent);
  assertNoPrivate(own);
  assert.deepEqual(opponent.state.shrineDraft, observe(s, 1).shrineDraft);
});

test('nested unit, reaction, clock, effect and event snapshots are explicitly projected and detached', () => {
  const s = fixture(),
    u = add(s, 1, 1, 4, 5);
  Object.assign(u, { secret: 'not public', rng: 77 });
  u.effects.push(
    Object.assign(
      { type: 'attack' as const, from: 1, until: 10, owner: 1 as const, amount: 5 },
      { secret: 'effect' },
    ),
  );
  s.pending = [{ kind: 'death-shot', owner: 1, source: u, amount: 10 }];
  s.clockFrames = { 1: { current: { ply: s.ply, turns: s.turns, units: [u] } }, 2: {} };
  s.events = [
    {
      id: 'public-event',
      type: 'skill',
      actor: Object.assign(
        { id: u.id, kind: u.kind, owner: u.owner, size: u.size, x: u.x, y: u.y },
        { secret: 'actor' },
      ),
      from: Object.assign({ x: 4, y: 5 }, { secret: 'coordinate' }),
    },
  ];
  const before = JSON.stringify(s);
  const view = wire(getPlayerView(s, 2));
  assertNoPrivate(view);
  assert.equal(view.state.units[0].effects[0].amount, 5);
  assert.equal(view.state.clockFrames![1].current!.units[0].id, u.id);
  view.state.units[0].hp = 1;
  view.state.pending[0].source.effects[0].amount = 999;
  assert.equal(JSON.stringify(s), before);
});

test('ordinary public position keeps all present game information except random fields', () => {
  const s = fixture();
  add(s, 'u7', 1, 2, 4);
  add(s, 1, 2, 6, 8);
  s.clockFrames = { 1: { current: { ply: 5, turns: s.turns, units: s.units } }, 2: {} };
  const { seed: _seed, rng: _rng, ...expected } = wire(s);
  assert.deepEqual(wire(getPlayerView(s, 1).state), expected);
  assertNoPrivate(getPlayerView(s, 1));
});

test('public preflight stops before randomness instead of inventing random fields or results', () => {
  const s = createGame(7),
    view = getPlayerView(s, 1),
    before = wire(view);
  assert.equal(inspectCommand(view.state, { type: 'summon' }).status, 'uncertain');
  assert.equal(queryCommandError(view.state, { type: 'summon' }), null);
  assert.equal(view.state.hands[1].length, 0);
  assert.deepEqual(view, before);
  // @ts-expect-error 公开局面不能用作权威 GameState。
  const misuse = () => applyCommand(view.state, { type: 'summon' });
  assert.throws(misuse, /完整随机状态/);
  const p = fixture(),
    u = add(p, 9, 1, 4, 5);
  const valid = ALL_CELLS.map((to) => ({ type: 'move' as const, unitId: u.id, ...to })).find(
    (c) => commandError(p, c) === null,
  )!;
  assert.ok(valid);
  assert.equal(canAttemptCommand(getPlayerView(p, 1).state, valid), true);
  assert.equal(inspectCommand(getPlayerView(p, 1).state, { ...valid, x: 0 }).status, 'invalid');
  assert.deepEqual(p.units[0], u);
});

test('public secret-choice preflight can validate the second seat without reading the first choice', () => {
  let s = createGame(90, 'shrine');
  s = applyPlayerCommand(s, 2, {
    type: 'choose-shrine',
    shrineKind: s.shrineDraft!.offers[2][0],
    parity: 'odd',
  });
  const view = getPlayerView(s, 1);
  assert.equal(view.state.shrineDraft!.choices[2], undefined);
  assert.equal(
    inspectCommand(view.state, {
      type: 'choose-shrine',
      player: 1,
      shrineKind: view.state.shrineDraft!.offers[1][0],
      parity: 'odd',
    }).status,
    'available',
  );
  assert.equal(view.state.shrineDraft!.revealed, false);
});

test('local random transitions, turns and deployments are identical through the authority adapter', () => {
  for (const seed of [19, 90, 20260920]) {
    let local = createGame(seed),
      online = structuredClone(local);
    const step = (c: Command) => {
      const actor: Player = local.pending[0]?.owner ?? local.active;
      assert.equal(queryCommandError(local, c), commandError(local, c));
      local = applyCommand(local, c);
      online = applyPlayerCommand(online, actor, wire(c));
      assert.deepEqual(online, local);
    };
    for (let turn = 0; turn < 4; turn++) {
      if (local.phase === 'synthesis') step({ type: 'skip-synthesis' });
      while (local.summonSlots > 0) step({ type: 'summon' });
      step({ type: 'begin' });
      for (const card of [...local.hands[local.active]]) {
        if (!canDeployKind(card.kind)) continue;
        const c = ALL_CELLS.map((p) => ({ type: 'deploy' as const, cardId: card.id, ...p })).find(
          (candidate) => commandError(local, candidate) === null,
        );
        assert.ok(c, 'fixture hand must have a deployable destination');
        step(c);
      }
      step({ type: 'end' });
    }
  }
});

test('host deduplicates the same submission and does not replay stale moves or summons', () => {
  const room = new DemoRoom('room', createGame(19));
  const request = { requestId: 'one', baseRevision: 0, command: { type: 'summon' } };
  const first = room.submit(1, request),
    after = room.inspect();
  assert.deepEqual(first, { ok: true, revision: 1 });
  assert.deepEqual(room.submit(1, request), first);
  assert.deepEqual(room.inspect(), after);
  assert.equal(room.submit(1, { ...request, command: { type: 'end' } }).ok, false);
  assert.equal(room.submit(1, { ...request, requestId: 'stale' }).ok, false);
  assert.equal(room.revision, 1);
});

test('same-revision independent secret selections are rebased only by game-owned policy', () => {
  const initial = createGame(90, 'shrine'),
    room = new DemoRoom('shrine', initial);
  for (const actor of [2, 1] as Player[]) {
    const receipt = room.submit(actor, {
      requestId: `seat-${actor}`,
      baseRevision: 0,
      command: {
        type: 'choose-shrine',
        shrineKind: initial.shrineDraft!.offers[actor][0],
        parity: 'odd',
      },
    });
    assert.equal(receipt.ok, true);
  }
  assert.equal(room.revision, 2);
  assert.equal(room.inspect().phase, 'shrine-setup');
  assert.equal(canRebasePlayerCommand(room.inspect(), 1, { type: 'end' }), false);
});

test('controlled update selection rejects incompatible versions and ignores equal/older revisions', () => {
  const room = new DemoRoom('room', createGame(19));
  const zero = room.update(1, 'snapshot');
  room.submit(1, { requestId: 'one', baseRevision: 0, command: { type: 'summon' } });
  const one = room.update(1);
  assert.equal(selectOnlineUpdate(zero, one), one);
  assert.equal(selectOnlineUpdate(one, wire(one)), one);
  assert.equal(selectOnlineUpdate(one, zero), one);
  assert.ok(
    updateError({ ...one, view: { ...one.view, ruleset: 'wrong' } } as unknown as typeof one),
  );
  assert.throws(() => selectOnlineUpdate(one, { ...one, matchId: 'another' }));
});
