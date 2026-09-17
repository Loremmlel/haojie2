import test from 'node:test';
import assert from 'node:assert/strict';
import {
  applyCommand,
  asTarget,
  canSkipReaction,
  getStats,
  hitPullDestination,
  hutSpawnPoints,
  isLegal,
  parseSession,
  createSession,
  firelordStrike,
} from '../../src/engine';
import { addEffect } from '../../src/engine/state';
import { damage, kill, resolution } from '../../src/engine/combat';
import { add, card, fixture, round, strike, unit } from '../helpers';
import { distribution } from '../../src/ai/simulate';

// These contracts exercise rules through real commands where a player has a choice.
// Direct damage/kill calls isolate passive damage packets and causal death chains.
test('2.5 sage aura is live and stackable, heals allies for 25, and on death restores all allied pieces but not bases', () => {
  const s = fixture(),
    sage = add(s, 'sage', 1, 3, 4),
    friend = add(s, 26, 1, 3, 6),
    far = add(s, 26, 1, 9, 12);
  const second = add(s, 'sage', 1, 4, 4);
  friend.hp = 1;
  far.hp = 1;
  s.bases[1] = 100;
  assert.equal(getStats(s, friend).attack, 30);
  assert.equal(getStats(s, sage).attack, 35);
  assert.equal(getStats(s, far).attack, 20);
  second.silenced = true;
  assert.equal(getStats(s, friend).attack, 25);
  const healed = strike(s, sage, friend);
  assert.equal(unit(healed, friend.id).hp, 26);
  const n = structuredClone(s);
  kill(n, unit(n, sage.id), { owner: 2, kind: 'attack' }, resolution());
  assert.equal(unit(n, friend.id).hp, friend.maxHp);
  assert.equal(unit(n, far.id).hp, far.maxHp);
  assert.equal(n.bases[1], 100);
  assert.equal(getStats(n, unit(n, friend.id)).attack, 20);
  const silenced = structuredClone(s);
  unit(silenced, sage.id).silenced = true;
  kill(silenced, unit(silenced, sage.id), { owner: 2, kind: 'attack' }, resolution());
  assert.equal(unit(silenced, friend.id).hp, 1);
  const frozen = structuredClone(s);
  addEffect(frozen, unit(frozen, sage.id), 'freeze', 2, 0, 4);
  assert.equal(getStats(frozen, unit(frozen, friend.id)).attack, 20);
});

test('2.5 formless needs an earlier attack charge, consumes one shot and offers optional target-bound front pull', () => {
  let s = fixture();
  const hook = add(s, 'formless', 1, 3, 4),
    target = add(s, 'grave', 2, 3, 10);
  assert.equal(isLegal(s, { type: 'attack', unitId: hook.id, targetId: target.id }), false);
  s = applyCommand(s, { type: 'charge', unitId: hook.id, mode: 'attack' });
  assert.equal(isLegal(s, { type: 'attack', unitId: hook.id, targetId: target.id }), false);
  s = round(s);
  s = strike(s, unit(s, hook.id), unit(s, target.id));
  assert.equal(unit(s, target.id).hp, 60);
  assert.equal(unit(s, hook.id).charge, 0);
  assert.equal(s.pending[0].kind, 'hit-pull');
  assert.equal(s.pending[0].targetId, target.id);
  assert.deepEqual(hitPullDestination(s, s.pending[0]), { x: 3, y: 5 });
  const saved = parseSession(JSON.stringify(createSession(s))).present;
  const skipped = applyCommand(saved, { type: 'react' });
  assert.equal(unit(skipped, target.id).y, 10);
  assert.equal(skipped.pending.length, 0);
  const pulled = applyCommand(saved, {
    type: 'react',
    mode: 'pull',
    targetId: 'invented-target',
    x: 9,
    y: 9,
  });
  assert.equal(
    unit(pulled, target.id).y,
    5,
    'client coordinates and target IDs cannot replace the recorded victim/front',
  );
  assert.equal(unit(pulled, hook.id).operations, unit(s, hook.id).operations);
  assert.equal(isLegal(pulled, { type: 'attack', unitId: hook.id, targetId: target.id }), false);
});

test('2.5 formless pull handles occupied fronts, immunity, lethal hits and large mirrored targets without losing the skip option', () => {
  const s = fixture(),
    hook = add(s, 'formless', 1, 3, 4),
    target = add(s, 'grave', 2, 3, 9);
  hook.charge = hook.readyCharge = 1;
  hook.chargeType = 'attack';
  add(s, 26, 1, 3, 5);
  const hit = strike(s, hook, target),
    snapshot = structuredClone(hit);
  assert.equal(hitPullDestination(hit, hit.pending[0]), null);
  assert.throws(() => applyCommand(hit, { type: 'react', mode: 'pull' }));
  assert.deepEqual(hit, snapshot);
  assert.equal(canSkipReaction(hit), true);
  assert.equal(applyCommand(hit, { type: 'react' }).pending.length, 0);
  const immune = fixture(),
    h = add(immune, 'formless', 1, 3, 4),
    t = add(immune, 'grave', 2, 3, 9);
  h.charge = h.readyCharge = 1;
  h.chargeType = 'attack';
  addEffect(immune, t, 'immune', 2, 0, 4);
  const blocked = applyCommand(strike(immune, h, t), { type: 'react', mode: 'pull' });
  assert.equal(unit(blocked, t.id).y, 9);
  assert.equal(unit(blocked, t.id).hp, 70);
  const lethal = structuredClone(immune);
  unit(lethal, t.id).effects = [];
  unit(lethal, t.id).hp = 5;
  assert.equal(strike(lethal, unit(lethal, h.id), unit(lethal, t.id)).pending.length, 0);
  const mirror = fixture();
  mirror.active = 2;
  const mh = add(mirror, 'formless', 2, 6, 8),
    big = add(mirror, 5, 1, 3, 3);
  mh.charge = mh.readyCharge = 1;
  mh.chargeType = 'attack';
  const moved = applyCommand(strike(mirror, mh, big), { type: 'react', mode: 'pull' });
  assert.equal(unit(moved, big.id).x, 6);
  assert.equal(unit(moved, big.id).y, 6);
});

test('2.5 slayer pierces a straight ray, drains actual loss at 100%, and has no inherited burn or vampire crits', () => {
  const s = fixture(),
    slayer = add(s, 'slayer', 1, 3, 4),
    a = add(s, 'grave', 2, 3, 6),
    b = add(s, 'grave', 2, 3, 8);
  slayer.hp = 10;
  const hit = strike(s, slayer, a);
  assert.equal(unit(hit, a.id).hp, 60);
  assert.equal(unit(hit, b.id).hp, 60);
  assert.equal(unit(hit, slayer.id).hp, 30);
  assert.equal(unit(hit, slayer.id).shots, 1);
  assert.deepEqual(unit(hit, a.id).effects, []);
  assert.equal(hit.rng, s.rng);
  const diagonal = add(s, 'grave', 2, 5, 6);
  assert.equal(isLegal(s, { type: 'attack', unitId: slayer.id, targetId: diagonal.id }), false);
});

test('2.5 slayer reflection is half actual health loss, survives lethal damage, respects immunity and never reflects reflection', () => {
  const s = fixture(),
    a = add(s, 9, 1, 3, 4),
    b = add(s, 'slayer', 2, 3, 6);
  a.attackBonus = 10;
  const hit = strike(s, a, b);
  assert.equal(unit(hit, b.id).hp, 50);
  assert.equal(unit(hit, a.id).hp, a.hp - 10);
  const lethal = structuredClone(s);
  unit(lethal, b.id).hp = 7;
  const killed = strike(lethal, unit(lethal, a.id), unit(lethal, b.id));
  assert.ok(!killed.units.some((u) => u.id === b.id));
  assert.equal(unit(killed, a.id).hp, a.hp - 3.5);
  const both = fixture(),
    c = add(both, 'slayer', 1, 3, 4),
    d = add(both, 'slayer', 2, 3, 6);
  c.hp = 30;
  const duel = strike(both, c, d);
  assert.equal(unit(duel, c.id).hp, 35);
  assert.equal(unit(duel, d.id).hp, 60);
  const gold = structuredClone(s);
  addEffect(gold, unit(gold, b.id), 'immune', 2, 0, 4);
  assert.equal(unit(strike(gold, unit(gold, a.id), unit(gold, b.id)), a.id).hp, a.hp);
});

test('2.5 citadel observes individual clone deaths and its own spawned runner deaths, paying max HP only with legal placement', () => {
  let s = fixture();
  const city = add(s, 'citadel', 1, 3, 4);
  const a = add(s, 'u25', 1, 3, 6),
    b = add(s, 'u25', 1, 3, 6);
  a.group = b.group = 'clones';
  kill(s, a, { owner: 2, kind: 'spell' }, resolution());
  kill(s, b, { owner: 2, kind: 'spell' }, resolution());
  assert.equal(s.pending.filter((r) => r.source.id === city.id).length, 2);
  assert.equal(s.heads[2], 1);
  assert.equal(canSkipReaction(s), false);
  assert.equal(isLegal(s, { type: 'react' }), false);
  s = applyCommand(s, { type: 'react', x: 3, y: 5 });
  s = applyCommand(s, { type: 'react', x: 4, y: 5 });
  assert.equal(unit(s, city.id).maxHp, 160);
  const runners = s.units.filter((u) => u.kind === 20);
  assert.equal(runners.length, 2);
  const n = structuredClone(s);
  kill(n, n.units.find((u) => u.id === runners[0].id)!, { kind: 'expire' }, resolution());
  assert.equal(n.pending.length, 1, 'runner deaths also produce another city summon');
  const last = structuredClone(s);
  last.pending = [
    { kind: 'hut-spawn', owner: 1, source: structuredClone(unit(last, city.id)), amount: 0 },
  ];
  unit(last, city.id).maxHp = unit(last, city.id).hp = 10;
  const spent = applyCommand(last, { type: 'react', x: 2, y: 4 });
  assert.ok(!spent.units.some((u) => u.id === city.id));
  assert.equal(spent.units.filter((u) => u.kind === 20).length, 3);
  assert.equal(spent.heads[2], s.heads[2]);
  const short = structuredClone(last);
  unit(short, city.id).maxHp = unit(short, city.id).hp = 5;
  assert.equal(hutSpawnPoints(short, short.pending[0]).length, 0);
  assert.equal(unit(applyCommand(short, { type: 'react' }), city.id).maxHp, 5);
});

test('2.5 citadel stale, frozen and completely surrounded reactions can drain safely without paying or duplicating a unit', () => {
  for (const mode of ['dead', 'frozen', 'blocked'] as const) {
    const s = fixture(),
      city = add(s, 'citadel', 1, 4, 4);
    s.pending = [{ kind: 'hut-spawn', owner: 1, source: structuredClone(city), amount: 0 }];
    if (mode === 'dead') s.units = [];
    if (mode === 'frozen') addEffect(s, city, 'freeze', 2, 0, 4);
    if (mode === 'blocked')
      for (const p of [
        { x: 3, y: 4 },
        { x: 5, y: 4 },
        { x: 4, y: 3 },
        { x: 4, y: 5 },
      ])
        add(s, 'grave', 2, p.x, p.y);
    assert.equal(canSkipReaction(s), true, mode);
    const n = applyCommand(s, { type: 'react' });
    assert.equal(n.pending.length, 0);
    assert.equal(
      n.units.some((u) => u.kind === 20),
      false,
    );
    if (mode !== 'dead') assert.equal(unit(n, city.id).maxHp, 180);
  }
});

test('2.5 firelord fires only at its own end, uses square geometry through blockers, and splashes only orthogonal enemies', () => {
  const s = fixture(),
    lord = add(s, 'firelord', 1, 3, 4);
  const main = add(s, 'grave', 2, 5, 7);
  main.maxHp = main.hp = 120;
  const ortho = add(s, 'grave', 2, 6, 7),
    diagonal = add(s, 'grave', 2, 6, 8),
    friend = add(s, 26, 1, 5, 6);
  add(s, 'grave', 2, 3, 5); // direct path blockers do not affect the square trigger
  assert.equal(firelordStrike(s, lord)!.target.id, main.id);
  assert.equal(isLegal(s, { type: 'attack', unitId: lord.id, targetId: main.id }), false);
  const n = applyCommand(s, { type: 'end' });
  assert.equal(unit(n, main.id).hp, 40);
  assert.equal(unit(n, ortho.id).hp, 60);
  assert.equal(unit(n, diagonal.id).hp, 70);
  assert.equal(unit(n, friend.id).hp, friend.hp);
  const enemyEnd = structuredClone(s);
  enemyEnd.active = 2;
  assert.equal(unit(applyCommand(enemyEnd, { type: 'end' }), main.id).hp, 120);
  const silent = structuredClone(s);
  unit(silent, lord.id).silenced = true;
  assert.equal(isLegal(silent, { type: 'attack', unitId: lord.id, targetId: main.id }), false);
  assert.equal(firelordStrike(silent, unit(silent, lord.id)), null);
  const corner = fixture(),
    l = add(corner, 'firelord', 1, 1, 1),
    target = add(corner, 'grave', 2, 7, 7);
  assert.equal(firelordStrike(corner, l)!.target.id, target.id);
});

test('2.5 firelord excludes frozen neutrals, hits enemy stacks and large footprints once per packet, and never fires after dying to end damage', () => {
  const s = fixture(),
    lord = add(s, 'firelord', 1, 3, 4),
    a = add(s, 'u25', 2, 5, 7),
    b = add(s, 'u25', 2, 5, 7);
  a.group = b.group = 'batch';
  a.maxHp = a.hp = 120;
  b.maxHp = b.hp = 120;
  const giant = add(s, 5, 2, 6, 7),
    neutral = add(s, 'grave', 2, 2, 4);
  neutral.maxHp = neutral.hp = 500;
  addEffect(s, neutral, 'freeze', 1, 0, 4, 0);
  const n = applyCommand(s, { type: 'end' });
  assert.equal(unit(n, a.id).hp, 40);
  assert.equal(unit(n, b.id).hp, 40);
  assert.equal(unit(n, giant.id).hp, 101);
  assert.equal(unit(n, neutral.id).hp, 500);
  const dead = structuredClone(s);
  unit(dead, lord.id).hp = 5;
  addEffect(dead, unit(dead, lord.id), 'burn', 2, 0, 8, 5);
  const gone = applyCommand(dead, { type: 'end' });
  assert.equal(unit(gone, a.id).hp, 120);
});

test('2.5 archmage exposes exact two-thirds counter branches, rewards only the successful source and remains a mage', () => {
  const s = fixture(),
    caster = add(s, 26, 1, 3, 4),
    arch = add(s, 'archmage', 2, 8, 11),
    id = card(s, 17);
  arch.hp = 40;
  const result = distribution(s, { type: 'cast', cardId: id, targetId: caster.id }, 20);
  assert.equal(result.sampled, false);
  assert.equal(result.outcomes.length, 2);
  const success = result.outcomes.find((o) => unit(o.state, arch.id).maxHp === 105)!;
  assert.ok(Math.abs(success.weight - 2 / 3) < 1e-12);
  assert.equal(unit(success.state, arch.id).hp, 55);
  assert.equal(unit(success.state, caster.id).effects.length, 0);
  const failure = result.outcomes.find((o) => unit(o.state, arch.id).maxHp === 90)!;
  assert.ok(Math.abs(failure.weight - 1 / 3) < 1e-12);
  assert.equal(unit(failure.state, arch.id).hp, 40);
  assert.equal(unit(failure.state, caster.id).effects.length, 1);
  const equip = fixture(),
    own = add(equip, 'archmage', 1, 3, 4),
    staff = card(equip, 'u5');
  assert.equal(isLegal(equip, { type: 'equip', cardId: staff, targetId: own.id }), true);
  const silent = structuredClone(s);
  unit(silent, arch.id).silenced = true;
  assert.equal(
    distribution(silent, { type: 'cast', cardId: id, targetId: caster.id }).outcomes.length,
    1,
  );
});

test('2.5 ordinary and synthesized counter mages resolve independently in actual entry order, stopping on first success', () => {
  const s = fixture(),
    receiver = add(s, 26, 1, 3, 4),
    old = add(s, 'u3', 2, 7, 10),
    arch = add(s, 'archmage', 2, 8, 10);
  old.deployedAt = 1;
  arch.deployedAt = 3;
  const id = card(s, 17);
  const d = distribution(s, { type: 'cast', cardId: id, targetId: receiver.id }, 20);
  assert.equal(d.outcomes.length, 3);
  const probabilities = d.outcomes.map((o) => o.weight).sort((a, b) => a - b);
  for (const [i, p] of [2 / 9, 1 / 3, 4 / 9].entries())
    assert.ok(Math.abs(probabilities[i] - p) < 1e-12);
  assert.ok(
    Math.abs(
      d.outcomes
        .filter((o) => unit(o.state, arch.id).maxHp === 105)
        .reduce((n, o) => n + o.weight, 0) -
        4 / 9,
    ) < 1e-12,
  );
});
