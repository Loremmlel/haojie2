import test from 'node:test';
import assert from 'node:assert/strict';
import { add, card, fixture, round } from '../helpers';
import { createGame, applyCommand, createSession } from '../../src/engine';
import { asTarget, resetUnit } from '../../src/engine/state';
import { attackPath, basePoint, distance, targets } from '../../src/engine/geometry';
import { hitDistance, actionWindow } from '../../src/ai/spatial';
import { incoming } from '../../src/ai/threats';
import { evaluate, explainEvaluation } from '../../src/ai/evaluate';
import { decide } from '../../src/ai/search';
import { observe } from '../../src/ai/observation';
import { distribution } from '../../src/ai/simulate';
import { Arena } from '../../src/match/arena';
import type { Difficulty } from '../../src/ai/types';
const levels: Difficulty[] = ['easy', 'medium', 'hard'];
const limits = { simulations: 900, milliseconds: 100000 };
const choose = (s: ReturnType<typeof fixture>, d: Difficulty = 'hard') =>
  decide(observe(s), s.pending[0]?.owner ?? s.active, d, limits);

test('incoming danger uses the opponent response window: uncharged cannon zero, ready cannon real', () => {
  const s = fixture(),
    victim = add(s, 7, 1, 4, 5),
    cannon = add(s, 4, 2, 4, 8);
  assert.equal(incoming(s, victim), 0);
  cannon.charge = 2;
  const ready = structuredClone(s);
  assert.equal(incoming(ready, ready.units[0]), 100);
  // Newly deployed ordinary enemies wake next turn; filtering sleeping NOW would also be wrong.
  const fresh = fixture(),
    v = add(fresh, 7, 1, 4, 5),
    a = add(fresh, 26, 2, 4, 7);
  a.born = fresh.turns[2];
  assert.ok(incoming(fresh, v) >= 20);
});
test('real blockers, control expiry, and zero-attack units change incoming pressure correctly', () => {
  const s = fixture(),
    victim = add(s, 7, 1, 4, 5);
  add(s, 3, 2, 4, 6);
  assert.equal(incoming(s, victim), 0);
  const cannon = add(s, 4, 2, 4, 8);
  cannon.charge = 2;
  for (const x of [1, 2, 3, 4, 5, 6, 7, 8, 9]) add(s, 'grave', 1, x, 7);
  assert.equal(incoming(structuredClone(s), victim), 0);
  const frozen = fixture(),
    v = add(frozen, 7, 1, 4, 5),
    enemy = add(frozen, 26, 2, 4, 6);
  enemy.effects.push({ type: 'freeze', owner: 1, from: 5, until: 9 });
  assert.equal(incoming(frozen, v), 0);
  const thaw = structuredClone(frozen);
  thaw.units[1].effects[0].until = 6;
  assert.ok(incoming(thaw, thaw.units[0]) > 0);
});
test('cached hit fields agree with engine pathfinding around normal, large, frozen and stacked targets', () => {
  for (let seed = 1; seed <= 6; seed++) {
    const s = fixture();
    add(s, 9, 1, 1, 2);
    add(s, 4, 2, 7, 11).charge = 2;
    add(s, 5, 2, 4, 6);
    add(s, 7, 1, 3, 5);
    const f = add(s, 26, 1, 5, 4);
    if (seed % 2) f.effects.push({ type: 'freeze', owner: 2, from: 0, until: 100 });
    for (let i = 0; i < 6; i++)
      add(s, 'grave', ((i % 2) + 1) as 1 | 2, 1 + i, 8 + ((i + seed) % 3));
    const view = actionWindow(s, 1, true);
    for (const u of view.units)
      for (const t of targets(view)) {
        // Test each unit at its native range via a shared state, without mutating cached objects.
        const range =
          u.kind === 9 ? 5 : u.kind === 'grave' ? 0 : u.kind === 7 ? 3 : u.kind === 5 ? 2 : 4;
        assert.equal(
          Number.isFinite(hitDistance(view, u, t)),
          !!attackPath(view, u, t, range),
          `${seed}/${u.id}/${t.id}`,
        );
      }
  }
});
test('all levels establish safe contact instead of fleeing the uncharged opening cannon', () => {
  for (const d of levels) {
    const s = createGame(20260907);
    s.phase = 'play';
    s.summonSlots = 0;
    s.active = 2;
    s.ply = 2;
    s.turns = { 1: 1, 2: 1 };
    const cannon = add(s, 4, 1, 5, 8),
      straight = add(s, 13, 1, 4, 8);
    cannon.born = straight.born = 1;
    card(s, 7);
    const before = structuredClone(s),
      result = choose(s, d);
    assert.ok(result.command);
    const next = applyCommand(s, result.command),
      deployed = next.units.find((u) => u.owner === 2);
    assert.ok(deployed, JSON.stringify(result.command));
    assert.ok(deployed.y <= 10, `${d}: (${deployed.x},${deployed.y})`);
    assert.deepEqual(s, before);
  }
});
test('all levels spend affordable head stockpiles on a genuinely stronger expected draw', () => {
  for (const difficulty of levels)
    for (const heads of [2, 4, 16]) {
      const s = createGame(37);
      s.heads[1] = heads;
      const d = choose(s, difficulty);
      assert.equal(d.command?.type, 'summon');
      assert.equal(d.command?.ultimate, true, `${difficulty}/${heads}`);
    }
});
test('upgraded summon comparison enumerates both full public pools including transformations and clones', () => {
  const s = createGame(4);
  s.heads[1] = 2;
  for (const ultimate of [false, true]) {
    const d = distribution(s, { type: 'summon', ultimate }, 64, 4);
    assert.equal(d.sampled, false);
    assert.ok(d.outcomes.length > 26);
    assert.ok(Math.abs(d.outcomes.reduce((n, o) => n + o.weight, 0) - 1) < 1e-10);
    if (ultimate)
      assert.ok(
        d.outcomes.some((o) => o.state.hands[1].filter((c) => c.kind === 'u25').length === 8),
      );
  }
});
test('catapults do not mark a damaged unsupported base while a useful full-health unit exists', () => {
  for (const d of levels) {
    const s = fixture(),
      cat = add(s, 10, 1, 2, 4),
      target = add(s, 4, 2, 4, 7);
    target.charge = 2;
    s.bases[2] = 295;
    const result = choose(s, d);
    assert.ok(result.command);
    assert.notEqual(result.command.targetId, 'base-2');
    if (result.command.type === 'attack') assert.equal(result.command.targetId, target.id);
  }
});
test('catapult plus non-catapult follow-up redeems a mark within its lifetime', () => {
  const s = fixture(),
    cat = add(s, 10, 1, 2, 4),
    archer = add(s, 9, 1, 3, 4),
    target = add(s, 4, 2, 4, 7);
  target.hp = 30;
  s.bases[2] = 295;
  const result = choose(s, 'medium');
  let next = s;
  for (const step of result.plan) next = applyCommand(next, step.command);
  assert.ok(next.units.find((u) => u.id === target.id)!.hp <= 15, JSON.stringify(result.plan));
});
test('line spells can choose the thirteenth row even at the smallest candidate budget', () => {
  for (const d of levels) {
    const s = fixture();
    add(s, 1, 2, 2, 13).hp = 20;
    add(s, 9, 2, 7, 13).hp = 20;
    const id = card(s, 'u9');
    const result = choose(s, d);
    assert.equal(result.command?.type, 'cast');
    assert.equal(result.command?.cardId, id);
    assert.equal(result.command?.row, 13);
  }
});
test('conversion preparation prefers a surviving victim and reachable carrier over lethal overkill', () => {
  const s = fixture(),
    weak = add(s, 9, 1, 4, 4),
    strong = add(s, 4, 1, 6, 4),
    victim = add(s, 4, 2, 5, 6);
  strong.charge = strong.readyCharge = 2;
  const id = card(s, 22);
  const result = choose(s, 'medium');
  // Killing now can correctly beat preparation. Whenever conversion is chosen it must not go on the overkill cannon.
  for (const step of result.plan)
    if (step.command.type === 'cast' && step.command.cardId === id)
      assert.equal(step.command.targetId, weak.id);
  const safe = structuredClone(s);
  safe.units = safe.units.filter((u) => u.id !== strong.id);
  const chosen = choose(safe, 'medium'),
    cast = chosen.plan.find((p) => p.command.cardId === id);
  assert.ok(cast, JSON.stringify(chosen.plan));
  assert.equal(cast.command.targetId, weak.id);
});
test('a clone army develops multiple lanes rather than seven rear units on one square', () => {
  let s = fixture();
  s.active = 2;
  s.ply = 6;
  s.turns = { 1: 3, 2: 3 };
  add(s, 9, 1, 5, 5);
  for (let i = 0; i < 8; i++) {
    const id = card(s, 'u25');
    s.hands[2].find((c) => c.id === id)!.group = 'batch';
  }
  for (let i = 0; i < 10 && s.hands[2].length; i++) {
    const r = decide(observe(s), 2, 'easy', { simulations: 200, milliseconds: 100000 });
    assert.ok(r.command);
    s = applyCommand(s, r.command);
  }
  const clones = s.units.filter((u) => u.owner === 2),
    counts = new Map<string, number>();
  for (const u of clones) {
    const k = `${u.x},${u.y}`;
    counts.set(k, (counts.get(k) ?? 0) + 1);
  }
  assert.equal(clones.length, 8);
  assert.ok(counts.size >= 4, JSON.stringify([...counts]));
  assert.ok(Math.max(...counts.values()) <= 3);
});
test('a distant mobile fighter advances toward contact instead of oscillating without threats', () => {
  let s = fixture();
  const u = add(s, 26, 1, 5, 3);
  const positions = [u.y];
  for (let turn = 0; turn < 3; turn++) {
    const r = choose(s, 'medium');
    assert.equal(r.command?.type, 'move');
    s = applyCommand(s, r.command!);
    positions.push(s.units[0].y);
    s = round(s);
  }
  assert.ok(
    positions.every((y, i) => i === 0 || y > positions[i - 1]),
    JSON.stringify(positions),
  );
});
test('headless arena respects reaction ownership and emits auditable public decision traces', () => {
  const s = fixture(),
    source = add(s, 2, 1, 1, 1);
  s.units = [];
  s.active = 2;
  s.pending.push({ kind: 'death-shot', owner: 1, source, amount: 20 });
  add(s, 1, 2, 2, 2);
  const a = new Arena({ ...createSession(s), match: { mode: 'ai', human: 1, difficulty: 'hard' } });
  assert.equal(a.step(), null);
  const before = JSON.stringify(a.session);
  assert.throws(() => a.play({ type: 'end' }));
  assert.equal(JSON.stringify(a.session), before);
  a.play({ type: 'react', targetId: s.units[0].id });
  const r = a.step({ ...limits, trace: true });
  assert.ok(r?.decision?.trace);
  assert.equal('rng' in r.decision.trace.initial, false);
  const terms = explainEvaluation(s, 1);
  assert.ok(Math.abs(terms.total - evaluate(s, 1)) < 1e-8);
});

test('complete firing operations clear a screen and unlock a loaded ally instead of missing a six-hit combination', () => {
  for (const difficulty of levels) {
    const s = fixture(),
      gun = add(s, 4, 1, 5, 9),
      lone = add(s, 23, 1, 7, 9);
    gun.charge = gun.readyCharge = 2;
    lone.born = 1;
    s.bases[2] = 100;
    for (let x = 1; x <= 9; x++) add(s, 'grave', 2, x, 11).hp = x === 5 ? 30 : 70;
    const result = decide(observe(s), 1, difficulty, { simulations: 1800, milliseconds: 100000 });
    assert.ok(result.command);
    let next = s;
    for (const step of result.plan) next = applyCommand(next, step.command);
    // It may choose the cannon for another legal immediate win; actual end state is the contract.
    for (let i = 0; i < 8 && !next.winner; i++) {
      const r = decide(observe(next), 1, difficulty, { simulations: 500, milliseconds: 100000 });
      if (!r.command) break;
      next = applyCommand(next, r.command);
    }
    assert.equal(next.winner, 1, JSON.stringify(result.plan));
  }
});
test('idle zero-damage hits and full-health heals cannot consume an AI turn while real on-hit effects remain', () => {
  const s = fixture(),
    hut = add(s, 'u22', 1, 3, 4),
    nurse = add(s, 2, 1, 2, 4),
    king = add(s, 'u18', 2, 3, 6),
    cat = add(s, 10, 1, 5, 3);
  s.bases[2] = 295;
  king.hp = king.maxHp;
  const r = choose(s, 'medium');
  for (const step of r.plan) {
    assert.ok(!(step.command.type === 'attack' && step.command.unitId === hut.id));
    assert.ok(
      !(
        step.command.type === 'attack' &&
        step.command.unitId === nurse.id &&
        step.command.targetId === nurse.id
      ),
    );
    assert.ok(
      !(
        step.command.type === 'attack' &&
        step.command.unitId === cat.id &&
        step.command.targetId === king.id
      ),
    );
  }
});

test('damage to a threatened enemy never makes that enemy a more valuable asset; charge is expendable power', () => {
  for (const kind of ['grave', 1, 7] as const) {
    const s = fixture(),
      cannon = add(s, 4, 1, 4, 4),
      victim = add(s, kind, 2, 4, 6);
    cannon.charge = cannon.readyCharge = 2;
    const before = evaluate(s, 1),
      after = structuredClone(s);
    after.units.find((u) => u.id === victim.id)!.hp -= 5;
    assert.ok(
      evaluate(after, 1) > before,
      `${kind}: wounding an enemy must not improve their position`,
    );
  }
  const s = fixture(),
    u = add(s, 'u2', 1, 4, 4),
    foe = add(s, 1, 2, 4, 6);
  u.charge = u.readyCharge = 2;
  foe.hp = 40;
  const d = choose(s, 'medium');
  assert.equal(d.command?.type, 'attack');
  assert.equal(d.command?.targetId, foe.id);
});
