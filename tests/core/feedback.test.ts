import test from 'node:test';
import assert from 'node:assert/strict';
import {
  applyCommand,
  attackRoutes,
  attackPath,
  asTarget,
  canPlace,
  cardActions,
  commandError,
  createGame,
  createSession,
  definition,
  dispatch,
  getStats,
  parseSession,
  pathDirection,
  redo,
  template,
  undo,
  unitActions,
} from '../../src/engine';
import type { AttackDirection, Command, Source } from '../../src/engine';
import { damage, resolution } from '../../src/engine/commands/combat';
import { withRandomSource } from '../../src/engine/core/random';
import {
  advanceIntent,
  canChoose,
  commandFor,
  intentRoutes,
  startIntent,
} from '../../src/ui/game/interaction/selection';
import { add, card, fixture, round, unit } from '../helpers';

test('feedback 01: charge and normal deployment are explicit alternatives before a board selection', () => {
  const s = fixture(),
    id = card(s, 1);
  const choices = cardActions(s, s.hands[1][0]).filter((a) => a.command.type === 'deploy');
  assert.equal(choices.length, 2);
  assert.deepEqual(
    choices.map((a) => a.command.charge),
    [false, true],
  );
  for (const a of choices) {
    const c = commandFor(s, startIntent(a), { x: 2, y: 3 })!;
    const n = applyCommand(s, c),
      u = n.units[0];
    assert.equal(u.hp, c.charge ? 40 : 50);
    assert.equal(u.maxHp, u.hp);
    assert.equal(getStats(n, u).sleeping, !c.charge);
    assert.equal(commandError(n, { type: 'move', unitId: u.id, x: 2, y: 4 }) === null, !!c.charge);
  }
  const original = structuredClone(s);
  assert.throws(() => applyCommand(s, { type: 'deploy', cardId: id, charge: true, x: 5, y: 1 }));
  assert.deepEqual(s, original);
  const played = dispatch(createSession(s), {
    type: 'deploy',
    cardId: id,
    charge: true,
    x: 2,
    y: 3,
  });
  assert.deepEqual(redo(undo(played)).present, played.present);
});

test('feedback paths: a longer in-range flank is legal, and the target cannot be crossed to reach its back', () => {
  const s = fixture(),
    a = add(s, 26, 1, 4, 4),
    b = add(s, 24, 2, 4, 6);
  const routes = attackRoutes(s, a, asTarget(b), 4);
  assert.equal(routes.find((r) => r.direction === 'down')!.path.length - 1, 2);
  assert.equal(routes.find((r) => r.direction === 'left')!.path.length - 1, 4);
  assert.equal(routes.find((r) => r.direction === 'right')!.path.length - 1, 4);
  assert.equal(
    routes.some((r) => r.direction === 'up'),
    false,
  ); // 绕到背面需要六步，不能穿过目标。
  for (const r of routes) {
    assert.equal(pathDirection(r.path), r.direction);
    assert.ok(r.path.slice(0, -1).every((p) => p.x !== b.x || p.y !== b.y));
  }
  const command: Command = { type: 'attack', unitId: a.id, targetId: b.id };
  assert.equal(unit(applyCommand(s, command), b.id).hp, 40);
  assert.equal(unit(applyCommand(s, { ...command, direction: 'left' }), b.id).hp, 30);
  const original = structuredClone(s);
  assert.throws(() => applyCommand(s, { ...command, direction: 'up' }));
  assert.throws(() => applyCommand(s, { ...command, direction: 'invalid' as AttackDirection }));
  assert.deepEqual(s, original);
  a.rangeBonus = -1;
  assert.equal(attackPath(s, a, asTarget(b), 3, 'left'), null);
});

test('feedback paths: directional BFS obeys blockers, board edges, friendly traversal and 2x2 target footprint', () => {
  const s = fixture(),
    a = add(s, 26, 1, 4, 4),
    b = add(s, 24, 2, 4, 6);
  const blocker = add(s, 'grave', 2, 5, 6);
  assert.equal(attackPath(s, a, asTarget(b), 4, 'left'), null);
  blocker.owner = 1;
  assert.ok(attackPath(s, a, asTarget(b), 4, 'left'));
  b.size = 2;
  const path = attackPath(s, a, asTarget(b), 4, 'right')!;
  assert.ok(path);
  assert.equal(path.at(-1)!.x, 4);
  assert.ok(path.slice(0, -1).every((p) => !(p.x >= 4 && p.x <= 5 && p.y >= 6 && p.y <= 7)));
  a.x = 1;
  a.y = 1;
  b.x = 1;
  b.y = 3;
  b.size = 1;
  assert.equal(attackPath(s, a, asTarget(b), 4, 'right'), null);
});

test('feedback paths: mirrored frontal protection and knockback directions remain available with Heart', () => {
  const s = fixture();
  s.active = 2;
  const a = add(s, 26, 2, 4, 8),
    b = add(s, 24, 1, 4, 6);
  const c: Command = { type: 'attack', unitId: a.id, targetId: b.id };
  assert.equal(unit(applyCommand(s, { ...c, direction: 'up' }), b.id).hp, 40);
  assert.equal(unit(applyCommand(s, { ...c, direction: 'left' }), b.id).hp, 30);
  a.equipment.push('u28');
  assert.equal(unit(applyCommand(s, { ...c, direction: 'left' }), b.id).hp, 30);
  const n = fixture(),
    mage = add(n, 'u20', 1, 4, 4),
    target = add(n, 'grave', 2, 4, 6);
  for (const [direction, x, y] of [
    ['down', 4, 8],
    ['left', 2, 6],
    ['right', 6, 6],
  ] as const) {
    const after = applyCommand(n, {
      type: 'attack',
      unitId: mage.id,
      targetId: target.id,
      direction,
    });
    assert.deepEqual([unit(after, target.id).x, unit(after, target.id).y], [x, y]);
    assert.equal(pathDirection(after.events.find((e) => e.type === 'attack')!.path!), direction);
  }
});

test('feedback paths: choosing a target does not attack before selecting a direction; cancellation is state-free', () => {
  const s = fixture(),
    a = add(s, 26, 1, 4, 4),
    b = add(s, 24, 2, 4, 6);
  const intent = startIntent(unitActions(s, a).find((a) => a.id === 'attack')!);
  const original = structuredClone(s);
  assert.equal(canChoose(s, intent, b), true);
  assert.equal(commandFor(s, intent, b), null);
  const choosing = advanceIntent(intent, b, s);
  const route = intentRoutes(s, choosing).find((r) => r.direction === 'left')!;
  assert.equal(canChoose(s, choosing, b), false);
  assert.equal(canChoose(s, choosing, route.path.at(-2)!), true);
  assert.equal(commandFor(s, choosing, route.path.at(-2)!)!.direction, 'left');
  assert.deepEqual(s, original);
});

test('feedback 23: only initial extra fatigue, then consecutive own turns are actionable', () => {
  let s = fixture();
  s = applyCommand(s, { type: 'deploy', cardId: card(s, 23), x: 4, y: 5 });
  const id = s.units[0].id;
  assert.equal(getStats(s, unit(s, id)).sleeping, true);
  s = round(s);
  assert.equal(getStats(s, unit(s, id)).sleeping, true);
  for (let i = 0; i < 4; i++) {
    s = round(s);
    assert.equal(getStats(s, unit(s, id)).remaining, 6);
    s = applyCommand(s, { type: 'move', unitId: id, x: 4 + (i % 2), y: 6 });
    assert.equal(getStats(s, unit(s, id)).operationsLeft, 0);
  }
});

test('feedback 23: all eight friendly deployment and movement neighbors remain forbidden', () => {
  const s = fixture(),
    lone = add(s, 23, 1, 4, 5);
  for (let dx = -1; dx <= 1; dx++)
    for (let dy = -1; dy <= 1; dy++) {
      if (!dx && !dy) continue;
      const p = { x: lone.x + dx, y: lone.y + dy },
        ghost = template(9, 1, 0, p);
      assert.equal(canPlace(s, ghost, p, true), false);
      assert.equal(canPlace(s, ghost, p), false);
      assert.equal(canPlace(s, { ...ghost, owner: 2 }, p), true);
    }
});

test('feedback 14: sacrificing another cannon grants a same-turn summon even with no enemy', () => {
  const s = fixture(),
    a = add(s, 14, 1, 3, 4),
    b = add(s, 14, 1, 4, 4);
  const c: Command = { type: 'skill', unitId: a.id, targetId: b.id, mode: 'summon' };
  const after = applyCommand(s, c);
  assert.equal(after.summonSlots, 1);
  assert.equal(after.phase, 'play');
  assert.equal(unit(after, a.id).maxHp, 45);
  assert.equal(
    after.units.some((u) => u.id === b.id),
    false,
  );
  assert.equal(after.bonus[1], 0);
  assert.equal(after.heads[1], s.heads[1]);
  const original = structuredClone(after);
  after.heads[1] = 1;
  assert.throws(() => applyCommand(after, { type: 'summon', ultimate: true }));
  assert.equal(after.rng, original.rng);
  assert.equal(after.summonSlots, 1);
  after.heads[1] = 2;
  const drawn = applyCommand(after, { type: 'summon', ultimate: true }, () => 0.01);
  assert.equal(drawn.heads[1], 0);
  assert.equal(drawn.summonSlots, 0);
  assert.equal(drawn.hands[1][0].kind, 'u1');
  assert.equal(drawn.phase, 'play');
  assert.equal(round(original).summonSlots, 0); // 辅助函数也会完成普通召唤阶段。
  const ended = applyCommand(original, { type: 'end' });
  assert.equal(ended.summonSlots, 2);
  assert.equal(ended.bonus[1], 0);
  const session = dispatch(createSession(s), c);
  assert.deepEqual(redo(undo(session)).present, session.present);
  assert.deepEqual(parseSession(JSON.stringify(session)).present, session.present);
});

test('feedback 14: normal sacrifice still needs a shot, while cannon-on-cannon column shooting also awards exactly one summon', () => {
  const s = fixture(),
    a = add(s, 14, 1, 3, 4),
    b = add(s, 14, 1, 4, 4),
    friend = add(s, 9, 1, 2, 4);
  assert.throws(() =>
    applyCommand(s, { type: 'skill', unitId: a.id, targetId: friend.id, mode: 'summon' }),
  );
  assert.throws(() =>
    applyCommand(s, { type: 'skill', unitId: a.id, targetId: friend.id, column: 3 }),
  );
  assert.throws(() =>
    applyCommand(s, { type: 'skill', unitId: a.id, targetId: a.id, mode: 'summon' }),
  );
  const enemy = add(s, 'grave', 2, 3, 6);
  const n = applyCommand(s, { type: 'skill', unitId: a.id, targetId: b.id, column: 3 });
  assert.equal(n.summonSlots, 1);
  assert.equal(unit(n, enemy.id).hp, 65);
  const normal = applyCommand(s, { type: 'skill', unitId: a.id, targetId: friend.id, column: 3 });
  assert.equal(normal.summonSlots, 0);
  assert.equal(unit(normal, enemy.id).hp, 60);
});

test('feedback 15: only the first shot receives charge; moves preserve it, attacks clear it even when blocked', () => {
  const s = fixture(),
    a = add(s, 15, 1, 3, 4),
    b = add(s, 'grave', 2, 3, 6);
  a.charge = a.readyCharge = 4;
  let n = applyCommand(s, { type: 'attack', unitId: a.id, targetId: b.id });
  assert.equal(unit(n, b.id).hp, 35);
  assert.equal(unit(n, a.id).charge, 0);
  assert.equal(getStats(n, unit(n, a.id)).range, 2);
  n = applyCommand(n, { type: 'attack', unitId: a.id, targetId: b.id });
  assert.equal(unit(n, b.id).hp, 20);
  const moved = applyCommand(s, { type: 'move', unitId: a.id, x: 4, y: 4 });
  assert.equal(unit(moved, a.id).charge, 4);
  b.effects.push({ type: 'immune', from: s.ply, until: s.ply + 2, owner: 2 });
  const immune = applyCommand(s, { type: 'attack', unitId: a.id, targetId: b.id });
  assert.equal(unit(immune, b.id).hp, 70);
  assert.equal(unit(immune, a.id).charge, 0);
  b.effects = [];
  b.y = 9;
  const long = applyCommand(s, { type: 'attack', unitId: a.id, targetId: b.id });
  assert.ok(commandError(long, { type: 'attack', unitId: a.id, targetId: b.id }));
  a.effects.push({ type: 'execute', from: s.ply, until: s.ply + 2, owner: 1 });
  assert.equal(
    unit(applyCommand(s, { type: 'attack', unitId: a.id, targetId: b.id }), a.id).charge,
    0,
  );
});

test('feedback 05: attack ten, skill fifteen, charged movement unchanged and conversion can change its side', () => {
  const s = fixture(),
    a = add(s, 9, 1, 3, 4),
    big = add(s, 5, 2, 3, 6);
  assert.equal(definition(5).attack, 10);
  a.effects.push({ type: 'convert', from: s.ply, until: s.ply + 2, owner: 1 });
  for (const silenced of [false, true]) {
    big.silenced = silenced;
    const n = applyCommand(s, { type: 'attack', unitId: a.id, targetId: big.id });
    assert.equal(unit(n, big.id).owner, 1);
    assert.equal(unit(n, big.id).hp, 101);
    assert.equal(
      unit(n, a.id).effects.some((e) => e.type === 'convert'),
      false,
    );
  }
  big.owner = 1;
  big.silenced = false;
  assert.ok(commandError(s, { type: 'move', unitId: big.id, x: 4, y: 6 }));
  const charged = round(applyCommand(s, { type: 'charge', unitId: big.id, mode: 'move' }));
  assert.equal(commandError(charged, { type: 'move', unitId: big.id, x: 4, y: 6 }), null);
});

test('feedback 17: conditional draw has exact 1/5 boundary, derived unit is not an extra pool entry, spell lasts 100 own turns', () => {
  for (const [roll, kind] of [
    [0.199999, 17],
    [0.2, '17p'],
    [0.999, '17p'],
  ] as const) {
    const s = createGame(1);
    let count = 0;
    const n = applyCommand(s, { type: 'summon' }, (cuts) => {
      if (count++ === 0) {
        assert.equal(cuts.length, 27);
        return 16.5 / 26;
      }
      assert.deepEqual(cuts, [0, 0.2, 1]);
      return roll;
    });
    assert.equal(n.hands[1][0].kind, kind);
    assert.equal(count, 2);
    assert.equal(n.hands[1][0].expiresAt, kind === 17 ? n.turns[1] + 100 : undefined);
    assert.deepEqual(parseSession(JSON.stringify(createSession(n))).present, n);
  }
  assert.equal(definition('3p').name, '破碎名刀');
  const d = definition('17p');
  assert.deepEqual(
    [d.name, d.attack, d.health, d.range, d.actions, d.move],
    ['小金耶', 10, 25, 5, 1, 1],
  );
});

test('feedback 17p: each positive damage packet rolls independently, including spells, skills, status and collision', () => {
  for (const kind of ['attack', 'spell', 'skill', 'status', 'collision', 'reflect'] as const) {
    const s = fixture(),
      u = add(s, '17p', 1, 3, 4);
    let rolls = 0;
    const source: Source = { owner: 2, kind };
    withRandomSource(
      s,
      (cuts) => {
        assert.deepEqual(cuts, [0, 0.5, 1]);
        return rolls++ === 0 ? 0.4999 : 0.5;
      },
      () => {
        assert.equal(damage(s, asTarget(u), 7, source, resolution()), 0);
        assert.equal(damage(s, asTarget(u), 7, source, resolution()), 7);
        assert.equal(damage(s, asTarget(u), 0, source, resolution()), 0);
      },
    );
    assert.equal(rolls, 2);
    assert.equal(u.hp, 18);
    u.silenced = true;
    withRandomSource(
      s,
      () => {
        throw new Error('silenced passive must not roll');
      },
      () => damage(s, asTarget(u), 3, source),
    );
    assert.equal(u.hp, 15);
    u.silenced = false;
    u.effects.push({ type: 'immune', from: s.ply, until: s.ply + 2, owner: 1 });
    withRandomSource(
      s,
      () => {
        throw new Error('guaranteed immunity must not roll');
      },
      () => damage(s, asTarget(u), 3, source),
    );
    assert.equal(u.hp, 15);
  }
});

test('feedback 17p: reroll remains in the normal pool; zero-damage attacks do not consume its lottery', () => {
  const s = fixture(),
    mage = add(s, 'u13', 1, 2, 4),
    id = card(s, '17p');
  const n = applyCommand(s, { type: 'reroll', unitId: mage.id, cardId: id }, (cuts) => {
    assert.equal(cuts.length, 27);
    return 0.01;
  });
  assert.equal(n.hands[1][0].kind, 1);
  const a = add(s, 14, 1, 3, 5),
    b = add(s, '17p', 2, 3, 6);
  a.attackBonus = -5; // 定炮基础攻击增强后，该测试仍需要一次真实零伤害攻击。
  applyCommand(s, { type: 'attack', unitId: a.id, targetId: b.id }, () => {
    throw new Error('zero damage must not roll');
  });
});
