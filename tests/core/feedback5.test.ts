import test from 'node:test';
import assert from 'node:assert/strict';
import {
  allegiance,
  applyCommand,
  applyPlayerCommand,
  canPlace,
  createSession,
  deploymentRows,
  dispatch,
  getPlayerView,
  getStats,
  parseSession,
  redo,
  undo,
} from '../../src/engine';
import { distribution } from '../../src/ai/simulation/simulate';
import { add, card, fixture, unit } from '../helpers';

test('超级跑得快亡语两分支各半，反伤20且法术返还真实基地，输入不可变', () => {
  for (const spell of [false, true]) {
    const s = fixture();
    const attacker = add(s, 26, 1, 3, 4);
    const runner = add(s, 20, 2, 3, 5);
    const c = spell
      ? { type: 'cast' as const, cardId: card(s, 8), x: 3, y: 5 }
      : { type: 'attack' as const, unitId: attacker.id, targetId: runner.id };
    const before = structuredClone(s);
    const result = distribution(s, c);
    assert.equal(result.sampled, false);
    assert.deepEqual(
      result.outcomes.map((o) => o.weight),
      [0.5, 0.5],
    );
    const denied = result.outcomes.find((o) => o.state.heads[1] === 6)!.state;
    const reflected = result.outcomes.find((o) => o.state.heads[1] === 7)!.state;
    assert.equal(spell ? denied.bases[1] : unit(denied, attacker.id).hp, spell ? 300 : 55);
    assert.equal(spell ? reflected.bases[1] : unit(reflected, attacker.id).hp, spell ? 280 : 35);
    assert.deepEqual(s, before);
  }
});

test('小屋不被普通超级跑得快循环触发，其他友方与王城保留原召唤', () => {
  for (const kind of [20, 'u20', 1, 'grave'] as const) {
    const s = fixture();
    const hut = add(s, 'u22', 2, 3, 6);
    const attacker = add(s, 26, 1, 3, 4);
    const victim = add(s, kind, 2, 3, 5);
    victim.hp = 1;
    const next = applyCommand(
      s,
      { type: 'attack', unitId: attacker.id, targetId: victim.id },
      () => 0,
    );
    assert.equal(next.pending.length, kind === 20 || kind === 'grave' ? 0 : 1);
    if (!next.pending.length) {
      assert.equal(unit(next, hut.id).maxHp, 50);
      continue;
    }
    const spawned = applyCommand(next, { type: 'react', x: 3, y: 5 });
    const child = spawned.units.find((u) => u.kind === 20)!;
    assert.ok(child && child.id !== victim.id);
    assert.deepEqual([getStats(spawned, child).attack, child.hp, child.maxHp], [20, 20, 20]);
    assert.equal(unit(spawned, hut.id).maxHp, 40);
    unit(spawned, child.id).hp = 1;
    const followup = add(spawned, 26, 1, 4, 4);
    const again = applyCommand(
      spawned,
      { type: 'attack', unitId: followup.id, targetId: child.id },
      () => 0,
    );
    assert.equal(again.pending.length, 0);
  }
  const s = fixture();
  add(s, 'citadel', 2, 3, 6);
  const attacker = add(s, 26, 1, 3, 4);
  const runner = add(s, 20, 2, 3, 5);
  assert.equal(
    applyCommand(s, { type: 'attack', unitId: attacker.id, targetId: runner.id }, () => 0).pending
      .length,
    1,
  );
});

test('跑得快墓地中立，重铸及公开预检原子拒绝，不计行权或友方死亡光环', () => {
  let s = fixture();
  const attacker = add(s, 26, 1, 3, 4);
  const runner = add(s, 12, 2, 3, 5);
  runner.hp = 1;
  s = applyCommand(s, { type: 'attack', unitId: attacker.id, targetId: runner.id });
  const grave = s.units.find((u) => u.kind === 'grave')!;
  assert.equal(allegiance(s, grave), 0);
  assert.equal(grave.hp, 70);
  s.active = 2;
  const ally = add(s, 1, 2, 7, 5);
  const id = card(s, 25);
  const invalid = {
    type: 'cast' as const,
    cardId: id,
    mode: 'double',
    sacrificeIds: [grave.id, ally.id],
  };
  const before = structuredClone(s);
  assert.throws(() => applyCommand(s, invalid), /友方/);
  assert.throws(() => applyPlayerCommand(s, 2, invalid), /友方/);
  assert.deepEqual(s, before);
  assert.ok(!deploymentRows(s, 2).includes(5));
  const restored = parseSession(JSON.stringify(createSession(s))).present;
  assert.equal(allegiance(restored, unit(restored, grave.id)), 0);
  const followup = add(s, 26, 1, 4, 4);
  followup.effects.push({ type: 'convert', owner: 1, from: s.ply, until: s.ply + 2 });
  const destroyed = applyCommand(
    { ...s, active: 1 },
    { type: 'attack', unitId: followup.id, targetId: grave.id },
  );
  assert.equal(unit(destroyed, grave.id).hp, 50);
  assert.equal(unit(destroyed, grave.id).owner, 2);
  assert.equal(allegiance(destroyed, unit(destroyed, grave.id)), 0);
  assert.ok(unit(destroyed, followup.id).effects.some((e) => e.type === 'convert'));
  const lane = fixture();
  const cannon = add(lane, 14, 1, 3, 4);
  const sacrifice = add(lane, 1, 1, 4, 4);
  const ownGrave = add(lane, 'grave', 1, 3, 6);
  const hit = applyCommand(lane, {
    type: 'skill',
    unitId: cannon.id,
    targetId: sacrifice.id,
    column: 3,
  });
  assert.equal(unit(hit, ownGrave.id).hp, 50);
});

test('双方可在抽卡后移动开放部署行，落子、悔棋、载入与公开视图一致', () => {
  for (const owner of [1, 2] as const) {
    const s = fixture();
    s.active = owner;
    const row = owner === 1 ? 9 : 5;
    const start = owner === 1 ? 8 : 6;
    add(s, 9, owner, 1, row);
    const mover = add(s, 9, owner, 3, start);
    const id = card(s, 9);
    const deploy = { type: 'deploy' as const, cardId: id, x: 8, y: row };
    assert.throws(() => applyCommand(s, deploy), /非法部署/);
    let session = dispatch(createSession(s), { type: 'move', unitId: mover.id, x: 3, y: row });
    assert.ok(session.present.deployRows[owner].includes(row));
    assert.ok(getPlayerView(session.present, owner).state.deployRows[owner].includes(row));
    assert.ok(
      canPlace(
        getPlayerView(session.present, owner).state,
        unit(session.present, mover.id),
        { x: 8, y: row },
        true,
      ),
    );
    const loaded = parseSession(JSON.stringify(session));
    session = dispatch(loaded, deploy);
    assert.ok(session.present.units.some((u) => u.x === 8 && u.y === row));
    assert.deepEqual(redo(undo(session)).present, session.present);
    assert.ok(!deploymentRows(undo(undo(session)).present, owner).includes(row));
    assert.ok(
      applyPlayerCommand(loaded.present, owner, deploy).units.some((u) => u.x === 8 && u.y === row),
    );
  }
});

test('死亡即时撤回部署行，旧档行权限不能授权非法部署且失败不花卡或随机数', () => {
  const s = fixture();
  const first = add(s, 9, 1, 1, 9);
  add(s, 9, 1, 3, 9);
  const foe = add(s, 26, 2, 1, 8);
  first.hp = 1;
  s.active = 2;
  const next = applyCommand(s, { type: 'attack', unitId: foe.id, targetId: first.id });
  assert.ok(!next.deployRows[1].includes(9));
  next.active = 1;
  next.deployRows[1].push(9);
  const id = card(next, 9);
  const before = structuredClone(next);
  assert.throws(() => applyCommand(next, { type: 'deploy', cardId: id, x: 8, y: 9 }), /非法部署/);
  assert.deepEqual(next, before);
  assert.ok(!parseSession(JSON.stringify(createSession(next))).present.deployRows[1].includes(9));
});
