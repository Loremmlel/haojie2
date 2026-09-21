import test from 'node:test';
import assert from 'node:assert/strict';
import { applyCommand, createSession, dispatch, parseSession, redo, undo } from '../../src/engine';
import { add, fixture, round, strike, unit } from '../helpers';

// 通过公开命令验证；局面已保存独立效果记录。
test('catapult marks stack from one or several catapults and all detonate on an allied follow-up', () => {
  for (const separate of [false, true]) {
    let s = fixture();
    const a = add(s, 10, 1, 3, 4),
      b = separate ? add(s, 10, 1, 4, 4) : a,
      ally = add(s, 23, 1, 5, 4),
      target = add(s, 'grave', 2, 4, 6);
    target.hp = 50;
    s = strike(strike(s, a, target), b, target);
    assert.equal(unit(s, target.id).hp, 50);
    assert.equal(unit(s, target.id).effects.filter((e) => e.type === 'mark').length, 2);
    const original = structuredClone(s);
    const played = dispatch(createSession(s), {
      type: 'attack',
      unitId: ally.id,
      targetId: target.id,
    });
    assert.equal(unit(played.present, target.id).hp, 35); // 五点攻击加两层各五点标记。
    assert.equal(
      unit(played.present, target.id).effects.filter((e) => e.type === 'mark').length,
      0,
    );
    assert.equal(
      played.present.events.filter((e) => e.type === 'damage' && e.action === 'mark').length,
      2,
    );
    assert.deepEqual(s, original);
    assert.deepEqual(redo(undo(played)).present, played.present);
    assert.deepEqual(parseSession(JSON.stringify(createSession(s))).present, s);
  }
});

test('base marks stack and expire at the next caster turn; lethal detonation awards only one head', () => {
  let s = fixture();
  const catapult = add(s, 10, 1, 4, 10),
    ally = add(s, 23, 1, 6, 10);
  s.bases[2] = 290;
  const mark = { type: 'attack' as const, unitId: catapult.id, targetId: 'base-2' };
  s = applyCommand(applyCommand(s, mark), mark);
  assert.equal(s.baseEffects[2].length, 2);
  assert.equal(round(s).baseEffects[2].length, 0);
  s = applyCommand(s, { type: 'attack', unitId: ally.id, targetId: 'base-2' });
  assert.equal(s.bases[2], 275);
  assert.equal(s.baseEffects[2].length, 0);
  s = fixture();
  const a = add(s, 10, 1, 3, 4),
    friend = add(s, 23, 1, 5, 4),
    target = add(s, 1, 2, 4, 6);
  target.hp = 9;
  s = strike(strike(s, a, target), a, target);
  const heads = s.heads[1];
  s = strike(s, friend, target);
  assert.equal(
    s.units.some((u) => u.id === target.id),
    false,
  );
  assert.equal(s.heads[1], heads + 1);
  assert.equal(s.deaths.filter((d) => d.kind === 1).length, 1);
});

test('stacked marks remain separate damage packets and full healing detonates every live mark', () => {
  let s = fixture();
  const a = add(s, 10, 1, 3, 4),
    zero = add(s, 'u19', 1, 5, 4),
    target = add(s, '17p', 2, 4, 6);
  target.hp = 20;
  s = strike(strike(s, a, target), a, target);
  let rolls = 0;
  s = applyCommand(s, { type: 'attack', unitId: zero.id, targetId: target.id }, () =>
    ++rolls === 1 ? 0.25 : 0.75,
  );
  assert.equal(rolls, 2);
  assert.equal(unit(s, target.id).hp, 15);
  assert.equal(unit(s, target.id).effects.filter((e) => e.type === 'mark').length, 0);
  s = fixture();
  const catapult = add(s, 10, 1, 3, 4),
    victim = add(s, 1, 2, 4, 6),
    healer = add(s, 2, 2, 5, 6);
  victim.hp = 40;
  s = strike(strike(s, catapult, victim), catapult, victim);
  s.active = 2;
  s = strike(s, healer, victim);
  assert.equal(unit(s, victim.id).hp, 40); // 先治疗至50，再分别结算两层五点标记。
  assert.equal(unit(s, victim.id).effects.filter((e) => e.type === 'mark').length, 0);
});

test('conversion requires positive direct attack damage, never catapult detonation or other mark damage', () => {
  for (const full of [true, false]) {
    let s = fixture();
    const a = add(s, 10, 1, 3, 4),
      zero = add(s, 'u19', 1, 5, 4),
      target = add(s, 'grave', 2, 4, 6);
    if (!full) target.hp = 50;
    for (const u of [a, zero])
      u.effects.push({ type: 'convert', owner: 1, from: s.ply, until: s.ply + 2 });
    s = strike(s, a, target);
    assert.equal(unit(s, target.id).owner, 2);
    assert.ok(unit(s, a.id).effects.some((e) => e.type === 'convert'));
    if (full) s = strike(s, a, target); // 当前已经受伤，为零攻击友方留下标记。
    const before = unit(s, target.id).hp;
    s = strike(s, zero, target);
    assert.equal(unit(s, target.id).hp, before - 5);
    assert.equal(unit(s, target.id).owner, 2);
    assert.ok(unit(s, zero.id).effects.some((e) => e.type === 'convert'));
  }
});

test('blocked direct damage preserves conversion; a later damaging hit converts or consumes it on lethal damage', () => {
  for (const lethal of [false, true]) {
    let s = fixture();
    const a = add(s, 9, 1, 3, 4),
      king = add(s, 'u18', 2, 4, 6),
      target = add(s, 1, 2, 5, 4);
    a.effects.push({ type: 'convert', owner: 1, from: s.ply, until: s.ply + 2 });
    if (lethal) target.hp = 5;
    s = strike(s, a, king);
    assert.equal(unit(s, king.id).hp, 50);
    assert.equal(unit(s, king.id).owner, 2);
    assert.ok(unit(s, a.id).effects.some((e) => e.type === 'convert'));
    s = strike(s, a, target);
    if (lethal)
      assert.equal(
        s.units.some((u) => u.id === target.id),
        false,
      );
    else assert.equal(unit(s, target.id).owner, 1);
    assert.equal(
      unit(s, a.id).effects.some((e) => e.type === 'convert'),
      false,
    );
  }
});
