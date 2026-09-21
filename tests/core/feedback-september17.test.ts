import test from 'node:test';
import assert from 'node:assert/strict';
import {
  applyCommand,
  asTarget,
  availableGuardians,
  canRerollWith,
  commandError,
  createSession,
  definition,
  dispatch,
  getStats,
  has,
  parseSession,
  redo,
  rerollCommands,
  summonRerolls,
  undo,
  validState,
} from '../../src/engine';
import { damage, kill, resolution } from '../../src/engine/commands/combat';
import { unitStatus } from '../../src/ui/inspector/unit-status';
import { add, card, fixture, pass, round, strike, unit } from '../helpers';

for (const spell of [18, 22] as const) {
  test(`${spell} waits for an actual own turn in both horn/cast orders, then consumes once`, () => {
    for (const hornFirst of [false, true]) {
      let s = fixture();
      const carrier = add(s, 9, 1, 3, 4),
        victim = add(s, 'grave', 2, 3, 6);
      const enchantment = card(s, spell),
        horn = card(s, 'u17');
      for (const id of hornFirst ? [horn, enchantment] : [enchantment, horn])
        s = applyCommand(s, { type: 'cast', cardId: id, targetId: carrier.id });
      const type = spell === 18 ? 'execute' : 'convert';
      assert.equal(has(s, unit(s, carrier.id), type), false);
      assert.equal(
        unitStatus(s, unit(s, carrier.id)).find((e) => e.label === definition(spell).name)?.pending,
        true,
      );
      s = strike(s, unit(s, carrier.id), unit(s, victim.id));
      assert.equal(unit(s, victim.id).hp, 60);
      assert.equal(unit(s, victim.id).owner, 2);
      assert.ok(unit(s, carrier.id).effects.some((e) => e.type === type));
      s = round(s);
      assert.equal(has(s, unit(s, carrier.id), type), true);
      s = strike(s, unit(s, carrier.id), unit(s, victim.id));
      if (spell === 18) assert.ok(!s.units.some((u) => u.id === victim.id));
      else {
        assert.equal(unit(s, victim.id).owner, 1);
        assert.equal(unit(s, victim.id).hp, 50);
      }
      assert.ok(!unit(s, carrier.id).effects.some((e) => e.type === type));
    }
  });
}

test('execution bypasses every knife source; gold still blocks and consumes the first hit', () => {
  for (const gold of [false, true]) {
    let s = fixture();
    const a = add(s, 9, 1, 4, 4),
      v = add(s, 'grave', 2, 4, 6);
    add(s, 3, 2, 3, 6);
    add(s, 3, 2, 5, 6);
    const id = card(s, 18);
    s = applyCommand(s, { type: 'cast', cardId: id, targetId: a.id });
    s = round(s);
    if (gold)
      unit(s, v.id).effects.push({ type: 'immune', owner: 2, from: s.ply, until: s.ply + 2 });
    assert.equal(availableGuardians(s, unit(s, v.id)).length, 2);
    s = strike(s, unit(s, a.id), unit(s, v.id));
    assert.equal(
      s.units.some((u) => u.id === v.id),
      gold,
    );
    if (gold) {
      assert.equal(unit(s, v.id).hp, 70);
      assert.equal(availableGuardians(s, unit(s, v.id)).length, 2);
    }
    assert.ok(!unit(s, a.id).effects.some((e) => e.type === 'execute'));
  }
});

test('knife protections are consumed oldest-source-first and disappear with the correct source', () => {
  for (const lost of ['older', 'newer', 'neither'] as const) {
    const s = fixture(),
      recipient = add(s, 9, 2, 4, 6);
    const older = add(s, 3, 2, 3, 6),
      newer = add(s, 3, 2, 5, 6);
    older.deployedAt = 1;
    newer.deployedAt = 3;
    recipient.hp = 20;
    const source = { kind: 'attack' as const, owner: 1 as const };
    damage(s, asTarget(recipient), 20, source, resolution());
    assert.equal(recipient.hp, 1);
    assert.deepEqual(recipient.guardSourceIds, [older.id]);
    assert.equal(s.events.find((e) => e.text === '名刀')?.actor?.id, older.id);
    if (lost !== 'neither') kill(s, lost === 'older' ? older : newer, source, resolution());
    assert.deepEqual(
      availableGuardians(s, recipient).map((u) => u.id),
      lost === 'newer' ? [] : [newer.id],
    );
    damage(s, asTarget(recipient), 20, source, resolution());
    assert.equal(s.units.includes(recipient), lost !== 'newer');
    if (lost !== 'newer') {
      assert.deepEqual(recipient.guardSourceIds, [older.id, newer.id]);
      damage(s, asTarget(recipient), 20, source, resolution());
      assert.ok(!s.units.includes(recipient));
    }
  }
});

test('knife source loss, silence and range are live; leaving/reentering never replenishes a source', () => {
  const s = fixture(),
    recipient = add(s, 9, 2, 4, 6);
  const source = add(s, 3, 2, 3, 6),
    other = add(s, 3, 2, 5, 6);
  recipient.hp = 10;
  damage(s, asTarget(recipient), 20, { kind: 'attack', owner: 1 }, resolution());
  assert.deepEqual(recipient.guardSourceIds, [source.id]);
  other.silenced = true;
  assert.equal(availableGuardians(s, recipient).length, 0);
  other.silenced = false;
  other.y = 13;
  assert.equal(availableGuardians(s, recipient).length, 0);
  other.y = 6;
  assert.deepEqual(
    availableGuardians(s, recipient).map((u) => u.id),
    [other.id],
  );
  source.y = 13;
  source.y = 6;
  assert.deepEqual(
    availableGuardians(s, recipient).map((u) => u.id),
    [other.id],
  );
  assert.match(unitStatus(s, recipient).find((r) => r.key === 'guards')!.detail, /已消耗.*可用/);
});

test('knife source ledger survives save/undo; legacy aggregate saves remain loadable without free refills', () => {
  const s = fixture(),
    attacker = add(s, 26, 1, 4, 4),
    victim = add(s, 9, 2, 4, 6);
  const older = add(s, 3, 2, 3, 6),
    newer = add(s, 3, 2, 5, 6);
  victim.hp = 10;
  const history = dispatch(createSession(s), {
    type: 'attack',
    unitId: attacker.id,
    targetId: victim.id,
  });
  const loaded = parseSession(JSON.stringify(history));
  assert.deepEqual(loaded.present, history.present);
  assert.deepEqual(redo(undo(loaded)).present, loaded.present);
  assert.deepEqual(
    availableGuardians(loaded.present, unit(loaded.present, victim.id)).map((u) => u.id),
    [newer.id],
  );
  const legacy = structuredClone(history);
  delete unit(legacy.present, victim.id).guardSourceIds;
  const migrated = parseSession(JSON.stringify(legacy));
  assert.deepEqual(unit(migrated.present, victim.id).guardSourceIds, [older.id, newer.id]);
  const fresh = add(migrated.present, 3, 2, 4, 7);
  assert.deepEqual(
    availableGuardians(migrated.present, unit(migrated.present, victim.id)).map((u) => u.id),
    [fresh.id],
  );
  unit(migrated.present, victim.id).guardSourceIds = [older.id, older.id];
  assert.equal(validState(migrated.present), false);
});

test('14 and 15 author stats, twenty-cap sacrifice including exact-cost death, and invalid cost atomicity', () => {
  assert.deepEqual([definition(14).attack, definition(14).health], [5, 65]);
  assert.deepEqual([definition(15).attack, definition(15).health], [15, 35]);
  for (const hp of [65, 20, 19]) {
    const s = fixture(),
      cannon = add(s, 14, 1, 4, 4),
      sacrifice = add(s, 15, 1, 3, 4);
    const enemy = add(s, 'grave', 2, 4, 6);
    cannon.maxHp = cannon.hp = hp;
    sacrifice.charge = sacrifice.readyCharge = 4;
    assert.equal(getStats(s, sacrifice).attack, 35);
    const command = {
      type: 'skill' as const,
      unitId: cannon.id,
      targetId: sacrifice.id,
      column: 4,
    };
    if (hp === 19) {
      const snapshot = structuredClone(s);
      assert.ok(commandError(s, command));
      assert.deepEqual(s, snapshot);
    } else {
      const n = applyCommand(s, command);
      assert.equal(n.units.find((u) => u.id === cannon.id)?.maxHp, hp === 20 ? undefined : 45);
      assert.equal(unit(n, enemy.id).hp, 35);
      assert.equal(n.heads[2], s.heads[2]);
      assert.ok(!n.units.some((u) => u.id === sacrifice.id));
    }
  }
});

test('U13 waits for all summons, independently spends each live source, and horn never refreshes it', () => {
  let s = fixture();
  s.phase = 'summon';
  s.summonSlots = 1;
  const a = add(s, 'u13', 1, 2, 4),
    b = add(s, 'u13', 1, 3, 4);
  const old = card(s, 1);
  const snap = structuredClone(s);
  assert.ok(commandError(s, { type: 'reroll', cardId: old, unitId: a.id }));
  assert.equal(summonRerolls(s).length, 0);
  assert.deepEqual(s, snap);
  s = applyCommand(s, { type: 'summon' }, () => 0.05);
  assert.equal(summonRerolls(s).length, 2);
  s = applyCommand(s, { type: 'reroll', cardId: old, unitId: a.id }, () => 0.05);
  assert.equal(canRerollWith(s, unit(s, a.id)), false);
  assert.equal(canRerollWith(s, unit(s, b.id)), true);
  s = applyCommand(s, { type: 'begin' });
  const horn = card(s, 'u17');
  s = applyCommand(s, { type: 'cast', cardId: horn, targetId: a.id });
  assert.equal(canRerollWith(s, unit(s, a.id)), false);
  const target = s.hands[1][0];
  assert.ok(commandError(s, { type: 'reroll', cardId: target.id, unitId: a.id }));
  const heads = s.heads[1],
    slots = s.summonSlots;
  s = applyCommand(s, { type: 'reroll', cardId: target.id, unitId: b.id }, () => 0.05);
  assert.equal(s.heads[1], heads);
  assert.equal(s.summonSlots, slots);
  assert.equal(summonRerolls(s).length, 0);
  s = round(s);
  card(s, 1);
  assert.equal(canRerollWith(s, unit(s, a.id)), true);
  assert.equal(canRerollWith(s, unit(s, b.id)), true);
});

test('U13 self-reroll works through turn five, cannot self-chain, and ultimate clones reroll as one paid result', () => {
  for (const turn of [1, 5, 6]) {
    const s = fixture();
    s.turns[1] = turn;
    const id = card(s, 'u13');
    if (turn === 6) {
      assert.ok(commandError(s, { type: 'reroll', cardId: id }));
      continue;
    }
    const n = applyCommand(s, { type: 'reroll', cardId: id }, () => 12.5 / 28);
    assert.equal(n.hands[1][0].kind, 'u13');
    assert.equal(n.hands[1][0].summonPool, 'ultimate');
    assert.equal(rerollCommands(n, n.hands[1][0]).length, 0);
  }
  let s = fixture();
  s.phase = 'summon';
  s.summonSlots = 1;
  const mage = add(s, 'u13', 1, 3, 4);
  s = applyCommand(s, { type: 'summon', ultimate: true }, () => 24.5 / 28);
  assert.equal(s.hands[1].length, 8);
  const choices = summonRerolls(s);
  assert.equal(choices.length, 1);
  assert.equal(choices[0].pool, 'ultimate');
  const partial = structuredClone(s);
  partial.hands[1].pop();
  assert.ok(
    commandError(partial, { type: 'reroll', unitId: mage.id, cardId: partial.hands[1][0].id }),
  );
  const before = createSession(s);
  const after = dispatch(before, choices[0].commands[0]);
  assert.equal(after.present.heads[1], 4);
  assert.ok(after.present.hands[1].every((c) => c.summonPool === 'ultimate'));
  assert.deepEqual(redo(undo(after)), after);
  assert.deepEqual(parseSession(JSON.stringify(after)), after);
});

test('status lists pending/active effects, sources and equipment while ignoring expired entries', () => {
  const s = fixture(),
    mage = add(s, 'u6', 1, 4, 4);
  mage.equipment = ['u5'];
  mage.offset = 2;
  mage.effects.push(
    { type: 'execute', owner: 1, from: 7, until: 9, global: true },
    { type: 'attack', owner: 1, from: 5, until: 9, amount: 10 },
    { type: 'attack', owner: 2, from: 5, until: 9, amount: -15 },
    { type: 'stun', owner: 1, from: 3, until: 4 },
  );
  add(s, 3, 1, 3, 4);
  add(s, 'u15', 1, 5, 4);
  const original = structuredClone(s),
    rows = unitStatus(s, mage);
  assert.equal(rows.find((r) => r.label === '死吧！')?.pending, true);
  assert.ok(rows.some((r) => r.label.includes('剩余 1 次')));
  assert.ok(rows.some((r) => r.label === '免疫塔保护'));
  assert.ok(rows.some((r) => r.label === '装备 · 寒冰法杖'));
  assert.match(rows.find((r) => r.label === '攻击强化')!.detail, /攻击 \+10/);
  assert.match(rows.find((r) => r.label === '攻击削弱')!.detail, /攻击 -15/);
  assert.ok(!rows.some((r) => r.label === '眩晕'));
  assert.deepEqual(s, original);
});

test('unused delayed execution and conversion expire after the next actual own turn, not on enemy reactions', () => {
  for (const spell of [18, 22] as const) {
    let s = fixture();
    const u = add(s, 'u18', 1, 4, 4),
      id = card(s, spell);
    s = applyCommand(s, { type: 'cast', cardId: id, targetId: u.id });
    s = round(s);
    const type = spell === 18 ? 'execute' : 'convert';
    assert.ok(has(s, unit(s, u.id), type));
    s = pass(s);
    assert.equal(s.active, 2);
    assert.equal(has(s, unit(s, u.id), type), false);
  }
});
