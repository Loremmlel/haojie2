import assert from 'node:assert/strict';
import test from 'node:test';
import {
  applyCommand,
  applyPlayerCommand,
  cardActions,
  commandError,
  createSession,
  dispatch,
  getPlayerView,
  getStats,
  hitPullDestination,
  now,
  other,
  parseSession,
  redo,
  undo,
  unitActions,
  type Command,
  type GameState,
} from '../../src/engine';
import { refreshDeployment } from '../../src/engine/core/geometry';
import {
  startIntent,
  advanceIntent,
  canChoose,
  commandFor,
} from '../../src/ui/game/interaction/selection';
import { add, card, fixture, pass, round, unit } from '../helpers';

function assertAtomic(s: GameState, c: Command, reason: RegExp) {
  const before = structuredClone(s);
  assert.match(commandError(s, c) ?? '', reason);
  assert.throws(() => applyCommand(s, c), reason);
  assert.throws(() => applyPlayerCommand(s, s.active, c), reason);
  assert.deepEqual(s, before);
}

// 同时覆盖本地和公开界面选点及权威命令，不只验证辅助函数。
function deployable(s: GameState, row: number, expected: boolean) {
  const id = card(s, 9);
  for (const position of [s, getPlayerView(s, s.active).state]) {
    const a = cardActions(position, position.hands[s.active].find((c) => c.id === id)!).find(
      (a) => a.id === 'deploy',
    )!;
    assert.ok(a);
    assert.equal(canChoose(position, startIntent(a), { x: 8, y: row }), expected);
  }
  const c: Command = { type: 'deploy', cardId: id, x: 8, y: row };
  assert.equal(commandError(s, c) === null, expected);
  if (expected)
    assert.ok(applyPlayerCommand(s, s.active, c).units.some((u) => u.x === 8 && u.y === row));
  else assert.throws(() => applyPlayerCommand(s, s.active, c));
  s.hands[s.active] = s.hands[s.active].filter((c) => c.id !== id);
}

for (const owner of [1, 2] as const) {
  test(`feedback4 giant ring: player ${owner} spares friendly/frozen pieces and its base`, () => {
    const s = fixture();
    s.active = owner;
    const y = owner === 1 ? 2 : 11;
    const big = add(s, 5, owner, 4, y);
    const friend = add(s, 1, owner, 3, y);
    friend.effects.push({ type: 'freeze', owner: other(owner), from: 0, until: 100 });
    const enemy = add(s, 1, other(owner), 6, y);
    const before = structuredClone(s);
    const c: Command = { type: 'skill', unitId: big.id };
    const n = applyPlayerCommand(s, owner, c);
    assert.equal(unit(n, friend.id).hp, friend.hp);
    assert.equal(n.bases[owner], s.bases[owner]);
    assert.equal(unit(n, enemy.id).hp, enemy.hp - 15);
    assert.equal(n.rng, s.rng);
    assert.deepEqual(s, before);
    const history = dispatch(createSession(s), c);
    assert.deepEqual(undo(history).present, s);
    assert.deepEqual(redo(undo(history)).present, n);
    assert.deepEqual(parseSession(JSON.stringify(history)).present, n);
  });

  test(`feedback4 cannon: player ${owner} can select a lane containing only the enemy base`, () => {
    const s = fixture();
    s.active = owner;
    const y = owner === 1 ? 10 : 4;
    const cannon = add(s, 14, owner, 5, y);
    const sacrifice = add(s, 1, owner, 4, y);
    const base = { x: 5, y: owner === 1 ? 13 : 1 };
    let command: Command | null = null;
    for (const position of [s, getPlayerView(s, owner).state]) {
      const a = unitActions(position, position.units.find((u) => u.id === cannon.id)!).find(
        (a) => a.id === 'sacrifice',
      )!;
      assert.ok(a);
      const first = startIntent(a);
      assert.ok(canChoose(position, first, sacrifice));
      const second = advanceIntent(first, sacrifice, position);
      assert.ok(canChoose(position, second, base));
      command = commandFor(position, second, base);
      assert.equal(command?.column, 5);
    }
    assert.ok(command);
    const n = applyPlayerCommand(s, owner, command);
    assert.equal(n.bases[other(owner)], s.bases[other(owner)] - getStats(s, sacrifice).attack);
    assert.equal(unit(n, cannon.id).maxHp, cannon.maxHp - 20);
    assert.ok(!n.units.some((u) => u.id === sacrifice.id));
    const blocker = add(s, 1, other(owner), 5, owner === 1 ? 11 : 3);
    const blocked = applyPlayerCommand(s, owner, command);
    assert.equal(blocked.bases[other(owner)], s.bases[other(owner)]);
    assert.equal(unit(blocked, blocker.id).hp, blocker.hp - getStats(s, sacrifice).attack);
    s.units = s.units.filter((u) => u.id !== blocker.id);
    cannon.y = owner === 1 ? 9 : 5;
    sacrifice.y = cannon.y;
    assertAtomic(s, command, /没有射程内/);
  });

  test(`feedback4 deployment: player ${owner} snapshots count differences at turn start`, () => {
    const row = owner === 1 ? 9 : 5;
    for (const [own, enemy, expected] of [
      [1, 1, false],
      [2, 0, true],
      [2, 1, false],
      [3, 1, true],
    ] as const) {
      let s = fixture();
      s.active = other(owner);
      for (let i = 0; i < own; i++) add(s, 9, owner, 1 + i * 2, row);
      for (let i = 0; i < enemy; i++) add(s, 9, other(owner), 7, row);
      s = pass(s);
      s.summonSlots = 0;
      s = applyCommand(s, { type: 'begin' });
      assert.equal(s.active, owner);
      assert.equal(s.deployRows[owner].includes(row), expected, `${own}:${enemy}`);
      deployable(s, row, expected);
    }
  });
}

test('feedback4 row permissions survive movement, save/load and public projection until next own turn', () => {
  const s = fixture();
  add(s, 9, 1, 1, 9);
  const second = add(s, 9, 1, 3, 9);
  refreshDeployment(s, 1);
  const moved = applyCommand(s, { type: 'move', unitId: second.id, x: 3, y: 8 });
  add(moved, 9, 2, 7, 9); // 当前为一比一，但回合开始时为二比零。
  assert.ok(moved.deployRows[1].includes(9));
  const restored = parseSession(JSON.stringify(createSession(moved))).present;
  deployable(restored, 9, true);
  const next = round(restored);
  deployable(next, 9, false);
  const gained = fixture();
  add(gained, 9, 1, 1, 9);
  const incoming = add(gained, 9, 1, 3, 8);
  refreshDeployment(gained, 1);
  const nowTwo = applyCommand(gained, { type: 'move', unitId: incoming.id, x: 3, y: 9 });
  deployable(nowTwo, 9, false);
  deployable(round(nowTwo), 9, true);
});

test('feedback4 deployment counts each giant once per touched row and stacked pieces individually', () => {
  const s = fixture();
  const big = add(s, 5, 1, 1, 9);
  refreshDeployment(s, 1);
  assert.ok(!s.deployRows[1].includes(9) && !s.deployRows[1].includes(10));
  const ally = add(s, 9, 1, 4, 9);
  refreshDeployment(s, 1);
  assert.ok(s.deployRows[1].includes(9) && !s.deployRows[1].includes(10));
  const counter = add(s, 9, 2, 7, 9);
  refreshDeployment(s, 1);
  assert.ok(!s.deployRows[1].includes(9));
  add(s, 9, 1, ally.x, ally.y); // 统计独立棋子数量，不统计去重后的占格数。
  refreshDeployment(s, 1);
  assert.ok(s.deployRows[1].includes(9));
  assert.equal(big.size, 2);
  assert.equal(counter.owner, 2);
});

test('feedback4 giant cannot receive spell22 or execute conversion, including silence, inherited traits and legacy effects', () => {
  for (const inherited of [false, true])
    for (const silenced of [false, true]) {
      const s = fixture();
      const big = add(s, inherited ? 9 : 5, 1, 3, 4);
      if (inherited) big.traits = [5];
      big.silenced = silenced;
      add(s, 'archmage', 2, 8, 11); // 非法施法不能消耗反制随机数。
      const id = card(s, 22);
      assertAtomic(s, { type: 'cast', cardId: id, targetId: big.id }, /不能使用策反/);
      for (const position of [s, getPlayerView(s, 1).state]) {
        const a = cardActions(position, position.hands[1].find((c) => c.id === id)!)[0];
        assert.equal(canChoose(position, startIntent(a), big), false);
      }
      const victim = add(s, 1, 2, 3, 6);
      big.effects.push({ type: 'convert', owner: 1, from: 0, until: 100 });
      const n = applyCommand(s, { type: 'attack', unitId: big.id, targetId: victim.id });
      assert.equal(unit(n, victim.id).owner, 2);
      assert.ok(unit(n, victim.id).hp < victim.hp);
    }
});

test('feedback4 all hooks reject giants atomically, including inherited/silenced restrictions and saved pulls', () => {
  for (const kind of [7, 'u23'] as const)
    for (const silenced of [false, true]) {
      const s = fixture();
      const hook = add(s, kind, 1, 3, 4);
      hook.hookReadyAt = now(s, hook);
      hook.hookExpiresAt = now(s, hook) + 2;
      const big = add(s, 5, 2, 3, 6);
      big.silenced = silenced;
      const c: Command = {
        type: 'skill',
        unitId: hook.id,
        targetId: big.id,
        ...(kind === 7 ? { x: 4, y: 4 } : {}),
      };
      assertAtomic(s, c, /不能被钩子/);
      for (const position of [s, getPlayerView(s, 1).state]) {
        const a = unitActions(position, position.units.find((u) => u.id === hook.id)!).find(
          (a) => a.id === (kind === 7 ? 'hook' : 'superhook'),
        )!;
        assert.ok(a);
        assert.equal(canChoose(position, startIntent(a), big), false);
      }
      big.kind = 9;
      big.size = 1;
      big.traits = [5];
      assertAtomic(s, c, /不能被钩子/);
    }
  const s = fixture();
  const hook = add(s, 'formless', 1, 3, 4);
  const big = add(s, 5, 2, 3, 8);
  hook.charge = hook.readyCharge = 1;
  hook.chargeType = 'attack';
  const n = applyCommand(s, { type: 'attack', unitId: hook.id, targetId: big.id });
  assert.equal(unit(n, big.id).hp, big.hp - getStats(s, hook).attack);
  assert.equal(n.pending.length, 0); // 攻击仍造成伤害，但不能出现无法完成的牵引提示。
  s.pending.push({
    kind: 'hit-pull',
    owner: 1,
    source: structuredClone(hook),
    targetId: big.id,
    amount: 0,
  });
  const loaded = parseSession(JSON.stringify(createSession(s))).present;
  assert.equal(hitPullDestination(loaded, loaded.pending[0]), null);
  assert.throws(() => applyCommand(loaded, { type: 'react', mode: 'pull' }));
  assert.equal(applyCommand(loaded, { type: 'react' }).pending.length, 0);
});

test('feedback4 CX can convert an enemy giant but a carrier inheriting giant cannot convert', () => {
  for (const inherited of [false, true]) {
    const s = fixture();
    const cx = add(s, 's3', 1, 3, 4);
    const big = add(s, 5, 2, 3, 6);
    if (inherited) cx.traits = [5];
    let rolls = 0;
    const n = applyCommand(s, { type: 'attack', unitId: cx.id, targetId: big.id }, () => {
      rolls++;
      return 0;
    });
    assert.equal(unit(n, big.id).owner, inherited ? 2 : 1);
    assert.ok(unit(n, big.id).hp < big.hp);
    assert.equal(rolls, 2); // 该限制不改变已有 CX 随机消耗顺序。
  }
});
