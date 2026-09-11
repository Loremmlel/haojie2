import { eventActor, withEventFacts } from './event-facts';
import { alive, damage, findTarget, performAttack, lowerMax } from './combat';
import type { Resolution } from './combat';
import {
  ALL_CELLS,
  attackPath,
  basePoint,
  canPlace,
  cells,
  distance,
  equal,
  inside,
  key,
  movementPath,
  neighbors,
  occupants,
  targets,
} from './geometry';
import {
  actor,
  addUnit,
  allegiance,
  asTarget,
  chooseMode,
  emit,
  ensure,
  findUnit,
  finishOperation,
  getStats,
  hasWeapon,
  isRunner,
  point,
  template,
} from './state';
import type { Command, GameState, Point, Unit } from './types';
export function emptyFor(s: GameState, u: Unit, p: Point = u) {
  return canPlace(s, u, p);
}
function canEnter(s: GameState, u: Unit, p: Point) {
  if (!inside(p) || equal(p, basePoint(u.owner))) return false;
  if (u.kind === 'u12p' && equal(p, basePoint(u.owner === 1 ? 2 : 1))) return false;
  return !occupants(s, p).some((v) => v.id !== u.id && allegiance(s, v) === u.owner);
}
function reachableExit(s: GameState, u: Unit, steps: number) {
  const queue = [{ p: { x: u.x, y: u.y }, steps: 0 }],
    seen = new Set([key(u)]);
  for (let i = 0; i < queue.length; i++) {
    const item = queue[i];
    if (emptyFor(s, u, item.p)) return true;
    if (item.steps >= steps) continue;
    for (const p of neighbors(item.p)) {
      if (seen.has(key(p)) || !canEnter(s, u, p)) continue;
      seen.add(key(p));
      queue.push({ p, steps: item.steps + 1 });
    }
  }
  return false;
}
export function moveUnit(s: GameState, c: Command, ctx: Resolution) {
  const u = findUnit(s, c.unitId);
  return withEventFacts(
    s,
    { actor: eventActor(u), ...(isRunner(u) ? { action: 'rush' as const } : {}) },
    () => resolveMove(s, c, ctx),
  );
}
function resolveMove(s: GameState, c: Command, ctx: Resolution) {
  const u = actor(s, c.unitId),
    to = point(c.x, c.y),
    starting = u.mode === 'none';
  chooseMode(s, u, 'move');
  if (isRunner(u)) {
    if (starting) {
      ensure(u.readyCharge >= 1 && u.chargeType === 'move', '需要在回合开始已有1层移动蓄力。');
      u.moves = 5;
      u.charge = u.readyCharge = 0;
    }
    ensure(
      u.moves > 0 && distance(u, to) === 1 && canEnter(s, u, to),
      '每次移动一格，不能进入友方或非法格。',
    );
    const t = targets(s).find(
      (t) => t.id !== u.id && (t.unit ? cells(t.unit) : [t]).some((p) => equal(p, to)),
    );
    emit(s, { type: 'move', from: u, to, unitId: u.id, owner: u.owner, text: '冲撞' });
    Object.assign(u, to);
    u.moves--;
    if (t) damage(s, t, 30, { owner: u.owner, unit: u, kind: 'collision' }, ctx);
    if (!alive(s, u)) return;
    if (u.kind === 'u12' && t) {
      s.pending.unshift({ kind: 'bounce', owner: u.owner, source: structuredClone(u), amount: 30 });
    } else if (u.kind === 'u12p')
      ensure(
        emptyFor(s, u) || (u.moves > 0 && reachableExit(s, u, u.moves)),
        '剩余移动次数无法返回空地，不能进行这次冲撞。',
      );
    if (u.moves === 0 && emptyFor(s, u)) {
      finishOperation(u);
      if (hasWeapon(u, 'u16')) u.bonusAttacks++;
    }
    return;
  }
  let limit = getStats(s, u).move;
  if (limit % 1 !== 0) {
    ensure(
      u.readyCharge >= 1 && u.chargeType === 'move',
      '分数移动需要在回合开始已有1层移动蓄力。',
    );
    limit *= 2;
  }
  const path = movementPath(s, u, to, limit, u.kind === 13 && !u.silenced);
  ensure(path, '移动距离、路径、占位或独行侠禁区不合法。');
  emit(s, { type: 'move', from: u, to, path, unitId: u.id, owner: u.owner });
  Object.assign(u, to);
  if (getStats(s, u).move % 1 !== 0) u.charge = u.readyCharge = 0;
  finishOperation(u);
  if (hasWeapon(u, 'u16')) u.bonusAttacks++;
}
export function finishMode(s: GameState, c: Command) {
  const u = actor(s, c.unitId);
  ensure(u.mode === 'attack' || (u.mode === 'move' && isRunner(u)), '没有可提前结束的连续操作。');
  ensure(emptyFor(s, u), '必须先移到空地，不能结束在另一个棋子或基地内。');
  const wasMove = u.mode === 'move';
  finishOperation(u);
  if (wasMove && hasWeapon(u, 'u16')) u.bonusAttacks++;
}
export function react(s: GameState, c: Command, ctx: Resolution) {
  const r = s.pending.shift();
  ensure(r, '没有待结算反应。');
  if (r.kind === 'bounce') {
    const u = s.units.find((u) => u.id === r.source.id);
    if (!u) return;
    const to = point(c.x, c.y);
    ensure(
      distance(u, to) === 1 && canEnter(s, u, to),
      '必须向一个合法方向弹出，不能进入友方或越界。',
    );
    const t = targets(s).find(
      (t) => t.id !== u.id && (t.unit ? cells(t.unit) : [t]).some((p) => equal(p, to)),
    );
    emit(s, { type: 'move', from: u, to, unitId: u.id, owner: u.owner, text: '免费弹出' });
    Object.assign(u, to);
    if (t) {
      damage(s, t, 30, { owner: u.owner, unit: u, kind: 'collision' }, ctx);
      if (alive(s, u))
        s.pending.unshift({
          kind: 'bounce',
          owner: u.owner,
          source: structuredClone(u),
          amount: 30,
        });
    } else if (u.moves === 0) {
      finishOperation(u);
      if (hasWeapon(u, 'u16')) u.bonusAttacks++;
    }
    return;
  }
  if (r.kind === 'hut-spawn') {
    const hut = s.units.find((u) => u.id === r.source.id);
    if (!hut || hut.silenced) return;
    if (c.x === undefined) {
      emit(s, { type: 'skill', owner: r.owner, text: '放弃召唤' });
      return;
    }
    const to = point(c.x, c.y),
      ghost = template(20, r.owner, s.turns[r.owner], to);
    ensure(
      canPlace(s, ghost, to) && attackPath(s, hut, to, getStats(s, hut).range),
      '小屋召唤须在其范围内的合法空地。',
    );
    ensure(hut.maxHp >= 10, '小屋生命上限不足。');
    addUnit(s, 20, r.owner, to);
    lowerMax(s, hut, 10, ctx);
    return;
  }
  if (!c.targetId) {
    emit(s, { type: 'skill', owner: r.owner, text: '放弃效果' });
    return;
  }
  const t = findTarget(s, c.targetId);
  if (r.kind === 'death-shot')
    performAttack(s, r.source, t, ctx, { reactive: true, unlimited: true });
  else {
    ensure(
      t.unit ? allegiance(s, t.unit) !== r.owner : t.owner !== r.owner,
      '伤害转化只能指向敌方或中立。',
    );
    const path = attackPath(s, r.source, t, getStats(s, r.source).range);
    ensure(path, '目标不在伤害转化器范围内。');
    emit(s, { type: 'attack', from: r.source, to: t, path, owner: r.owner, text: '伤害转化' });
    damage(s, t, r.amount, { owner: r.owner, unit: r.source, kind: 'reflect' }, ctx);
  }
}
