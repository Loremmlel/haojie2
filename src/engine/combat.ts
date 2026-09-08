import { definition, COMBAT_RULES } from './catalog';
import { attackProfile, vampireRate } from './attack-profile';
import {
  attackPath,
  basePoint,
  canPlace,
  cells,
  equal,
  frontal,
  inSquare,
  occupants,
  other,
  ring,
  targets,
  topTarget,
} from './geometry';
import {
  activeEffect,
  addEffect,
  addUnit,
  allegiance,
  asTarget,
  emit,
  ensure,
  faction,
  findUnit,
  getStats,
  has,
  hasWeapon,
  now,
  passive,
  random,
  resetUnit,
} from './state';
import type { Effect, GameState, Player, Point, Source, Target, Unit } from './types';
export interface Resolution {
  retaliations: Set<string>;
  protection: Map<string, boolean>;
  token: number;
}
export const resolution = (): Resolution => ({
  retaliations: new Set(),
  protection: new Map(),
  token: 0,
});
export const alive = (s: GameState, u: Unit) => s.units.some((v) => v.id === u.id);
export function findTarget(s: GameState, id?: string): Target {
  const t = targets(s).find((t) => t.id === id);
  ensure(t, '请选择有效的目标。');
  return t;
}
export const hostile = (s: GameState, u: Unit, t: Target) =>
  t.unit ? allegiance(s, t.unit) !== u.owner : t.owner !== u.owner;
const targetEffects = (s: GameState, t: Target) =>
  t.unit ? t.unit.effects : s.baseEffects[t.owner];
function removeEffect(s: GameState, t: Target, e: Effect) {
  if (t.unit) t.unit.effects = t.unit.effects.filter((v) => v !== e);
  else s.baseEffects[t.owner] = s.baseEffects[t.owner].filter((v) => v !== e);
}
/** One shield decision per target/effect packet; pure UI previews execute on a cloned state. */
export function protectedEffect(s: GameState, t: Target, source: Source, ctx: Resolution): boolean {
  if (!t.unit || source.owner === undefined || source.owner === t.owner) return false;
  if (has(s, t.unit, 'immune')) {
    emit(s, { type: 'shield', to: t, owner: t.owner, text: '金身免疫' });
    return true;
  }
  if (!['spell', 'skill'].includes(source.kind)) return false;
  const key = `${ctx.token}:${source.owner}:${t.id}`;
  if (ctx.protection.has(key)) return ctx.protection.get(key)!;
  const tower = s.units.find(
    (u) =>
      u.kind === 'u15' &&
      u.owner === t.owner &&
      passive(s, u) &&
      attackPath(s, u, t, getStats(s, u).range),
  );
  ctx.protection.set(key, !!tower);
  if (tower) {
    emit(s, { type: 'shield', to: t, from: tower, owner: t.owner, text: '免疫塔' });
    lowerMax(s, tower, 15, ctx);
    return true;
  }
  return false;
}
export function lowerMax(s: GameState, u: Unit, amount: number, ctx: Resolution) {
  if (!alive(s, u)) return;
  u.maxHp = Math.max(0, u.maxHp - amount);
  u.hp = Math.min(u.hp, u.maxHp);
  if (u.hp <= 0) kill(s, u, { owner: u.owner, kind: 'sacrifice' }, ctx);
}
export function heal(s: GameState, t: Unit | Target, amount: number, ctx = resolution()) {
  const target = 'kind' in t ? asTarget(t) : t,
    u = target.unit;
  if (u && !alive(s, u)) return;
  const gain = Math.max(0, Math.min(amount, u ? u.maxHp - u.hp : 300 - s.bases[target.owner]));
  if (u) u.hp = Math.round((u.hp + gain) * 1e6) / 1e6;
  else s.bases[target.owner] += gain;
  if (gain)
    emit(
      s,
      { type: 'heal', to: target, amount: gain, owner: target.owner },
      `${u ? definition(u.kind).name : '基地'}回复${gain}生命`,
    );
  if (u ? u.hp >= u.maxHp : s.bases[target.owner] >= 300)
    for (const e of [...targetEffects(s, target)])
      if (e.type === 'mark' && activeEffect(s, e, u)) {
        removeEffect(s, target, e);
        damage(
          s,
          target,
          5,
          { owner: e.owner, kind: 'status', unit: s.units.find((v) => v.id === e.sourceId) },
          ctx,
        );
      }
}
export function kill(
  s: GameState,
  u: Unit,
  source: Source = { kind: 'expire' },
  ctx = resolution(),
) {
  if (!alive(s, u)) return;
  const snap = structuredClone(u);
  u.hp = 0;
  s.units = s.units.filter((v) => v.id !== u.id);
  s.deaths.push({
    id: `dead${s.serial++}`,
    kind: u.kind,
    owner: u.owner,
    ply: s.ply,
    revived: false,
    ...(u.group ? { group: u.group } : {}),
  });
  s.deaths = s.deaths.slice(-240);
  emit(
    s,
    { type: 'death', to: u, unitId: u.id, owner: u.owner },
    `${faction(u.owner)}的${definition(u.kind).name}离场`,
  );
  s.siphons = s.siphons.filter((l) => l.sourceId !== u.id && l.fromId !== u.id && l.toId !== u.id);
  s.iceMarks = s.iceMarks.filter((m) => m.sourceId !== u.id);
  const groupFinal =
    !u.group ||
    (!s.units.some((v) => v.group === u.group) &&
      ![...s.hands[1], ...s.hands[2]].some((c) => c.group === u.group));
  if (!groupFinal) return;
  const enabled = !u.silenced;
  const denyHead = enabled && u.kind === 20 && random(s, [0, 0.5, 1]) < 0.5;
  const enemyKill =
    source.owner !== undefined && source.owner !== u.owner && source.kind !== 'expire';
  if (enemyKill && !denyHead) {
    s.heads[source.owner!]++;
    emit(
      s,
      { type: 'skill', owner: source.owner, text: '+1 人头' },
      `${faction(source.owner!)}获得1人头`,
    );
  }
  if (denyHead)
    emit(
      s,
      { type: 'skill', to: u, owner: u.owner, text: '人头遁逃' },
      '超级跑得快：本次死亡不提供人头',
    );
  const killer = source.unit && s.units.find((v) => v.id === source.unit!.id);
  if (enemyKill && killer && !killer.silenced) {
    killer.kills++;
    if (killer.kind === 26) {
      const step = (killer.kills - 1) % 3;
      if (step === 0) {
        killer.maxHp += 10;
        heal(s, killer, 10, ctx);
      }
      if (step === 1) killer.attackBonus += 5;
      if (step === 2) killer.rangeBonus++;
    }
    if (killer.kind === 'u8' && killer.kills === 5) killer.rangeBonus++;
    if (killer.kind === 'u23') {
      killer.hookReadyAt = now(s, killer) + 2;
      killer.hookExpiresAt = now(s, killer) + 4;
    }
  }
  if (enabled) {
    if (u.kind === 11) s.bonus[u.owner]++;
    if (u.kind === 12) {
      const grave = addUnit(s, 'grave', u.owner, u);
      grave.size = 1;
    }
    if (u.kind === 2)
      s.pending.push({ kind: 'death-shot', owner: u.owner, source: snap, amount: 20 });
    if (u.kind === 'u21')
      for (const friend of [...s.units])
        if (
          friend.owner === u.owner &&
          attackPath(s, snap, asTarget(friend), getStats(s, snap).range)
        )
          heal(s, friend, 25, ctx);
    if (
      u.kind === 20 &&
      !denyHead &&
      source.unit &&
      source.unit.id !== u.id &&
      !['sacrifice', 'expire'].includes(source.kind)
    ) {
      const target = s.units.find((v) => v.id === source.unit!.id);
      if (target)
        damage(
          s,
          asTarget(target),
          30,
          { owner: u.owner, unit: snap, kind: 'reflect', retaliated: true },
          ctx,
        );
    }
  }
  for (const hut of [...s.units])
    if (
      hut.kind === 'u22' &&
      hut.owner === u.owner &&
      passive(s, hut) &&
      attackPath(s, hut, asTarget(snap), getStats(s, hut).range)
    ) {
      if (
        !s.pending.some(
          (r) =>
            r.kind === 'hut-spawn' && r.source.id === hut.id && r.source.lastCharge === s.serial,
        )
      ) {
        s.pending.push({
          kind: 'hut-spawn',
          owner: hut.owner,
          source: structuredClone(hut),
          amount: 0,
        });
      }
    }
}
/** Damage returns actual HP removed. Source ownership is retained for spell/DOT head credit. */
export function damage(
  s: GameState,
  t: Target,
  amount: number,
  source: Source,
  ctx = resolution(),
): number {
  amount = Math.max(0, amount);
  if (amount <= 0) return 0;
  if (!t.unit) {
    const loss = Math.min(s.bases[t.owner], amount);
    s.bases[t.owner] -= loss;
    if (loss)
      emit(
        s,
        { type: 'damage', to: t, unitId: t.id, owner: t.owner, amount: loss },
        `${faction(t.owner)}基地受到${loss}伤害`,
      );
    return loss;
  }
  const u = t.unit;
  if (!alive(s, u)) return 0;
  if (has(s, u, 'immune')) {
    emit(s, { type: 'shield', to: u, owner: u.owner, text: '金身' });
    return 0;
  }
  if (protectedEffect(s, t, source, ctx)) return 0;
  if (
    !u.silenced &&
    u.kind === 'u18' &&
    source.kind === 'attack' &&
    amount <= COMBAT_RULES.kingAttackImmunity
  ) {
    emit(s, { type: 'shield', to: u, owner: u.owner, text: '王之蔑视' });
    return 0;
  }
  if (!u.silenced && u.kind === 24 && source.path && frontal(source.path, u.owner))
    amount = Math.min(amount, COMBAT_RULES.frontDamageCap);
  const before = u.hp;
  const guardian =
    !u.guardUsed &&
    amount >= u.hp &&
    s.units.find(
      (v) =>
        v.kind === 3 &&
        allegiance(s, v) === u.owner &&
        passive(s, v) &&
        attackPath(s, v, t, getStats(s, v).range),
    );
  if (guardian) {
    u.hp = 1;
    u.guardUsed = true;
    emit(s, { type: 'shield', to: u, owner: u.owner, text: '名刀' });
  } else u.hp = Math.max(0, Math.round((u.hp - amount) * 1e6) / 1e6);
  const loss = before - u.hp;
  if (loss)
    emit(
      s,
      { type: 'damage', to: u, unitId: u.id, amount: loss, owner: u.owner },
      `${definition(u.kind).name}受到${loss}伤害`,
    );
  if (loss > 0 && !u.silenced && u.kind === 'u10') u.attackBonus += 15;
  const snap = structuredClone(u);
  if (u.hp <= 0) kill(s, u, source, ctx);
  if (
    loss > 0 &&
    !snap.silenced &&
    u.kind === 16 &&
    source.unit &&
    source.owner === u.owner &&
    source.unit.id !== u.id
  )
    s.pending.push({ kind: 'reflect', owner: u.owner, source: snap, amount: loss });
  if (
    loss > 0 &&
    alive(s, u) &&
    !u.silenced &&
    u.kind === 'u18' &&
    source.unit &&
    source.unit.id !== u.id &&
    !has(s, u, 'freeze') &&
    !has(s, u, 'stun')
  ) {
    const attacker = s.units.find((v) => v.id === source.unit!.id),
      pair = `${u.id}>${source.unit.id}`;
    if (
      attacker &&
      !ctx.retaliations.has(pair) &&
      attackPath(s, u, asTarget(attacker), getStats(s, u).range)
    ) {
      ctx.retaliations.add(pair);
      performAttack(s, u, asTarget(attacker), ctx, { reactive: true, forceHostile: true });
    }
  }
  return loss;
}
export function freeze(s: GameState, t: Target, source: Source, ctx: Resolution, extra = 0) {
  if (
    !t.unit ||
    !alive(s, t.unit) ||
    hasWeapon(t.unit, 'u28') ||
    protectedEffect(s, t, source, ctx)
  )
    return;
  t.unit.effects = t.unit.effects.filter((e) => e.type !== 'freeze');
  addEffect(s, t.unit, 'freeze', source.owner!, 0, 4, 5 + extra, source.unit?.id);
  emit(s, { type: 'shield', to: t, owner: source.owner, text: '冰冻 · 中立' });
}
export function burn(s: GameState, t: Target, source: Source, ctx: Resolution) {
  if (!t.unit || !alive(s, t.unit) || protectedEffect(s, t, source, ctx)) return;
  t.unit.effects = t.unit.effects.filter((e) => e.type !== 'burn');
  addEffect(s, t.unit, 'burn', source.owner!, 0, 12, 5, source.unit?.id);
  emit(s, { type: 'skill', to: t, owner: source.owner, text: '灼烧' });
}
function knockback(s: GameState, u: Unit, t: Target, path: Point[], ctx: Resolution) {
  if (!t.unit || !alive(s, t.unit) || path.length < 2) return;
  if (protectedEffect(s, t, { owner: u.owner, unit: u, kind: 'skill' }, ctx)) return;
  const victim = t.unit,
    a = path.at(-2)!,
    b = path.at(-1)!,
    dx = b.x - a.x,
    dy = b.y - a.y;
  const first = { x: victim.x + dx, y: victim.y + dy },
    second = { x: victim.x + 2 * dx, y: victim.y + 2 * dy };
  const behind = [
    ...new Set(
      [...cells({ ...victim, ...first }), ...cells({ ...victim, ...second })]
        .flatMap((p) => occupants(s, p))
        .filter((v) => v.id !== victim.id),
    ),
  ];
  if (behind.some((v) => v.size > 1) || behind.length > 1) return;
  if (behind.length === 1) {
    const follower = behind[0],
      to = { x: follower.x + dx, y: follower.y + dy };
    if (
      canPlace(s, victim, first, false, [follower.id]) &&
      canPlace(s, follower, to, false, [victim.id])
    ) {
      emit(s, {
        type: 'move',
        from: follower,
        to,
        unitId: follower.id,
        owner: follower.owner,
        text: '连带击退',
      });
      Object.assign(follower, to);
      emit(s, {
        type: 'move',
        from: victim,
        to: first,
        unitId: victim.id,
        owner: victim.owner,
        text: '击退',
      });
      Object.assign(victim, first);
    }
    return;
  }
  if (cells({ ...victim, ...second }).some((p) => p.x < 1 || p.x > 9 || p.y < 1 || p.y > 13)) {
    damage(s, t, 30, { owner: u.owner, unit: u, kind: 'skill' }, ctx);
    return;
  }
  if (canPlace(s, victim, first) && canPlace(s, victim, second)) {
    emit(s, {
      type: 'move',
      from: victim,
      to: second,
      unitId: victim.id,
      owner: victim.owner,
      text: '击退',
    });
    Object.assign(victim, second);
  }
}
interface AttackOptions {
  reactive?: boolean;
  unlimited?: boolean;
  forceHostile?: boolean;
  amount?: number;
  noPierce?: boolean;
  weaponFirst?: boolean;
}
/** Resolve a hit without spending mode resources. Command layer owns operation counters. */
export function performAttack(
  s: GameState,
  u: Unit,
  t: Target,
  ctx = resolution(),
  options: AttackOptions = {},
) {
  const ally = t.unit ? allegiance(s, t.unit) === u.owner : t.owner === u.owner;
  ensure(
    !ally ||
      options.forceHostile ||
      (t.unit &&
        ((!u.silenced && (u.kind === 2 || u.kind === 'u21')) ||
          (t.unit.kind === 16 && !t.unit.silenced && t.id !== u.id))),
    '只能攻击敌方或中立；奶妈可治疗友方，伤害转化器可受友伤。',
  );
  ensure(topTarget(s, t), '单体攻击或技能只能命中当前叠放栈顶。');
  const stats = getStats(s, u);
  ensure(u.kind !== 'firelord' || u.silenced, '炎魔之王不能普通攻击。');
  if (!options.reactive) {
    ensure(u.kind !== 4 || u.silenced || u.readyCharge >= 2, '定炮回合开始至少有2层蓄力才可开炮。');
    ensure(
      u.kind !== 9 || u.silenced || !u.attacked.includes(t.id),
      '射手不能重复攻击本回合的同一目标。',
    );
  }
  const rays = cells(u)
    .flatMap((start) =>
      (t.unit ? cells(t.unit) : [t])
        .filter((end) => (start.x === end.x || start.y === end.y) && !equal(start, end))
        .map((end) => {
          const length = Math.abs(end.x - start.x) + Math.abs(end.y - start.y);
          return Array.from({ length: length + 1 }, (_, i) => ({
            x: start.x + Math.sign(end.x - start.x) * i,
            y: start.y + Math.sign(end.y - start.y) * i,
          }));
        }),
    )
    .filter((p) => p.length - 1 <= stats.range)
    .sort((a, b) => a.length - b.length);
  const path =
    hasWeapon(u, 'u28') && !ally
      ? rays[0]
      : attackPath(s, u, t, options.unlimited ? 117 : stats.range);
  ensure(path, '目标不在射程内，或所有路径均被阻挡；炎魔之心必须选同一直线。');
  if (hasWeapon(u, 'u28') && !options.noPierce && !ally) {
    const origin = path[0],
      dx = path[1].x - origin.x,
      dy = path[1].y - origin.y,
      victims: Target[] = [];
    for (let i = 1; i <= stats.range; i++) {
      const p = { x: origin.x + dx * i, y: origin.y + dy * i };
      const rows = targets(s).filter(
        (v) => (v.unit ? cells(v.unit) : [v]).some((c) => equal(c, p)) && hostile(s, u, v),
      );
      for (const v of rows)
        if (!victims.some((old) => old.id === v.id) && topTarget(s, v)) victims.push(v);
    }
    for (const victim of victims)
      if (!victim.unit || alive(s, victim.unit))
        performAttack(s, u, victim, ctx, {
          ...options,
          noPierce: true,
          amount: stats.attack,
          weaponFirst: !u.weaponFirstUsed,
        });
    if (hasWeapon(u, 'u11')) u.weaponFirstUsed = true;
    return;
  }
  ctx.retaliations.add(`${u.id}>${t.id}`);
  emit(s, {
    type: 'attack',
    from: u,
    to: t,
    path,
    unitId: u.id,
    owner: u.owner,
    text: ally && (u.kind === 2 || u.kind === 'u21') ? '治疗' : '攻击',
    ultimate: definition(u.kind).tier !== 'normal',
  });
  if (
    ally &&
    !options.forceHostile &&
    !u.silenced &&
    (u.kind === 2 || u.kind === 'u21') &&
    t.unit
  ) {
    heal(s, t.unit, u.kind === 2 ? 20 : 25, ctx);
    return;
  }
  const skillSource: Source = { owner: u.owner, unit: u, kind: 'skill' };
  const execute =
    !ally && t.unit && u.effects.find((e) => e.type === 'execute' && activeEffect(s, e, u));
  if (execute && t.unit) {
    u.effects = u.effects.filter((e) => e !== execute);
    if (!protectedEffect(s, t, skillSource, ctx)) kill(s, t.unit, skillSource, ctx);
    return;
  }
  const mark =
    !ally &&
    u.kind !== 10 &&
    targetEffects(s, t).find(
      (e) => e.type === 'mark' && e.owner === u.owner && activeEffect(s, e, t.unit),
    );
  if (mark) removeEffect(s, t, mark);
  const profile = attackProfile(
    u.kind,
    options.amount ?? stats.attack,
    u.kills,
    u.silenced,
    !t.unit,
  );
  let amount = profile.packets[0].damage;
  if (profile.cuts) {
    const roll = random(s, profile.cuts);
    const index = profile.cuts.slice(1).findIndex((cut) => roll < cut);
    amount = profile.packets[index < 0 ? profile.packets.length - 1 : index].damage;
  }
  const lifesteal = !u.silenced && u.kind === 'u8' ? vampireRate(u.kills) : 0;
  let loss = 0;
  if (u.kind === 10 && !u.silenced && !ally) {
    loss = damage(s, t, amount, { owner: u.owner, unit: u, kind: 'attack', path }, ctx);
    const old = targetEffects(s, t).find((e) => e.type === 'mark' && e.owner === u.owner);
    if (old) removeEffect(s, t, old);
    const e: Effect = {
      type: 'mark',
      owner: u.owner,
      sourceId: u.id,
      from: s.ply,
      until: s.ply + 2,
      global: true,
    };
    targetEffects(s, t).push(e);
    if (t.unit ? t.unit.hp >= t.unit.maxHp : s.bases[t.owner] >= 300) {
      removeEffect(s, t, e);
      loss = damage(
        s,
        t,
        COMBAT_RULES.catapultMarkDamage,
        { owner: u.owner, unit: u, kind: 'attack', path },
        ctx,
      );
    } else emit(s, { type: 'skill', to: t, owner: u.owner, text: '标记' });
  } else loss = damage(s, t, amount, { owner: u.owner, unit: u, kind: 'attack', path }, ctx);
  if (mark)
    loss += damage(
      s,
      t,
      COMBAT_RULES.catapultMarkDamage,
      { owner: u.owner, unit: u, kind: 'status' },
      ctx,
    );
  let drain = lifesteal;
  if (hasWeapon(u, 'u11') && (options.weaponFirst ?? !u.weaponFirstUsed)) drain += 1;
  if (loss > 0 && drain > 0) heal(s, u, loss * drain, ctx);
  // The first-attack weapon applies to this complete attack, not to every turn hit.
  if (hasWeapon(u, 'u11') && options.weaponFirst === undefined) u.weaponFirstUsed = true;
  if (!u.silenced && u.kind === 'u2') u.charge = u.readyCharge = 0;
  if (!u.silenced && u.kind === 4) u.charge = u.readyCharge = 0;
  const victim = t.unit;
  if (victim && alive(s, victim) && !ally) {
    const conversion = u.effects.find((e) => e.type === 'convert' && activeEffect(s, e, u));
    if (conversion && loss > 0) {
      u.effects = u.effects.filter((e) => e !== conversion);
      if (!protectedEffect(s, t, skillSource, ctx)) {
        victim.owner = u.owner;
        victim.offset = 0;
        victim.born = s.turns[u.owner] - (victim.kind === 23 ? 1 : 0);
        victim.effects = [];
        resetUnit(s, victim);
        victim.operations = 1;
        emit(
          s,
          { type: 'skill', to: victim, owner: u.owner, text: '策反' },
          `${definition(victim.kind).name}加入${faction(u.owner)}`,
        );
        return;
      }
    }
    if (!u.silenced && u.kind === 'u4' && !protectedEffect(s, t, skillSource, ctx)) {
      victim.silenced = true;
      addEffect(s, victim, 'stun', u.owner, 0, 2, undefined, u.id);
      emit(s, { type: 'skill', to: victim, owner: u.owner, text: '沉默 · 眩晕' });
    }
    if (hasWeapon(u, 'u5')) freeze(s, t, skillSource, ctx);
    if (hasWeapon(u, 'u28') || (!u.silenced && u.kind === 'u6' && !hasWeapon(u, 'u5')))
      burn(s, t, skillSource, ctx);
    if (!u.silenced && u.kind === 'u20') knockback(s, u, t, path, ctx);
  } else if (victim && !ally && loss > 0)
    u.effects = u.effects.filter((e) => !(e.type === 'convert' && activeEffect(s, e, u)));
}
export function pruneSiphons(s: GameState) {
  s.siphons = s.siphons.filter((l) => {
    const u = s.units.find((v) => v.id === l.sourceId),
      a = targets(s).find((t) => t.id === l.fromId),
      b = targets(s).find((t) => t.id === l.toId);
    return (
      u &&
      passive(s, u) &&
      a &&
      b &&
      attackPath(s, u, a, getStats(s, u).range) &&
      attackPath(s, u, b, getStats(s, u).range)
    );
  });
}
