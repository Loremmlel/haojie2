import { eventActor, withEventFacts, type EventAction } from './event-facts';
import { definition, isStored } from './catalog';
import {
  alive,
  damage,
  findTarget,
  freeze,
  heal,
  kill,
  lowerMax,
  performAttack,
  protectedEffect,
  pruneSiphons,
} from './combat';
import type { Resolution } from './combat';
import {
  ALL_CELLS,
  attackPath,
  canPlace,
  cells,
  equal,
  inSquare,
  inside,
  movementPath,
  occupants,
  ring,
  targets,
  topTarget,
} from './geometry';
import {
  actor,
  addEffect,
  addUnit,
  age,
  allegiance,
  asTarget,
  chooseMode,
  draw,
  emit,
  ensure,
  findUnit,
  finishOperation,
  getStats,
  has,
  hasWeapon,
  now,
  passive,
  point,
  random,
  template,
} from './state';
import { advanceUnit } from './lifecycle';
import type { Card, Command, GameState, Kind, Player, Source, Unit } from './types';
export function chargeAction(s: GameState, c: Command) {
  return withEventFacts(s, { action: 'charge', actor: eventActor(findUnit(s, c.unitId)) }, () =>
    resolveCharge(s, c),
  );
}
function resolveCharge(s: GameState, c: Command) {
  const u = actor(s, c.unitId),
    d = definition(u.kind);
  chooseMode(s, u, 'charge');
  const mode = c.mode as 'move' | 'attack' | 'skill';
  const max =
    mode === 'move' && d.move % 1 !== 0
      ? 1
      : mode === 'attack' && u.kind === 4 && !u.silenced
        ? 5
        : mode === 'attack' && u.kind === 'u2' && !u.silenced
          ? 4
          : mode === 'skill' && u.kind === 21 && !u.silenced
            ? 2
            : mode === 'skill' && u.kind === 'u6' && !u.silenced && !u.onceUsed
              ? 1
              : 0;
  ensure(max > 0, '这枚随从没有此类蓄力。');
  ensure(u.charge < max, '该类蓄力已满。');
  ensure(u.charge === 0 || u.chargeType === mode, '当前蓄力属于另一模式。');
  u.chargeType = mode;
  u.charge++;
  u.lastCharge = now(s, u);
  finishOperation(u);
  emit(s, { type: 'skill', to: u, owner: u.owner, text: `蓄力 ${u.charge}/${max}` });
}
export function useSkill(s: GameState, c: Command, ctx: Resolution) {
  const u = findUnit(s, c.unitId);
  const actions: Partial<Record<Kind, EventAction>> = {
    5: 'quake',
    7: 'pull',
    21: 'rush',
    u6: 'cross',
    u14: 'siphon',
    u23: 'pull',
    u24: 'ice-mark',
  };
  const area =
    u.kind === 5
      ? ALL_CELLS.filter((p) => ring(u, p))
      : u.kind === 'u6'
        ? ALL_CELLS.filter((p) => inSquare(u, p) && (p.x === c.x || p.y === c.y))
        : u.kind === 'u24'
          ? [point(c.x, c.y)]
          : undefined;
  const linkIds = new Set(s.siphons.map((l) => l.id));
  const endpointA = c.targetId ? targets(s).find((t) => t.id === c.targetId) : undefined;
  const endpointB = c.secondId ? targets(s).find((t) => t.id === c.secondId) : undefined;
  return withEventFacts(
    s,
    {
      action: actions[u.kind] ?? 'buff',
      ability: u.kind,
      actor: eventActor(u),
      ...(area ? { area } : {}),
    },
    () => {
      resolveSkill(s, c, ctx);
      const summary = s.events.at(-1);
      if (summary?.type !== 'skill') return;
      if (u.kind === 'u14') {
        summary.stage = s.siphons.some((l) => !linkIds.has(l.id)) ? 'apply' : 'blocked';
        summary.actor = eventActor(endpointA);
        summary.subject = eventActor(endpointB);
        if (endpointA) summary.from = { x: endpointA.x, y: endpointA.y };
        if (endpointB) summary.to = { x: endpointB.x, y: endpointB.y };
        // A blocked link already has its actual protection event; do not invent a successful flow.
        if (summary.stage === 'blocked') {
          delete summary.to;
          delete summary.actor;
          delete summary.subject;
        }
      } else if (u.kind === 'u24') {
        summary.to = point(c.x, c.y);
        delete summary.subject;
      } else if ((u.kind === 'u7' || u.kind === 'u21') && endpointA) {
        summary.to = { x: endpointA.x, y: endpointA.y };
        summary.subject = eventActor(endpointA);
      }
    },
  );
}
function resolveSkill(s: GameState, c: Command, ctx: Resolution) {
  const raw = findUnit(s, c.unitId),
    free = raw.kind === 'u7' || raw.kind === 'u14';
  const u = raw.kind === 'u7' ? raw : actor(s, c.unitId);
  ensure(!u.silenced, '沉默已移除此随从的技能。');
  ensure(!has(s, u, 'freeze') && !has(s, u, 'stun'), '冻结或眩晕中不能施放技能。');
  if (!free) chooseMode(s, u, 'skill');
  else ensure(u.freeUsed !== now(s, u), '本回合免费技能已使用。');
  const range = getStats(s, u).range,
    source: Source = { owner: u.owner, unit: u, kind: 'skill' };
  const friendly = (id?: string) => {
    const v = findUnit(s, id);
    ensure(allegiance(s, v) === u.owner, '请选择友方随从。');
    ensure(attackPath(s, u, asTarget(v), range), '目标不在技能范围内。');
    return v;
  };
  const enemy = (id?: string, global = false) => {
    const t = findTarget(s, id);
    ensure(
      t.unit && allegiance(s, t.unit) !== u.owner && topTarget(s, t),
      '请选择敌方或中立的栈顶随从。',
    );
    ensure(global || attackPath(s, u, t, range), '目标不在技能射程内。');
    return t;
  };
  switch (u.kind) {
    case 5: {
      const victims = targets(s).filter(
        (t) => ring(u, t.unit ?? t) && (t.owner === u.owner || topTarget(s, t)),
      );
      for (const t of victims) damage(s, t, 20, source, ctx);
      break;
    }
    case 6:
      for (const friend of s.units)
        if (
          allegiance(s, friend) === u.owner &&
          friend.kind !== 10 &&
          attackPath(s, u, asTarget(friend), range)
        )
          addEffect(s, friend, 'attack', u.owner, 2, 2, 10, u.id);
      break;
    case 7: {
      const t = enemy(c.targetId),
        to = point(c.x, c.y);
      ensure(
        !equal(t, to) &&
          canPlace(s, t.unit!, to) &&
          attackPath(s, u, asTarget({ ...t.unit!, ...to }), range),
        '牵引落点必须合法且在钩子射程内。',
      );
      if (!protectedEffect(s, t, source, ctx)) {
        emit(s, { type: 'move', stage: 'trigger', from: t, to, unitId: t.id, owner: t.owner });
        Object.assign(t.unit!, to);
      }
      break;
    }
    case 14: {
      const victim = friendly(c.targetId);
      ensure(victim.id !== u.id && victim.kind !== 'u25', '不可献祭自身或克隆军团。');
      ensure(u.maxHp >= 10, '生命上限不足10。');
      ensure(Number.isInteger(c.column) && c.column! >= 1 && c.column! <= 9, '请选择一列。');
      const candidates = targets(s).filter(
        (t) =>
          t.owner !== u.owner &&
          (t.unit ? cells(t.unit) : [t]).some(
            (p) => p.x === c.column && (u.owner === 1 ? p.y >= u.y : p.y <= u.y),
          ) &&
          attackPath(s, u, t, range) &&
          topTarget(s, t),
      );
      candidates.sort((a, b) => (u.owner === 1 ? a.y - b.y : b.y - a.y));
      const t = candidates[0];
      ensure(t, '这一列没有射程内的敌方目标。');
      const amount = getStats(s, victim).attack;
      lowerMax(s, u, 10, ctx);
      kill(s, victim, { owner: u.owner, unit: u, kind: 'sacrifice' }, ctx);
      damage(s, t, amount, source, ctx);
      break;
    }
    case 15:
      ensure(u.upgrades < 3, '强化只能使用3次。');
      ensure(c.mode === 'attack' || c.mode === 'range', '请选择攻击或射程强化。');
      if (c.mode === 'attack') u.attackBonus += 10;
      else u.rangeBonus++;
      u.upgrades++;
      break;
    case 19: {
      const to = point(c.x, c.y),
        ghost = template('wall', u.owner, s.turns[u.owner], to);
      ensure(canPlace(s, ghost, to) && attackPath(s, u, to, range), '路障须放在射程内合法空格。');
      const v = addUnit(s, 'wall', u.owner, to);
      v.expiresAt = s.ply + 2;
      break;
    }
    case 21: {
      ensure(u.readyCharge >= 2 && u.chargeType === 'skill', '回合开始需要已经持有两层技能蓄力。');
      ensure(u.hp > 10, '突袭扣10血后必须存活。');
      const to = point(c.x, c.y),
        t = findTarget(s, c.targetId);
      const route = movementPath(s, u, to, 6);
      ensure(route, '落点必须在6格可达范围内。');
      ensure(
        t.owner !== u.owner && attackPath(s, { ...u, ...to }, t, range),
        '落点无法攻击所选敌方。',
      );
      u.hp -= 10;
      emit(s, {
        type: 'move',
        from: u,
        to,
        path: route,
        unitId: u.id,
        owner: u.owner,
        text: '神行突袭',
      });
      Object.assign(u, to);
      u.charge = u.readyCharge = 0;
      performAttack(s, u, t, ctx, { reactive: true });
      break;
    }
    case 'u6': {
      ensure(
        !u.onceUsed && u.readyCharge >= 1 && u.chargeType === 'skill',
        '十字浩劫需要前一回合蓄力，且一生只能发动一次。',
      );
      const p = point(c.x, c.y);
      ensure(inSquare(u, p), '交点必须位于自身11×11区域。');
      const victims = targets(s).filter(
        (t) =>
          (t.owner === u.owner || topTarget(s, t)) &&
          (t.unit ? cells(t.unit) : [t]).some(
            (cell) => inSquare(u, cell) && (cell.x === p.x || cell.y === p.y),
          ),
      );
      emit(s, { type: 'skill', to: p, owner: u.owner, text: '十字浩劫', ultimate: true });
      for (const t of victims) {
        const center = (t.unit ? cells(t.unit) : [t]).some((cell) => equal(cell, p));
        damage(s, t, center ? 40 : 20, source, ctx);
      }
      u.onceUsed = true;
      u.charge = u.readyCharge = 0;
      break;
    }
    case 'u7': {
      ensure(!u.onceUsed, '巨大化一生只能用一次。');
      const v = findUnit(s, c.targetId);
      ensure(
        v.owner === u.owner && v.id !== u.id && v.size === 1 && !has(s, v, 'freeze'),
        '请选择另一个单格友方随从。',
      );
      ensure(canPlace(s, { ...v, size: 2 }, v), '周围空间不足，无法形成完整2×2占位。');
      ensure(
        u.hp > 10 && u.maxHp > 10 && getStats(s, u).attack >= 10,
        '发动需要支付10攻击和10生命并存活。',
      );
      u.attackBonus -= 10;
      u.hp -= 10;
      v.size = 2;
      v.maxHp += 5;
      v.hp += 5;
      u.onceUsed = true;
      break;
    }
    case 'u14': {
      const a = findTarget(s, c.targetId),
        b = findTarget(s, c.secondId);
      ensure(a.id !== b.id, '虹吸的两个端点不能相同。');
      ensure(
        attackPath(s, u, a, range) && attackPath(s, u, b, range),
        '虹吸两个目标均须在射程内。',
      );
      ensure(s.siphons.filter((l) => l.sourceId === u.id).length < 3, '最多存在3条虹吸。');
      ensure(
        (a.owner === u.owner || topTarget(s, a)) && (b.owner === u.owner || topTarget(s, b)),
        '敌方叠放目标只能选栈顶。',
      );
      if (!protectedEffect(s, a, source, ctx) && !protectedEffect(s, b, source, ctx))
        s.siphons.push({
          id: `link${s.serial++}`,
          sourceId: u.id,
          owner: u.owner,
          fromId: a.id,
          toId: b.id,
        });
      break;
    }
    case 'u19': {
      ensure(u.maxHp >= 20, '生命上限不足以支付20点复活代价。');
      const record = s.deaths.find((r) => r.id === c.deathId);
      ensure(
        record &&
          record.owner === u.owner &&
          !record.revived &&
          record.ply < s.ply &&
          record.ply >= s.ply - 4,
        '只能选择前两个己方回合窗口内尚未复活的友方阵亡记录。',
      );
      const to = point(c.x, c.y),
        ghost = template(record.kind, u.owner, s.turns[u.owner], to);
      ensure(canPlace(s, ghost, to) && attackPath(s, u, to, range), '复活位置须在射程内合法空格。');
      addUnit(
        s,
        record.kind,
        u.owner,
        to,
        record.kind === 'u25' ? `revived-group${s.serial++}` : undefined,
      );
      record.revived = true;
      lowerMax(s, u, 20, ctx);
      break;
    }
    case 'u21': {
      const v = friendly(c.targetId);
      addEffect(s, v, 'attack', u.owner, 2, 2, 10, u.id);
      break;
    }
    case 'u23': {
      ensure(
        u.hookReadyAt !== undefined && u.hookReadyAt <= now(s, u) && u.hookExpiresAt! > now(s, u),
        '仅击杀后的下个己方回合可使用超级钩子。',
      );
      const t = enemy(c.targetId, true),
        v = t.unit!,
        to = { x: u.x, y: u.owner === 1 ? u.y + u.size : u.y - v.size };
      ensure(canPlace(s, v, to), '身前没有合法落位。');
      if (!protectedEffect(s, t, source, ctx)) {
        emit(s, {
          type: 'move',
          stage: 'trigger',
          from: v,
          to,
          unitId: v.id,
          owner: v.owner,
          text: '超级牵引',
        });
        Object.assign(v, to);
      }
      delete u.hookReadyAt;
      delete u.hookExpiresAt;
      break;
    }
    case 'u24': {
      const to = point(c.x, c.y);
      ensure(inSquare(u, to), '标记须位于自身11×11区域。');
      s.iceMarks.push({
        id: `ice${s.serial++}`,
        sourceId: u.id,
        owner: u.owner,
        ...to,
        due: now(s, u) + 2,
      });
      break;
    }
    default:
      ensure(false, '此棋子没有可主动使用的技能；被动能力由引擎自动结算。');
  }
  if (free) u.freeUsed = now(s, u);
  else finishOperation(u);
  emit(
    s,
    {
      type: 'skill',
      to: u,
      owner: u.owner,
      text: definition(u.kind).skill ?? '技能',
      ultimate: definition(u.kind).tier !== 'normal',
    },
    `${definition(u.kind).name}施放技能`,
  );
  pruneSiphons(s);
}
export function cast(s: GameState, c: Command, ctx: Resolution) {
  const kind = s.hands[s.active].find((v) => v.id === c.cardId)?.kind;
  const actions: Partial<Record<Kind, EventAction>> = {
    8: 'bomb',
    17: 'ward',
    18: 'execution',
    22: 'conversion',
    u9: 'storm',
    u17: 'clock',
    u26: 'inner-fire',
  };
  const area =
    kind === 8
      ? ALL_CELLS.filter((p) => p.x >= c.x! && p.x <= c.x! + 1 && p.y >= c.y! && p.y <= c.y! + 1)
      : kind === 'u9'
        ? ALL_CELLS.filter((p) => (c.mode === 'row' ? p.y === c.row : p.x === c.column))
        : undefined;
  const target = c.targetId ? targets(s).find((t) => t.id === c.targetId) : undefined;
  return withEventFacts(
    s,
    {
      ...(kind !== undefined ? { action: actions[kind] ?? 'buff', ability: kind } : {}),
      stage: 'apply',
      subject: eventActor(target),
      ...(area ? { area } : {}),
    },
    () => resolveCast(s, c, ctx),
  );
}
function resolveCast(s: GameState, c: Command, ctx: Resolution) {
  const owner = s.active,
    card = s.hands[owner].find((v) => v.id === c.cardId);
  ensure(card && definition(card.kind).spell !== undefined, '请选择法术牌。');
  const source: Source = { owner, kind: 'spell' };
  // Validate choices before spending randomness or a card. applyCommand clones the complete state.
  let target = c.targetId ? findTarget(s, c.targetId) : undefined;
  if ([17, 18, 22, 'u17'].includes(card.kind))
    ensure(target?.unit && allegiance(s, target.unit) === owner, '请选择友方随从。');
  if (card.kind === 'u26') ensure(target?.unit, '心灵之火只能选择随从。');
  if (card.kind === 8) {
    const p = point(c.x, c.y);
    ensure(inside(p) && p.x <= 8 && p.y <= 12, '爆弹须选择完整2×2区域的左上格。');
  }
  if (card.kind === 'u9')
    ensure(
      (c.mode === 'row' && Number.isInteger(c.row) && c.row! >= 1 && c.row! <= 13) ||
        (c.mode === 'column' && Number.isInteger(c.column) && c.column! >= 1 && c.column! <= 9),
      '烈焰风暴须选择整行或整列。',
    );
  if (card.kind === 25 && c.mode === 'double') {
    ensure(
      c.sacrificeIds?.length === 2 && new Set(c.sacrificeIds).size === 2,
      '请选择两个不同的半血以上友方。',
    );
    ensure(
      c.sacrificeIds
        .map((id) => findUnit(s, id))
        .every((v) => v.owner === owner && v.hp * 2 >= v.maxHp && v.kind !== 'u25'),
      '不可献祭克隆军团，且友方需至少半血。',
    );
  }
  for (const mage of s.units.filter((u) => u.owner !== owner && u.kind === 'u3' && passive(s, u)))
    if (random(s, [0, 1 / 3, 1]) < 1 / 3) {
      s.hands[owner] = s.hands[owner].filter((v) => v.id !== card.id);
      emit(
        s,
        {
          type: 'shield',
          to: mage,
          owner: mage.owner,
          action: 'counter',
          stage: 'blocked',
          text: '法术反制',
        },
        `${definition(card.kind).name}被反制并消耗`,
      );
      return;
    }
  switch (card.kind) {
    case 8: {
      const victims = targets(s).filter((t) =>
        (t.unit ? cells(t.unit) : [t]).some(
          (p) => p.x >= c.x! && p.x <= c.x! + 1 && p.y >= c.y! && p.y <= c.y! + 1,
        ),
      );
      emit(s, { type: 'skill', to: { x: c.x! + 0.5, y: c.y! + 0.5 }, owner, text: '爆弹' });
      for (const t of victims) damage(s, t, 20, source, ctx);
      break;
    }
    case 17:
    case 18:
    case 22: {
      const u = target!.unit!;
      const type = card.kind === 17 ? 'immune' : card.kind === 18 ? 'execute' : 'convert';
      u.effects = u.effects.filter((e) => e.type !== type);
      addEffect(s, u, type, owner, card.kind === 17 ? 0 : 2, 2);
      emit(s, { type: 'shield', to: u, owner, text: definition(card.kind).name });
      break;
    }
    case 25: {
      if (c.mode === 'double') {
        for (const id of c.sacrificeIds!)
          kill(s, findUnit(s, id), { owner, kind: 'sacrifice' }, ctx);
        draw(s, owner, 2);
      } else draw(s, owner, 1);
      break;
    }
    case 'u9': {
      const axis = c.mode as 'row' | 'column',
        line = axis === 'row' ? c.row! : c.column!;
      s.hazards.push({ id: `hazard${s.serial++}`, owner, axis, line, due: s.ply + 2 });
      const victims = targets(s).filter((t) =>
        (t.unit ? cells(t.unit) : [t]).some((p) => (axis === 'row' ? p.y === line : p.x === line)),
      );
      for (const t of victims) damage(s, t, 20, source, ctx);
      break;
    }
    case 'u17':
      advanceUnit(s, target!.unit!, ctx);
      break;
    case 'u26':
      if (!protectedEffect(s, target!, source, ctx)) {
        target!.unit!.effects = target!.unit!.effects.filter((e) => e.type !== 'inner-fire');
        addEffect(
          s,
          target!.unit!,
          'inner-fire',
          owner,
          0,
          Number.MAX_SAFE_INTEGER - now(s, target!.unit!),
        );
      }
      break;
    default:
      ensure(false, '未定义的法术。');
  }
  s.hands[owner] = s.hands[owner].filter((v) => v.id !== card.id);
  emit(
    s,
    {
      type: 'skill',
      owner,
      text: definition(card.kind).name,
      ultimate: definition(card.kind).tier === 'ultimate',
    },
    `施放${definition(card.kind).name}`,
  );
}
export function equip(s: GameState, c: Command) {
  return withEventFacts(
    s,
    { action: 'equip', ability: s.hands[s.active].find((v) => v.id === c.cardId)?.kind },
    () => resolveEquip(s, c),
  );
}
function resolveEquip(s: GameState, c: Command) {
  const card = s.hands[s.active].find((v) => v.id === c.cardId);
  ensure(card && definition(card.kind).weapon !== undefined, '请选择武器牌。');
  const u = findUnit(s, c.targetId),
    d = definition(u.kind);
  ensure(allegiance(s, u) === s.active, '只能装备给友方随从，冰冻中立随从不可装备。');
  ensure(card.kind !== 'u5' || d.mage, '寒冰法杖仅限文档指定的法师名单。');
  ensure(card.kind !== 'u28' || !d.mage, '炎魔之心仅限非法师。');
  ensure(
    card.kind !== 'u28' || !u.chargedOnDeploy || u.deployedAt !== s.ply,
    '冲锋随从部署当回合不能装备炎魔之心。',
  );
  for (const old of u.equipment) {
    if (old === 'u11') u.maxHp -= 25;
    if (old === 'u28') u.maxHp -= 5;
  }
  u.hp = Math.min(u.hp, u.maxHp);
  u.equipment = [card.kind];
  if (card.kind === 'u11') u.maxHp += 25;
  if (card.kind === 'u28') {
    u.maxHp += 5;
    u.hp += 5;
    u.effects = u.effects.filter((e) => e.type !== 'freeze');
  }
  s.hands[s.active] = s.hands[s.active].filter((v) => v.id !== card.id);
  emit(s, {
    type: 'shield',
    to: u,
    owner: u.owner,
    text: `装备 · ${definition(card.kind).name}`,
    ultimate: true,
  });
}
export function craft(s: GameState, c: Command) {
  ensure(
    c.cardIds?.length === 3 && new Set(c.cardIds).size === 3,
    '请选择3张不同的未装备炎魔之心。',
  );
  ensure(
    c.cardIds.every((id) => s.hands[s.active].some((v) => v.id === id && v.kind === 'u28')),
    '合成仅接受手牌中的炎魔之心。',
  );
  s.hands[s.active] = s.hands[s.active].filter((v) => !c.cardIds!.includes(v.id));
  s.hands[s.active].push({
    id: `c${s.serial++}`,
    kind: 'firelord',
    drawnAt: s.turns[s.active],
    summonedPly: s.ply,
  });
  emit(
    s,
    { type: 'summon', owner: s.active, text: '炎魔之王', ultimate: true },
    '3张炎魔之心合成为炎魔之王，需当回合部署',
  );
}
export function reroll(s: GameState, c: Command) {
  const card = s.hands[s.active].find((v) => v.id === c.cardId);
  ensure(card && card.summonedPly === s.ply, '改判仅限本回合刚召唤的牌。');
  if (card.group)
    ensure(
      s.hands[s.active].filter((v) => v.group === card.group).length === 8,
      '已有克隆部署，不可对半批军团改判。',
    );
  const self = card.kind === 'u13' && s.turns[s.active] <= 5 && !card.rerolled && !c.unitId;
  if (!self) {
    const u = findUnit(s, c.unitId);
    ensure(
      u.kind === 'u13' && u.owner === s.active && passive(s, u) && u.freeUsed !== now(s, u),
      '需要一名本回合尚未改判的友方改判小法师。',
    );
    u.freeUsed = now(s, u);
  }
  s.hands[s.active] = s.hands[s.active].filter((v) =>
    card.group ? v.group !== card.group : v.id !== card.id,
  );
  const cards = draw(s, s.active, 1, definition(card.kind).tier !== 'normal' && card.kind !== '3p');
  for (const v of cards) v.rerolled = true;
  emit(
    s,
    { type: 'skill', owner: s.active, text: '改判' },
    `${definition(card.kind).name}重新召唤`,
  );
}
