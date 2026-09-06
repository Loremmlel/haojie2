import { definition } from './catalog';
import {
  ALL_CELLS,
  attackPath,
  canPlace,
  cells,
  distance,
  equal,
  inside,
  movementPath,
  other,
  targets,
} from './geometry';
import { attack, damage, findTarget, kill, react } from './combat';
import {
  actor,
  addUnit,
  asTarget,
  draw,
  emit,
  ensure,
  faction,
  findUnit,
  getStats,
  point,
  refreshDeployment,
  RuleError,
  template,
} from './state';
import type { Command, Effect, GameState, Kind, Player, Unit } from './types';

function beginTurn(s: GameState): void {
  const p = s.active;
  s.turns[p]++;
  for (const u of [...s.units]) {
    u.effects = u.effects.filter((e) => e.until > s.ply);
    if (u.expiresAt !== undefined && u.expiresAt <= s.ply) kill(s, u);
    if (u.owner === p) {
      u.spent = 0;
      u.attacked = [];
      u.fired = false;
    }
  }
  for (const owner of [1, 2] as Player[])
    s.baseEffects[owner] = s.baseEffects[owner].filter((e) => e.until > s.ply);
  s.hands[p] = s.hands[p].filter((c) => {
    if (c.expiresAt !== undefined && c.expiresAt <= s.turns[p]) {
      emit(s, { type: 'skill', owner: p, text: '过期' }, `${definition(c.kind).name}储存期限已到`);
      return false;
    }
    return true;
  });
  refreshDeployment(s, p);
  emit(
    s,
    { type: 'turn', owner: p, text: `${faction(p)}回合` },
    `${faction(p)}的第${s.turns[p]}回合开始`,
  );
  const count = 2 + s.bonus[p];
  s.bonus[p] = 0;
  draw(s, p, count);
}
export function createGame(seed = 20260906): GameState {
  ensure(Number.isFinite(seed), '种子必须是有限数值。');
  const normalized = Math.trunc(seed) >>> 0 || 2654435769;
  const s: GameState = {
    version: 1,
    seed: normalized,
    rng: normalized,
    serial: 1,
    ply: 1,
    active: 1,
    turns: { 1: 0, 2: 0 },
    bases: { 1: 300, 2: 300 },
    baseEffects: { 1: [], 2: [] },
    hands: { 1: [], 2: [] },
    bonus: { 1: 0, 2: 0 },
    deployRows: { 1: [1, 2, 3, 4, 5, 6, 7, 8], 2: [6, 7, 8, 9, 10, 11, 12, 13] },
    units: [],
    pending: [],
    log: [],
    events: [],
  };
  beginTurn(s);
  return s;
}
function endTurn(s: GameState): void {
  const p = s.active,
    minions = s.hands[p].filter((c) => !definition(c.kind).spell);
  for (const c of minions) {
    const ghost = template(c.kind, p, s.turns[p], { x: 1, y: 1 });
    ensure(
      !ALL_CELLS.some((at) => canPlace(s, ghost, at, true)),
      `请先部署${definition(c.kind).name}，随从不能留到下回合。`,
    );
    emit(
      s,
      { type: 'skill', owner: p, text: '无处部署' },
      `${definition(c.kind).name}没有合法部署位置，自动弃置`,
    );
  }
  s.hands[p] = s.hands[p].filter((c) => !!definition(c.kind).spell);
  for (const u of s.units.filter((u) => u.owner === p && u.kind === 4 && !u.fired))
    u.charge = Math.min(5, u.charge + 1);
  s.active = other(p);
  s.ply++;
  beginTurn(s);
}
function useSkill(s: GameState, c: Extract<Command, { type: 'skill' }>): void {
  const u = actor(s, c.unitId),
    range = getStats(s, u).range;
  switch (u.kind) {
    case 5:
      ensure(u.charge === 0, '已经蓄步，可以选择移动。');
      u.charge = 1;
      break;
    case 6:
      for (const friend of s.units.filter(
        (v) => v.owner === u.owner && v.kind !== 10 && attackPath(s, u, asTarget(v), range),
      )) {
        friend.effects.push({
          type: 'attack',
          amount: 10,
          owner: u.owner,
          from: s.ply + 2,
          until: s.ply + 4,
        });
        emit(s, { type: 'skill', to: friend, owner: u.owner, text: '下回合+10攻' });
      }
      break;
    case 7: {
      const victim = findUnit(s, c.targetId),
        to = point(c.x, c.y);
      ensure(
        victim.owner !== u.owner && attackPath(s, u, asTarget(victim), range),
        '请选择钩子射程内的敌方随从。',
      );
      ensure(!equal(victim, to) && canPlace(s, victim, to), '牵引终点不合法。');
      ensure(
        attackPath(s, u, asTarget({ ...victim, ...to }), range),
        '牵引终点必须仍在钩子射程内。',
      );
      emit(s, {
        type: 'move',
        from: victim,
        to,
        unitId: victim.id,
        owner: victim.owner,
        text: '牵引',
      });
      Object.assign(victim, to);
      break;
    }
    case 14: {
      const victim = findUnit(s, c.targetId);
      ensure(u.maxHp > 10, '献祭炮的生命上限必须大于10。');
      ensure(
        victim.id !== u.id && victim.owner === u.owner && attackPath(s, u, asTarget(victim), range),
        '请选择射程内另一枚友方随从。',
      );
      ensure(Number.isInteger(c.column) && c.column! >= 1 && c.column! <= 9, '请选择要射击的列。');
      const candidates = targets(s).filter(
        (t) =>
          t.owner !== u.owner &&
          (t.unit ? cells(t.unit) : [t]).some(
            (p) => p.x === c.column && (u.owner === 1 ? p.y >= u.y : p.y <= u.y),
          ) &&
          attackPath(s, u, t, range),
      );
      candidates.sort((a, b) => (u.owner === 1 ? a.y - b.y : b.y - a.y));
      const target = candidates[0];
      ensure(target, '这一列的进攻方向上没有射程内可命中的敌方目标。');
      const amount = getStats(s, victim).attack;
      u.maxHp -= 10;
      u.hp = Math.min(u.hp, u.maxHp);
      kill(s, victim, u);
      emit(s, { type: 'attack', from: u, to: target, owner: u.owner, text: '献祭炮' });
      damage(s, target, amount, u, attackPath(s, u, target, range) ?? undefined);
      break;
    }
    case 15:
      ensure(u.upgrades < 3, '已经使用三次强化技能。');
      ensure(c.mode === 'attack' || c.mode === 'range', '请选择增加攻击或射程。');
      if (c.mode === 'attack') u.attackBonus += 10;
      else u.rangeBonus++;
      u.upgrades++;
      break;
    case 19: {
      const to = point(c.x, c.y),
        ghost = template('wall', u.owner, s.turns[u.owner], to);
      ensure(
        canPlace(s, ghost, to) && attackPath(s, u, to, range),
        '请选择射程内可放置路障的空格。',
      );
      const wall = addUnit(s, 'wall', u.owner, to);
      wall.expiresAt = s.ply + 2;
      break;
    }
    case 21:
      if (c.mode === 'dash') {
        ensure(
          u.charge >= 2 && u.lastCharge < s.turns[u.owner],
          '蓄势两回合后，再下一个己方回合才能突袭。',
        );
        ensure(u.hp > 10, '突袭需要扣10血并至少保留1血。');
        const to = point(c.x, c.y);
        ensure(movementPath(s, u, to, 6), '请选择6格以内可抵达的位置。');
        const target = findTarget(s, c.targetId),
          moved = { ...u, ...to };
        ensure(
          target.owner !== u.owner && attackPath(s, moved, target, range),
          '请选择突袭落点射程内的敌方目标。',
        );
        emit(s, { type: 'move', from: u, to, unitId: u.id, owner: u.owner, text: '突袭' });
        u.hp -= 10;
        Object.assign(u, to);
        u.charge = 0;
        attack(s, u, target);
      } else {
        ensure(u.charge < 2, '蓄势已完成，请在后续己方回合发动突袭。');
        ensure(u.lastCharge !== s.turns[u.owner], '同一己方回合只能蓄势一次。');
        u.charge++;
        u.lastCharge = s.turns[u.owner];
      }
      break;
    default:
      throw new RuleError('这枚棋子没有可主动施放的技能。');
  }
  u.spent++;
  emit(
    s,
    { type: 'skill', to: u, unitId: u.id, owner: u.owner, text: definition(u.kind).skill },
    `${definition(u.kind).name}施放技能`,
  );
}
function cast(s: GameState, c: Extract<Command, { type: 'cast' }>): void {
  const p = s.active,
    card = s.hands[p].find((v) => v.id === c.cardId);
  ensure(card && definition(card.kind).spell, '请选择手中的法术。');
  ensure(card.expiresAt! > s.turns[p], '这张法术已经过期。');
  switch (card.kind) {
    case 8: {
      const to = point(c.x, c.y);
      ensure(inside(to) && to.x <= 8 && to.y <= 12, '爆弹需要棋盘内完整的2×2区域。');
      const inArea = (p: { x: number; y: number }) =>
        p.x >= to.x && p.x <= to.x + 1 && p.y >= to.y && p.y <= to.y + 1;
      const victims = targets(s).filter((t) => (t.unit ? cells(t.unit) : [t]).some(inArea));
      emit(s, { type: 'skill', to: { x: to.x + 0.5, y: to.y + 0.5 }, owner: p, text: '爆弹' });
      for (const target of victims) damage(s, target, 20);
      break;
    }
    case 17:
    case 18:
    case 22: {
      const target = findUnit(s, c.targetId);
      ensure(target.owner === p, '法术只能施放于友方随从。');
      const type: Effect['type'] =
        card.kind === 17 ? 'immune' : card.kind === 18 ? 'execute' : 'convert';
      const from = s.ply + (card.kind === 17 ? 0 : 2),
        until = s.ply + (card.kind === 17 ? 2 : 4);
      target.effects = target.effects.filter((e) => !(e.type === type && e.from === from));
      target.effects.push({ type, owner: p, from, until });
      emit(s, {
        type: 'shield',
        to: target,
        owner: p,
        unitId: target.id,
        text: definition(card.kind).name,
      });
      break;
    }
    case 25:
      if (c.mode === 'double') {
        ensure(
          c.sacrificeIds?.length === 2 && new Set(c.sacrificeIds).size === 2,
          '请选择两个不同的友方随从。',
        );
        const victims = c.sacrificeIds.map((id) => findUnit(s, id));
        ensure(
          victims.every((u) => u.owner === p && u.hp * 2 >= u.maxHp),
          '重铸目标必须是至少半血的友方随从。',
        );
        for (const victim of victims) kill(s, victim);
        draw(s, p, 2);
      } else draw(s, p, 1);
      break;
    default:
      throw new RuleError('这张牌不是可施放法术。');
  }
  s.hands[p] = s.hands[p].filter((v) => v.id !== card.id);
  emit(
    s,
    { type: 'skill', owner: p, text: definition(card.kind).name },
    `${faction(p)}施放${definition(card.kind).name}`,
  );
}
/** Pure command boundary. Invalid commands never change the supplied state. */
export function applyCommand(previous: GameState, command: Command): GameState {
  ensure(!previous.winner, '对局已经结束；可以悔棋或开始新对局。');
  ensure(
    previous.pending.length === 0 || command.type === 'react',
    '请先由效果所属玩家完成临终行动或伤害转化。',
  );
  const s = structuredClone(previous);
  s.events = [];
  switch (command.type) {
    case 'end':
      endTurn(s);
      break;
    case 'react':
      react(s, command.targetId);
      break;
    case 'deploy': {
      const c = s.hands[s.active].find((v) => v.id === command.cardId);
      ensure(c && !definition(c.kind).spell, '请选择本回合召唤的随从。');
      const to = point(command.x, command.y),
        ghost = template(c.kind, s.active, s.turns[s.active], to);
      ensure(
        canPlace(s, ghost, to, true),
        '此处无法部署：请检查部署行、占位、基地与独行侠的禁区。',
      );
      const u = addUnit(s, c.kind, s.active, to);
      if (c.kind === 1 && command.charge) {
        u.hp -= 10;
        u.maxHp -= 10;
        u.born--;
      }
      s.hands[s.active] = s.hands[s.active].filter((v) => v.id !== c.id);
      break;
    }
    case 'move': {
      const u = actor(s, command.unitId),
        to = point(command.x, command.y);
      ensure(
        movementPath(s, u, to, getStats(s, u).move, u.kind === 13),
        '无法移动至该位置：距离、路径、占位或独行侠禁区不符合要求。',
      );
      emit(
        s,
        { type: 'move', from: u, to, unitId: u.id, owner: u.owner },
        `${definition(u.kind).name}移动至(${to.x},${to.y})`,
      );
      Object.assign(u, to);
      u.spent++;
      if (u.kind === 5) u.charge = 0;
      break;
    }
    case 'attack': {
      const u = actor(s, command.unitId);
      attack(s, u, findTarget(s, command.targetId));
      u.spent++;
      break;
    }
    case 'skill':
      useSkill(s, command);
      break;
    case 'cast':
      cast(s, command);
      break;
    default:
      throw new RuleError('无法识别的游戏指令。');
  }
  if (s.bases[1] <= 0 || s.bases[2] <= 0) {
    s.winner = s.bases[1] <= 0 && s.bases[2] <= 0 ? 'draw' : s.bases[1] <= 0 ? 2 : 1;
    s.pending = [];
    emit(
      s,
      { type: 'turn', text: '对局结束' },
      s.winner === 'draw' ? '双方基地同时失守，平局' : `${faction(s.winner)}获胜`,
    );
  }
  return s;
}
export function commandError(s: GameState, c: Command): string | null {
  try {
    applyCommand(s, c);
    return null;
  } catch (error) {
    if (error instanceof RuleError) return error.message;
    throw error;
  }
}
export const isLegal = (s: GameState, c: Command) => commandError(s, c) === null;

/** A separate, explicitly labelled playable demonstration; normal matches start empty. */
export function createDemoGame(): GameState {
  const s = createGame(424242);
  s.units = [];
  s.hands = { 1: [], 2: [] };
  s.turns = { 1: 3, 2: 2 };
  s.ply = 5;
  s.bases = { 1: 268, 2: 234 };
  s.log = [];
  s.events = [];
  const setup: [Kind, Player, number, number][] = [
    [9, 1, 3, 5],
    [26, 1, 6, 6],
    [2, 1, 2, 4],
    [4, 1, 7, 3],
    [24, 2, 6, 8],
    [20, 2, 3, 7],
    [7, 2, 7, 10],
    [5, 2, 2, 10],
  ];
  for (const [kind, owner, x, y] of setup) {
    const u = addUnit(s, kind, owner, { x, y });
    u.born = 0;
    if (kind === 4) u.charge = 3;
  }
  for (const kind of [1, 8, 17, 25]) {
    const d = definition(kind);
    s.hands[1].push({
      id: `demo-card-${kind}`,
      kind,
      drawnAt: 3,
      ...(d.spell ? { expiresAt: 3 + d.spell } : {}),
    });
  }
  s.events = [];
  s.log = ['5 · 演示棋局：选中杀手，尝试攻击超级跑得快，再试试悔棋。'];
  refreshDeployment(s, 1);
  return s;
}
