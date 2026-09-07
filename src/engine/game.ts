import { withRandomSource, type RandomSource } from './random';
import { definition, isStored } from './catalog';
import { chargeAction, craft, equip, reroll, useSkill, cast } from './abilities';
import { findTarget, performAttack, pruneSiphons, resolution } from './combat';
import { canPlace, refreshDeployment, cells, equal } from './geometry';
import { beginTurn, endTurn, switchTurn } from './lifecycle';
import { finishMode, moveUnit, react } from './movement';
import {
  actor,
  addUnit,
  chooseMode,
  draw,
  emit,
  ensure,
  faction,
  finishOperation,
  getStats,
  point,
  RuleError,
  template,
} from './state';
import type { Command, GameState, Kind, Player } from './types';
export function createGame(seed = 20260907): GameState {
  ensure(Number.isSafeInteger(seed), '种子须为整数。');
  const normalized = seed >>> 0 || 2654435769;
  const s: GameState = {
    version: 2,
    seed: normalized,
    rng: normalized,
    serial: 1,
    ply: 1,
    active: 1,
    phase: 'summon',
    summonSlots: 2,
    turns: { 1: 0, 2: 0 },
    bases: { 1: 300, 2: 300 },
    baseEffects: { 1: [], 2: [] },
    heads: { 1: 0, 2: 0 },
    hands: { 1: [], 2: [] },
    bonus: { 1: 0, 2: 0 },
    deployRows: { 1: [1, 2, 3, 4, 5, 6, 7, 8], 2: [6, 7, 8, 9, 10, 11, 12, 13] },
    units: [],
    pending: [],
    deaths: [],
    hazards: [],
    siphons: [],
    iceMarks: [],
    log: [],
    events: [],
  };
  beginTurn(s, resolution());
  return s;
}
export function applyCommand(
  previous: GameState,
  c: Command,
  randomSource?: RandomSource,
): GameState {
  ensure(previous.version === 2, '此命令只接受浩劫2.0局面；旧版对局不会被静默迁移。');
  ensure(!previous.winner, '对局已经结束，可悔棋或开新局。');
  ensure(!previous.pending.length || c.type === 'react', '请先处理当前待结算效果。');
  ensure(previous.summonSlots !== -1 || c.type === 'react', '当前正在结算回合结束效果。');
  const transit = previous.units.find(
    (u) =>
      u.kind === 'u12p' &&
      u.mode === 'move' &&
      previous.units.some(
        (v) => v.id !== u.id && cells(v).some((p) => cells(u).some((q) => equal(p, q))),
      ),
  );
  ensure(
    !transit || c.type === 'react' || (c.type === 'move' && c.unitId === transit.id),
    '小BW正在冲撞经过敌方，必须先用剩余移动次数回到空地。',
  );
  const s = structuredClone(previous);
  s.events = [];
  return withRandomSource(s, randomSource, () => {
    const ctx = resolution();
    if (!['summon', 'begin', 'reroll', 'react'].includes(c.type))
      ensure(s.phase === 'play', '先完成回合开始的召唤选择，再进入行动阶段。');
    switch (c.type) {
      case 'summon':
        ensure(s.phase === 'summon' && s.summonSlots > 0, '本回合开始召唤次数已用完。');
        if (c.ultimate) {
          ensure(s.heads[s.active] >= 2, '终极召唤需要2人头。');
          s.heads[s.active] -= 2;
        }
        draw(s, s.active, 1, !!c.ultimate);
        s.summonSlots--;
        break;
      case 'begin':
        ensure(s.phase === 'summon' && s.summonSlots === 0, '请先完成所有召唤。');
        s.phase = 'play';
        emit(s, { type: 'turn', owner: s.active, text: '行动阶段' });
        break;
      case 'reroll':
        reroll(s, c);
        break;
      case 'end':
        endTurn(s, ctx);
        break;
      case 'react':
        react(s, c, ctx);
        if (!s.pending.length && s.summonSlots === -1 && s.bases[1] > 0 && s.bases[2] > 0)
          switchTurn(s, ctx);
        break;
      case 'deploy': {
        const card = s.hands[s.active].find((v) => v.id === c.cardId);
        ensure(card && !isStored(definition(card.kind)), '请选择待部署随从。');
        const to = point(c.x, c.y),
          ghost = template(card.kind, s.active, s.turns[s.active], to);
        ensure(canPlace(s, ghost, to, true), '非法部署：检查行权限、占位、基地与独行侠禁区。');
        const u = addUnit(s, card.kind, s.active, to, card.group);
        if (card.kind === 1 && c.charge) {
          u.hp -= 10;
          u.maxHp -= 10;
          u.born--;
          u.chargedOnDeploy = true;
        }
        s.hands[s.active] = s.hands[s.active].filter((v) => v.id !== card.id);
        break;
      }
      case 'move':
        moveUnit(s, c, ctx);
        break;
      case 'attack': {
        const u = actor(s, c.unitId);
        chooseMode(s, u, 'attack');
        const t = findTarget(s, c.targetId);
        performAttack(s, u, t, ctx);
        u.attacked.push(t.id);
        u.shots++;
        if (u.shots >= getStats(s, u).actions) finishOperation(u);
        break;
      }
      case 'charge':
        chargeAction(s, c);
        break;
      case 'finish-mode':
        finishMode(s, c);
        break;
      case 'skill':
        useSkill(s, c, ctx);
        break;
      case 'cast':
        cast(s, c, ctx);
        break;
      case 'equip':
        equip(s, c);
        break;
      case 'craft':
        craft(s, c);
        break;
      default:
        throw new RuleError('无法识别的游戏命令。');
    }
    pruneSiphons(s);
    if (s.bases[1] <= 0 || s.bases[2] <= 0) {
      s.winner = s.bases[1] <= 0 && s.bases[2] <= 0 ? 'draw' : s.bases[1] <= 0 ? 2 : 1;
      s.pending = [];
      emit(
        s,
        { type: 'turn', text: '对局结束' },
        s.winner === 'draw' ? '双方基地失守，平局' : `${faction(s.winner)}获胜`,
      );
    }
    return s;
  });
}
export function commandError(s: GameState, c: Command): string | null {
  try {
    applyCommand(s, c);
    return null;
  } catch (e) {
    if (e instanceof RuleError) return e.message;
    throw e;
  }
}
export const isLegal = (s: GameState, c: Command) => commandError(s, c) === null;
export function createDemoGame(): GameState {
  const s = createGame(424242);
  s.phase = 'play';
  s.summonSlots = 0;
  s.ply = 5;
  s.turns = { 1: 3, 2: 2 };
  s.heads = { 1: 6, 2: 4 };
  s.bases = { 1: 268, 2: 234 };
  const setup: [Kind, Player, number, number][] = [
    [9, 1, 3, 5],
    [26, 1, 6, 6],
    [2, 1, 2, 4],
    ['u6', 1, 7, 3],
    [24, 2, 6, 8],
    [20, 2, 3, 7],
    ['u20', 2, 7, 10],
    [5, 2, 2, 10],
    ['u25', 2, 8, 8],
    ['u25', 2, 8, 8],
  ];
  for (const [kind, owner, x, y] of setup) {
    const u = addUnit(s, kind, owner, { x, y }, kind === 'u25' ? 'demo-clones' : undefined);
    u.born = 0;
    if (kind === 'u6') {
      u.charge = u.readyCharge = 1;
      u.chargeType = 'skill';
    }
  }
  for (const kind of [1, 8, 18, 'u5', 'u9', 'u17', 'u28'] as Kind[]) {
    const d = definition(kind),
      limit = d.spell ?? d.weapon;
    s.hands[1].push({
      id: `demo-${kind}`,
      kind,
      drawnAt: 3,
      summonedPly: s.ply,
      ...(limit !== undefined && limit >= 0 ? { expiresAt: 3 + limit } : {}),
    });
  }
  s.log = ['5 · 浩劫2.0演示局：试试大法师装备寒冰法杖、杀手攻击，以及回合末人头兑换终极召唤。'];
  s.events = [];
  refreshDeployment(s, 1);
  return s;
}
