/** Developer-only scenarios execute real commands; not imported by the production bundle. */
import { applyCommand, createSession, type Command, type GameState } from '../../src/engine';
import { addEffect } from '../../src/engine/state';
import { createGame, definition, type Kind, type Player } from '../../src/engine';
import { template, random } from '../../src/engine/state';
function fixture(): GameState {
  const s = createGame(19);
  s.phase = 'play';
  s.summonSlots = 0;
  s.ply = 5;
  s.turns = { 1: 3, 2: 2 };
  s.events = [];
  return s;
}
function add(s: GameState, kind: Kind, owner: Player, x: number, y: number) {
  const u = template(kind, owner, 0, { x, y }, `test${s.serial++}`);
  s.units.push(u);
  return u;
}
function card(s: GameState, kind: Kind) {
  const id = `card${s.serial++}`,
    limit = definition(kind).spell ?? definition(kind).weapon;
  s.hands[s.active].push({
    id,
    kind,
    drawnAt: s.turns[s.active],
    summonedPly: s.ply,
    ...(limit !== undefined ? { expiresAt: s.turns[s.active] + limit } : {}),
  });
  return id;
}
function seedFor(min: number, max: number) {
  for (let seed = 1; seed < 1e6; seed++) {
    const s = fixture();
    s.rng = seed;
    const value = random(s);
    if (value >= min && value < max) return seed;
  }
  throw new Error('No seed found');
}

export interface Scenario {
  name: string;
  detail: string;
  before: GameState;
  after: GameState;
  command: Command;
}
export function vfxScenarios(): Scenario[] {
  const rows: Scenario[] = [];
  const push = (name: string, detail: string, s: GameState, command: Command) =>
    rows.push({ name, detail, before: s, after: applyCommand(s, command), command });
  let s = fixture();
  let a = add(s, 26, 1, 4, 6),
    b = add(s, 13, 2, 5, 6);
  push('近身斩击', '相邻接触、探身与短刀光', s, { type: 'attack', unitId: a.id, targetId: b.id });
  s = fixture();
  a = add(s, 9, 1, 3, 5);
  b = add(s, 5, 2, 3, 8);
  push('射手箭矢', '遵循引擎路径；大体型中心命中反馈', s, {
    type: 'attack',
    unitId: a.id,
    targetId: b.id,
  });
  s = fixture();
  a = add(s, 4, 2, 6, 9);
  b = add(s, 5, 1, 5, 6);
  s.active = 2;
  a.readyCharge = a.charge = 2;
  a.chargeType = 'attack';
  push('赤焰重炮', '反向出手、聚能与厚重断环', s, { type: 'attack', unitId: a.id, targetId: b.id });
  s = fixture();
  a = add(s, 10, 1, 3, 5);
  b = add(s, 13, 2, 5, 8);
  b.hp = 20;
  push('投石挂标', '投射后挂印，不伪造伤害数字', s, {
    type: 'attack',
    unitId: a.id,
    targetId: b.id,
  });
  s = fixture();
  a = add(s, 7, 1, 3, 5);
  b = add(s, 26, 2, 5, 5);
  push('钩子牵引', '旧位置接钩，合法新位置落点', s, {
    type: 'skill',
    unitId: a.id,
    targetId: b.id,
    x: 4,
    y: 6,
  });
  s = fixture();
  a = add(s, 2, 2, 5, 8);
  b = add(s, 1, 2, 5, 6);
  b.hp = 10;
  s.active = 2;
  push('治疗流光', '正向回复，与攻击箭矢不同', s, { type: 'attack', unitId: a.id, targetId: b.id });
  s = fixture();
  add(s, 13, 1, 4, 6);
  add(s, 26, 2, 5, 6);
  add(s, 5, 2, 5, 7);
  push('爆弹区域', '准确2×2；大体型只结算一次', s, {
    type: 'cast',
    cardId: card(s, 8),
    x: 4,
    y: 6,
  });
  s = fixture();
  a = add(s, 26, 1, 4, 6);
  push('策反施加', '只是待触发印记，不是敌人已变阵营', s, {
    type: 'cast',
    cardId: card(s, 22),
    targetId: a.id,
  });
  s = fixture();
  a = add(s, 26, 1, 4, 6);
  b = add(s, 13, 2, 5, 6);
  addEffect(s, a, 'convert', 1, 0, 2);
  push('策反兑现', '攻击造成伤害后真实转化', s, { type: 'attack', unitId: a.id, targetId: b.id });
  s = fixture();
  a = add(s, 26, 1, 4, 6);
  push('处决施加', '附加印记，而非当场击杀', s, {
    type: 'cast',
    cardId: card(s, 18),
    targetId: a.id,
  });
  s = fixture();
  a = add(s, 26, 1, 4, 6);
  b = add(s, 5, 2, 5, 6);
  addEffect(s, a, 'execute', 1, 0, 2);
  push('处决触发', '已死亡单位由事件快照短暂表现', s, {
    type: 'attack',
    unitId: a.id,
    targetId: b.id,
  });
  s = fixture();
  a = add(s, 9, 1, 3, 5);
  b = add(s, 13, 2, 3, 7);
  addEffect(s, b, 'immune', 2, 0, 2);
  push('金身挡下', '防护面响应；没有成功伤害数字', s, {
    type: 'attack',
    unitId: a.id,
    targetId: b.id,
  });
  s = fixture();
  add(s, 13, 2, 4, 7);
  add(s, 26, 1, 7, 7);
  push('烈焰风暴', '整行扫过；持续危险区由规则状态保留', s, {
    type: 'cast',
    cardId: card(s, 'u9'),
    mode: 'row',
    row: 7,
  });
  s = fixture();
  add(s, 13, 2, 4, 7);
  s.hazards.push({ id: 'due-storm', owner: 2, axis: 'column', line: 4, due: 6 });
  push('风暴再临', '到期的真实二次结算，不依赖施法动画', s, { type: 'end' });
  s = fixture();
  a = add(s, 'u6', 1, 5, 5);
  a.readyCharge = a.charge = 1;
  a.chargeType = 'skill';
  add(s, 5, 2, 5, 7);
  add(s, 26, 2, 7, 7);
  push('十字浩劫', '真实十字格子，交点重叠不重复结算', s, {
    type: 'skill',
    unitId: a.id,
    x: 5,
    y: 7,
  });
  s = fixture();
  a = add(s, 'u6', 1, 4, 5);
  b = add(s, 26, 2, 4, 7);
  a.equipment = ['u5'];
  push('寒冰结晶', '仅成功冰冻时凝结', s, { type: 'attack', unitId: a.id, targetId: b.id });
  s = fixture();
  a = add(s, 'u8', 2, 5, 8);
  a.hp = 30;
  b = add(s, 5, 1, 4, 6);
  s.active = 2;
  push('吸血回流', '只按实际回复发出反向回流', s, { type: 'attack', unitId: a.id, targetId: b.id });
  s = fixture();
  a = add(s, 4, 1, 4, 6);
  push('蓄力收束', '内收箭头与蓄力符印', s, { type: 'charge', unitId: a.id, mode: 'attack' });
  s = fixture();
  a = add(s, 'u12', 1, 3, 5);
  a.readyCharge = a.charge = 1;
  a.chargeType = 'move';
  add(s, 13, 2, 3, 6);
  push('冲撞残影', '真实冲撞动作；反应选择仍由规则控制', s, {
    type: 'move',
    unitId: a.id,
    x: 3,
    y: 6,
  });
  s = fixture();
  add(s, 'u3', 2, 7, 8);
  s.rng = seedFor(0, 1 / 3);
  push('法术反制', '不能继续播放成功的爆弹区域', s, {
    type: 'cast',
    cardId: card(s, 8),
    x: 3,
    y: 5,
  });
  s = fixture();
  push('单位入场', '收拢边框；不延迟实际部署', s, {
    type: 'deploy',
    cardId: card(s, 26),
    x: 4,
    y: 6,
  });
  s = fixture();
  a = add(s, 'u14', 1, 3, 5);
  b = add(s, 26, 2, 5, 5);
  const c = add(s, 13, 1, 3, 7);
  c.hp = 5;
  s.siphons.push({ id: 'link', owner: 1, sourceId: a.id, fromId: b.id, toId: c.id });
  push('虹吸结算', '连接有方向；真实吸取与回复分别呈现', s, { type: 'end' });
  return rows;
}
export const savedScenario = (s: Scenario) => JSON.stringify(createSession(s.before));
