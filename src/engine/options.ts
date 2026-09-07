/** Renderer-independent command descriptions. The UI selects values; the engine validates them. */
import { definition, isStored } from './catalog';
import { commandError } from './game';
import { age, allegiance, getStats, has, now, passive } from './state';
import type { Card, Command, GameState, Unit } from './types';
export interface SelectionStep {
  kind: 'target' | 'point' | 'row' | 'column' | 'death';
  field?: 'targetId' | 'secondId' | 'sacrificeIds';
  label: string;
  relation?: 'friend' | 'enemy' | 'any';
  unitOnly?: boolean;
  range?: boolean;
}
export interface ActionSpec {
  id: string;
  label: string;
  icon: string;
  command: Command;
  steps: SelectionStep[];
  free?: boolean;
  hint?: string;
}
const target = (
  label: string,
  relation: 'friend' | 'enemy' | 'any' = 'enemy',
  field: SelectionStep['field'] = 'targetId',
  range = true,
  unitOnly = true,
): SelectionStep => ({ kind: 'target', label, relation, field, range, unitOnly });
const square = (label = '选择高亮落点'): SelectionStep => ({ kind: 'point', label });
const spec = (
  id: string,
  label: string,
  command: Command,
  steps: SelectionStep[] = [],
  icon = 'spark',
  free = false,
): ActionSpec => ({ id, label, command, steps, icon, free });
export function unitActions(s: GameState, u: Unit): ActionSpec[] {
  const result: ActionSpec[] = [],
    stats = getStats(s, u),
    command = { unitId: u.id };
  if (stats.move > 0)
    result.push(spec('move', '移动', { type: 'move', ...command }, [square()], 'move'));
  if (stats.actions > 0 && !(u.kind === 'firelord' && !u.silenced))
    result.push(
      spec(
        'attack',
        u.kind === 2 || u.kind === 'u21' ? '攻击 / 治疗' : '攻击',
        { type: 'attack', ...command },
        [target('选择高亮目标', 'any', 'targetId', true, false)],
        'sword',
      ),
    );
  if (u.mode === 'move' || u.mode === 'attack')
    result.push(spec('finish', '结束本次操作', { type: 'finish-mode', ...command }, [], 'check'));
  if (stats.move % 1 !== 0)
    result.push(
      spec('charge-move', '蓄力 · 移动', { type: 'charge', ...command, mode: 'move' }, [], 'clock'),
    );
  if (!u.silenced) {
    if (u.kind === 4 || u.kind === 'u2')
      result.push(
        spec(
          'charge-attack',
          '蓄力 · 攻击',
          { type: 'charge', ...command, mode: 'attack' },
          [],
          'clock',
        ),
      );
    if (u.kind === 21 || u.kind === 'u6')
      result.push(
        spec(
          'charge-skill',
          '蓄力 · 技能',
          { type: 'charge', ...command, mode: 'skill' },
          [],
          'clock',
        ),
      );
    const skill = { type: 'skill' as const, ...command };
    switch (u.kind) {
      case 5:
        result.push(spec('stomp', '震地 · 周围20伤', skill));
        break;
      case 6:
        result.push(spec('buff', '鼓舞友军', skill));
        break;
      case 7:
        result.push(
          spec('hook', '牵引', skill, [
            target('牵引 1/2：选择敌方随从'),
            square('牵引 2/2：选择新落点'),
          ]),
        );
        break;
      case 14:
        result.push(
          spec('sacrifice', '献祭射击', skill, [
            target('献祭 1/2：选择另一友方', 'friend'),
            { kind: 'column', label: '献祭 2/2：选择射击列' },
          ]),
        );
        break;
      case 15:
        result.push(
          spec('upgrade-atk', '强化 · +10攻击', { ...skill, mode: 'attack' }),
          spec('upgrade-range', '强化 · +1射程', { ...skill, mode: 'range' }),
        );
        break;
      case 19:
        result.push(spec('wall', '制造路障', skill, [square('选择射程内路障位置')]));
        break;
      case 21:
        result.push(
          spec('dash', '神行突袭', skill, [
            square('突袭 1/2：选择6格内落点'),
            target('突袭 2/2：选择落点射程内目标', 'enemy', 'targetId', true, false),
          ]),
        );
        break;
      case 'u6':
        result.push(
          spec('cross', '十字浩劫 · 一生一次', skill, [square('选择11×11区域内十字交点')]),
        );
        break;
      case 'u7':
        result.push(
          spec(
            'giant',
            '巨大化 · 免费',
            skill,
            [target('选择有空间变为2×2的单格友方', 'friend', 'targetId', false)],
            'spark',
            true,
          ),
        );
        break;
      case 'u14':
        result.push(
          spec(
            'siphon',
            '虹吸 · 免费',
            skill,
            [
              target('虹吸 1/2：选择每回合扣20血的目标', 'any', 'targetId', true, false),
              target('虹吸 2/2：选择每回合回复20血的目标', 'any', 'secondId', true, false),
            ],
            'spark',
            true,
          ),
        );
        break;
      case 'u19':
        result.push(
          spec('revive', '复活', skill, [
            { kind: 'death', label: '复活 1/2：选择阵亡记录' },
            square('复活 2/2：选择射程内落点'),
          ]),
        );
        break;
      case 'u21':
        result.push(
          spec('single-buff', '鼓舞一名友方', skill, [target('选择下回合+10攻击的友方', 'friend')]),
        );
        break;
      case 'u23':
        result.push(
          spec('superhook', '超级牵引', skill, [
            target('选择任意敌方，牵引到身前', 'enemy', 'targetId', false),
          ]),
        );
        break;
      case 'u24':
        result.push(spec('ice-mark', '寒冰标记', skill, [square('选择11×11内标记格，下回合判定')]));
        break;
    }
  }
  return result;
}
export function cardActions(s: GameState, c: Card): ActionSpec[] {
  const d = definition(c.kind),
    base = { cardId: c.id };
  const result: ActionSpec[] = [];
  if (!isStored(d))
    result.push(
      spec(
        'deploy',
        '部署随从',
        { type: 'deploy', ...base, charge: false },
        [square('选择部署格；2×2以该格为左上角')],
        'plus',
      ),
    );
  else if (d.weapon !== undefined)
    result.push(
      spec(
        'equip',
        '装备武器',
        { type: 'equip', ...base },
        [target('选择友方装备；已有武器会被替换', 'friend', 'targetId', false)],
        'sword',
      ),
    );
  else if (c.kind === 8)
    result.push(
      spec('blast', '施放爆弹', { type: 'cast', ...base }, [square('选择2×2爆弹区域的左上格')]),
    );
  else if (c.kind === 25) {
    result.push(spec('reforge-one', '重铸 · 召唤一次', { type: 'cast', ...base, mode: 'single' }));
    result.push(
      spec('reforge-two', '献祭两名 · 召唤两次', { type: 'cast', ...base, mode: 'double' }, [
        target('选择第一名至少半血的友方', 'friend', 'sacrificeIds', false),
        target('选择第二名至少半血的友方', 'friend', 'sacrificeIds', false),
      ]),
    );
  } else if (c.kind === 'u9')
    result.push(
      spec('storm-row', '烈焰风暴 · 横排', { type: 'cast', ...base, mode: 'row' }, [
        { kind: 'row', label: '选择受到两次烈焰风暴的横排' },
      ]),
      spec('storm-col', '烈焰风暴 · 竖排', { type: 'cast', ...base, mode: 'column' }, [
        { kind: 'column', label: '选择受到两次烈焰风暴的竖排' },
      ]),
    );
  else
    result.push(
      spec('cast', `施放${d.name}`, { type: 'cast', ...base }, [
        target('选择法术目标', c.kind === 'u26' ? 'any' : 'friend', 'targetId', false),
      ]),
    );
  if (c.summonedPly === s.ply) {
    if (c.kind === 'u13' && s.turns[s.active] <= 5 && !c.rerolled)
      result.push(spec('self-reroll', '改判自身 · 前5回合', { type: 'reroll', cardId: c.id }));
    for (const mage of s.units)
      if (
        mage.kind === 'u13' &&
        mage.owner === s.active &&
        passive(s, mage) &&
        mage.freeUsed !== now(s, mage)
      )
        result.push(
          spec(`reroll-${mage.id}`, `改判 · (${mage.x},${mage.y})`, {
            type: 'reroll',
            unitId: mage.id,
            cardId: c.id,
          }),
        );
  }
  return result;
}
export function reactionAction(s: GameState): ActionSpec | null {
  const r = s.pending[0];
  if (!r) return null;
  if (r.kind === 'bounce')
    return spec('bounce', 'SZF必须弹出', { type: 'react' }, [
      square('选择上下左右一格免费弹出；连续撞击继续弹出'),
    ]);
  if (r.kind === 'hut-spawn')
    return spec('hut-spawn', '小屋死亡召唤', { type: 'react' }, [
      square('选择小屋射程内位置，召唤普通20'),
    ]);
  return spec(
    r.kind,
    r.kind === 'death-shot' ? '奶妈临终行动' : `伤害转化 · ${r.amount}`,
    { type: 'react' },
    [
      target(
        r.kind === 'death-shot' ? '效果所属玩家选择无距离限制的攻击或治疗目标' : '选择伤害转化目标',
        'any',
        'targetId',
        false,
        false,
      ),
    ],
  );
}
export function actionError(s: GameState, a: ActionSpec): string | null {
  if (!a.steps.length) return commandError(s, a.command);
  if (s.winner) return '对局已经结束。';
  if (a.command.type === 'react') return null;
  if (s.pending.length) return '先处理待结算效果。';
  if (s.phase !== 'play' && a.command.type !== 'reroll') return '请先完成召唤阶段。';
  const u = a.command.unitId ? s.units.find((v) => v.id === a.command.unitId) : undefined;
  if (!u) return null;
  const stats = getStats(s, u);
  if (u.owner !== s.active && a.id !== 'giant') return '不是该随从所属方回合。';
  if (stats.frozen || stats.stunned || (stats.sleeping && a.id !== 'giant'))
    return '正在疲劳、休整、冰冻或眩晕中。';
  if (a.command.type === 'attack') {
    if (!stats.remaining) return '没有可用攻击操作。';
    if (u.kind === 4 && !u.silenced && u.readyCharge < 2) return '回合开始需要2层攻击蓄力。';
    return null;
  }
  if (
    !a.free &&
    ((u.mode !== 'none' && !(a.command.type === 'move' && u.mode === 'move')) ||
      stats.operationsLeft <= 0)
  )
    return '本回合已选其他模式或操作已用完。';
  if (a.command.type === 'move' && stats.move % 1 !== 0 && u.mode === 'none' && u.readyCharge < 1)
    return '须先蓄力，下一回合开始才可移动。';
  if (a.id === 'dash' && (u.readyCharge < 2 || u.hp <= 10))
    return '需2层已就绪的技能蓄力，且生命大于10。';
  if (a.id === 'cross' && (u.readyCharge < 1 || u.onceUsed))
    return '需要就绪的技能蓄力，且未使用过十字浩劫。';
  if (a.free && (u.freeUsed === now(s, u) || (a.id === 'giant' && u.onceUsed)))
    return '免费能力的次数已经用完。';
  if (
    a.id === 'superhook' &&
    !(u.hookReadyAt !== undefined && u.hookReadyAt <= now(s, u) && u.hookExpiresAt! > now(s, u))
  )
    return '超级牵引仅在击杀后的下个己方回合可用。';
  return null;
}
