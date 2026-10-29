import {
  actionError,
  cardActions,
  reactionAction,
  unitActions,
  type ActionSpec,
} from '../../engine/commands/options';
import { createCommandInspector, type CommandInspection } from '../../engine/commands/game';
import { actorCommandError } from '../../engine/online/authority';
import { allPieces, hasTrait } from '../../engine/core/traits';
import { ensure, getStats } from '../../engine/core/state';
import { definition } from '../../engine/catalog';
import { availableSyntheses, SYNTHESIS_RECIPES } from '../../engine/setup/synthesis';
import { canChooseSummon, commandSummonPool, selectableSummons } from '../../engine/setup/shrines';
import type { Command, GamePosition, Player } from '../../engine/types';
import type { Observation } from '../types';
import { viewPosition } from '../observation';

/** 仅接受策略观察；即使 JS 绕过 TS 类型也拒绝夹带正式随机状态/历史。查询不修改嵌套对象。 */
export function classicPosition(observation: Observation): GamePosition {
  const position = viewPosition(observation);
  if (position) return position;
  for (const key of ['seed', 'rng', 'log', 'events', 'past', 'future', 'present'])
    ensure(!Object.hasOwn(observation, key), `训练观察禁止携带 ${key}。`);
  return { ...observation, log: [], events: [] };
}

/** 解码器只提交规范命令；预检停在随机边界，同一观察复用规则查询缓存。 */
export function createClassicCommandInspector(
  observation: Observation,
  actor: Player,
  attributes: typeof getStats = getStats,
) {
  const s = classicPosition(observation);
  const inspect = createCommandInspector(s, attributes);
  const query = (command: Command) => {
    const error = actorCommandError(s, actor, command);
    return error
      ? { status: 'invalid' as const, message: error }
      : inspect(command.type === 'choose-shrine' ? { ...command, player: actor } : command);
  };
  return Object.assign(query, {
    attacks: inspect.attacks,
    points(command: Command) {
      if (
        command.type !== 'move' &&
        command.type !== 'deploy' &&
        command.type !== 'skill' &&
        command.type !== 'cast'
      )
        return;
      return actorCommandError(s, actor, command) ? [] : inspect.points(command);
    },
  });
}

export interface ClassicAction extends ActionSpec {
  materialIds?: string[];
  materialCount?: number;
  chosenKinds?: ReturnType<typeof selectableSummons>;
}

/** 动作树只消费动作，不为未展开的参数预先构造完整公开域。 */
export function classicActions(
  s: GamePosition,
  actor: Player,
  inspect: (command: Command) => CommandInspection,
  attributes: typeof getStats = getStats,
) {
  const actions: ClassicAction[] = [];
  const inspectError = (_s: GamePosition, command: Command) => {
    const result = inspect(command);
    return result.status === 'invalid' ? result.message : null;
  };
  const add = (a: ClassicAction) => {
    if (
      actorCommandError(s, actor, a.command) ||
      (!a.materialCount && actionError(s, a, inspectError, attributes))
    )
      return;
    const pool = commandSummonPool(s, a.command);
    actions.push({
      ...a,
      ...(pool && canChooseSummon(s, actor)
        ? { chosenKinds: selectableSummons(pool === 'ultimate') }
        : {}),
    });
  };
  const simple = (command: Command) =>
    add({ id: command.type, label: command.type, icon: '', command, steps: [] });
  if (!s.winner) {
    if (s.pending.length) {
      const action = reactionAction(s);
      if (action) add(action);
      simple({ type: 'react' });
    } else if (s.phase === 'shrine-draft') {
      for (const shrineKind of s.shrineDraft?.offers[actor] ?? [])
        for (const parity of shrineKind === 's9' ? (['odd', 'even'] as const) : [undefined])
          simple({
            type: 'choose-shrine',
            player: actor,
            shrineKind,
            ...(parity ? { parity } : {}),
          });
    } else {
      if (s.summonOffer) {
        for (let i = 0; i < s.summonOffer.groups.length; i++)
          for (let j = i + 1; j < s.summonOffer.groups.length; j++)
            simple({ type: 'choose-summons', offerIndices: [i, j] });
      } else {
        for (const type of ['begin', 'end', 'skip-synthesis', 'finish-shrine-setup'] as const)
          simple({ type });
        for (const type of ['summon', 'extra-summon'] as const)
          for (const ultimate of [false, true]) simple({ type, ultimate });
        for (const piece of allPieces(s)) {
          if (piece.owner !== actor) continue;
          const giant = hasTrait(piece, 'u7');
          if (!giant && (actor !== s.active || s.phase !== 'play')) continue;
          const stats = attributes(s, piece);
          // 只有巨大化免操作预算；改判由手牌动作单独生成，连续操作仍保留结束入口。
          if (!giant && piece.mode === 'none' && stats.operationsLeft <= 0 && stats.remaining <= 0)
            continue;
          for (const a of unitActions(s, piece, (position, unit) =>
            unit === piece ? stats : attributes(position, unit),
          ))
            add(a);
        }
        if (actor === s.active) {
          for (const card of s.hands[actor]) for (const a of cardActions(s, card)) add(a);
          add({
            id: 'clock',
            label: '时钟',
            icon: '',
            command: { type: 'clock' },
            steps: [{ kind: 'target', label: '选择回溯目标', relation: 'any', unitOnly: true }],
          });
          if (s.phase === 'synthesis')
            for (const { recipe, ids } of availableSyntheses(s))
              add({
                id: `synthesize:${recipe.id}`,
                label: definition(recipe.result).name,
                icon: '',
                command: { type: 'synthesize', recipeId: recipe.id },
                materialIds: ids,
                materialCount: 3,
                steps: definition(recipe.result).aura
                  ? []
                  : [{ kind: 'point', label: '选择合成落点' }],
              });
        }
      }
    }
  }
  return actions;
}
