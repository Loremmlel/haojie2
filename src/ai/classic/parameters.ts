import { classicActions, classicPosition, createClassicCommandInspector } from './options';
import type { ClassicAction } from './options';
import type { SelectionStep } from '../../engine/commands/options';
import {
  ALL_CELLS,
  targets,
  cells,
  neighbors,
  validAttackRoute,
  expansionAnchors,
} from '../../engine/core/geometry';
import { getStats, allegiance, piercing } from '../../engine/core/state';
import { SYNTHESIS_RECIPES, synthesisDestinations } from '../../engine/setup/synthesis';
import { decisionOwner } from '../observation';
import type { Observation } from '../types';
import type { Command } from '../../engine/types';

export type Step = SelectionStep | { kind: 'material' | 'chosen' };
/** 参数域共用于分层解码和差分参考，不包含神经网络评分。 */
export function commandDomain(observation: Observation) {
  if (observation.mode) throw new Error('训练只支持普通局');
  const s = classicPosition(observation),
    actor = decisionOwner(observation);
  const inspect = createClassicCommandInspector(observation, actor);
  const actions = classicActions(s, actor, inspect);
  function expand(
    command: Command,
    action: ClassicAction,
    step: Step,
    last: boolean,
  ): [Command, boolean][] {
    if (step.kind === 'point' && last) {
      const points = inspect.points(command);
      if (points) return points.map(({ point }) => [{ ...command, ...point }, false]);
    }
    const values: [Command, boolean][] = [];
    const add = (next: Command, repeat = false) => values.push([next, repeat]);
    switch (step.kind) {
      case 'chosen':
        add(command);
        for (const chosenKind of action.chosenKinds ?? []) add({ ...command, chosenKind });
        break;
      case 'material': {
        const selected = command.materialIds ?? [];
        const after = selected.length ? action.materialIds!.indexOf(selected.at(-1)!) + 1 : 0;
        for (const id of action.materialIds!.slice(after))
          add({ ...command, materialIds: [...selected, id] });
        break;
      }
      case 'target':
        for (const t of targets(s)) {
          if (step.unitOnly && !t.unit) continue;
          const side = t.unit ? allegiance(s, t.unit) : t.owner;
          if (
            (step.relation === 'friend' && side !== actor) ||
            (step.relation === 'enemy' && side === actor)
          )
            continue;
          const field = step.field ?? 'targetId';
          if (field === 'sacrificeIds') {
            if (!command.sacrificeIds?.includes(t.id))
              add({ ...command, sacrificeIds: [...(command.sacrificeIds ?? []), t.id] });
          } else add({ ...command, [field]: t.id });
        }
        break;
      case 'point': {
        const target = s.units.find((u) => u.id === command.targetId);
        const points =
          command.type === 'synthesize'
            ? synthesisDestinations(
                s,
                SYNTHESIS_RECIPES.find((r) => r.id === command.recipeId)!,
                command.materialIds ?? [],
              )
            : action.id === 'giant' && target
              ? expansionAnchors(s, target)
              : ALL_CELLS;
        for (const point of points) add({ ...command, ...point });
        break;
      }
      case 'death':
        for (const record of s.deaths) add({ ...command, deathId: record.id });
        break;
      case 'row':
      case 'column':
        for (let value = 1; value <= (step.kind === 'row' ? 13 : 9); value++)
          add({ ...command, [step.kind]: value });
        break;
      case 'direction': {
        add(command);
        const unit = s.units.find((u) => u.id === command.unitId),
          target = targets(s).find((t) => t.id === command.targetId);
        if (unit && target)
          for (const direction of inspect.attacks.directions(
            unit,
            target,
            getStats(s, unit).range,
            piercing(unit),
          ))
            add({ ...command, direction });
        break;
      }
      case 'path': {
        const unit = s.units.find((u) => u.id === command.unitId);
        if (!unit) break;
        const path = command.path ?? [];
        if (path.length) add(command);
        for (const point of path.length ? neighbors(path.at(-1)!) : cells(unit)) {
          const route = [...path, point];
          if (validAttackRoute(s, unit, route, getStats(s, unit).range))
            add({ ...command, path: route }, true);
        }
        break;
      }
    }
    return last && step.kind !== 'path'
      ? values.filter(([c]) => inspect(c).status !== 'invalid')
      : values;
  }
  return { actions, inspect, expand };
}
