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

type Step = SelectionStep | { kind: 'material' | 'chosen' };
/** 与 Rust learning/proposal 同序的公开参数采样；不使用手工评分。 */
export function propose(observation: Observation, seed: number, count = 32): Command[] {
  if (observation.mode) throw new Error('训练只支持普通局');
  if (!Number.isInteger(count) || count < 1 || count > 512) throw new Error('候选数必须为 1..512');
  const s = classicPosition(observation),
    actor = decisionOwner(observation);
  const inspect = createClassicCommandInspector(observation, actor);
  const actions = classicActions(s, actor, inspect);
  let remaining = 20000;
  const random = () => {
    if (!seed) seed = 0x9e3779b9;
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return seed >>> 0;
  };
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
  function complete(
    action: ClassicAction,
    command: Command,
    steps: Step[],
    offset: number,
    depth: number,
  ): Command | undefined {
    if (!remaining || depth > 256) return;
    remaining--;
    if (offset === steps.length) return inspect(command).status !== 'invalid' ? command : undefined;
    const choices = expand(command, action, steps[offset], offset + 1 === steps.length);
    while (choices.length && remaining) {
      const index = random() % choices.length,
        [next, repeat] = choices[index];
      choices[index] = choices[choices.length - 1];
      choices.pop();
      const found = complete(action, next, steps, offset + Number(!repeat), depth + 1);
      if (found) return found;
    }
  }
  const simple = actions.every(
    (a) =>
      !a.steps.length &&
      !a.materialIds?.length &&
      !a.chosenKinds?.length &&
      a.command.type !== 'attack',
  );
  const order = actions.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = random() % (i + 1);
    [order[i], order[j]] = [order[j], order[i]];
  }
  const result: Command[] = [],
    seen = new Set<string>();
  for (let attempt = 0; attempt < Math.max(count * 8, actions.length); attempt++) {
    if (
      (simple && attempt >= actions.length) ||
      result.length >= count ||
      !actions.length ||
      !remaining
    )
      break;
    const action = actions[attempt < order.length ? order[attempt] : random() % actions.length];
    const steps: Step[] = [
      ...(action.materialIds?.length
        ? Array.from({ length: 3 }, () => ({ kind: 'material' as const }))
        : []),
      ...action.steps,
    ];
    if (action.chosenKinds?.length) steps.push({ kind: 'chosen' });
    if (action.command.type === 'attack' && !action.command.path)
      steps.push({ kind: 'direction', label: '' });
    const command = complete(action, action.command, steps, 0, 0);
    if (command) {
      const key = JSON.stringify(
        Object.fromEntries(Object.entries(command).sort(([a], [b]) => a.localeCompare(b))),
      );
      if (!seen.has(key)) {
        seen.add(key);
        result.push(command);
      }
    }
  }
  if (!result.length && !s.winner)
    throw new Error(`没有完整命令，阶段 ${s.phase}，剩余预算 ${remaining}`);
  return result;
}
