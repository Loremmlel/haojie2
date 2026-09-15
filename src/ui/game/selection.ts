import type { ActionSpec, Command, GameState, Point } from '../../engine';
import {
  actionError,
  allegiance,
  attackPath,
  selectableAttackRoutes,
  targets,
  getStats,
  isLegal,
  movementPath,
  targetAt,
} from '../../engine';
export type Intent =
  | { kind: 'none' }
  | { kind: 'select'; action: ActionSpec; draft: Command; index: number };
export const startIntent = (a: ActionSpec): Intent => ({
  kind: 'select',
  action: a,
  draft: { ...a.command },
  index: 0,
});
export function advanceIntent(i: Intent, p: Point, s: GameState): Intent {
  if (i.kind === 'none') return i;
  const step = i.action.steps[i.index],
    draft = { ...i.draft };
  if (step.kind === 'target') {
    const t = targetAt(s, p);
    if (!t) return i;
    if (step.field === 'sacrificeIds') draft.sacrificeIds = [...(draft.sacrificeIds ?? []), t.id];
    else draft[step.field ?? 'targetId'] = t.id;
  }
  if (step.kind === 'direction') {
    const route = intentRoutes(s, i).find((r) => {
      const before = r.path.at(-2)!;
      return before.x === p.x && before.y === p.y;
    });
    if (!route) return i;
    draft.direction = route.direction;
  }
  if (step.kind === 'point') {
    draft.x = p.x;
    draft.y = p.y;
  }
  if (step.kind === 'row') draft.row = p.y;
  if (step.kind === 'column') draft.column = p.x;
  if (step.kind === 'target' && draft.type === 'attack') {
    const u = s.units.find((u) => u.id === draft.unitId),
      t = targetAt(s, p);
    if (u && t && isLegal(s, draft) && selectableAttackRoutes(s, u, t).length > 1)
      return {
        ...i,
        draft,
        index: i.index + 1,
        action: {
          ...i.action,
          steps: [
            ...i.action.steps,
            { kind: 'direction', label: '选择命中方向：点击目标旁高亮格；箭头显示最后一步方向' },
          ],
        },
      };
  }
  return { ...i, draft, index: i.index + 1 };
}
export function commandFor(s: GameState, i: Intent, p: Point): Command | null {
  if (i.kind === 'none') return null;
  const next = advanceIntent(i, p, s);
  if (next.kind === 'select' && next.index >= next.action.steps.length) return next.draft;
  return null;
}
export function canChoose(s: GameState, i: Intent, p: Point): boolean {
  if (i.kind === 'none' || actionError(s, i.action)) return false;
  const step = i.action.steps[i.index];
  if (!step || step.kind === 'death') return false;
  if (step.kind === 'direction') {
    const c = commandFor(s, i, p);
    return !!c && isLegal(s, c);
  }
  const complete = commandFor(s, i, p);
  if (complete) return isLegal(s, complete);
  const u = s.units.find((u) => u.id === i.draft.unitId),
    t = targetAt(s, p);
  if (step.kind === 'point') {
    if (i.action.id === 'dash' && u) return !!movementPath(s, u, p, 6);
    return true;
  }
  if (step.kind === 'target') {
    if (!t || (step.unitOnly && !t.unit)) return false;
    const owner = u?.owner ?? s.active,
      side = t.unit ? allegiance(s, t.unit) : t.owner;
    if (
      (step.relation === 'friend' && side !== owner) ||
      (step.relation === 'enemy' && side === owner)
    )
      return false;
    if (
      step.field === 'sacrificeIds' &&
      (t.unit?.kind === 'u25' ||
        !t.unit ||
        t.unit.hp * 2 < t.unit.maxHp ||
        i.draft.sacrificeIds?.includes(t.id))
    )
      return false;
    if (i.draft.type === 'attack' && !isLegal(s, { ...i.draft, targetId: t.id })) return false;
    if (i.action.id === 'sacrifice' && (t.id === u?.id || t.unit?.kind === 'u25')) return false;
    if (step.range && u && !attackPath(s, u, t, getStats(s, u).range)) return false;
    return true;
  }
  return false;
}
export function instruction(s: GameState, i: Intent): string {
  if (i.kind === 'select') return i.action.steps[i.index]?.label ?? '选择目标';
  if (s.phase === 'summon') return '先决定普通或终极召唤，再查看结果；2人头可替换一次召唤。';
  return '选择随从或手牌。每个随从每回合只能选择一种操作模式；选攻击后可连击。';
}
export function intentTone(i: Intent): string {
  if (i.kind === 'none') return 'move';
  const t = i.draft.type;
  return t === 'attack' || i.action.id === 'superhook'
    ? 'attack'
    : t === 'cast' || t === 'skill' || t === 'equip'
      ? 'magic'
      : 'move';
}

export const directionLabel = {
  up: '向上命中',
  down: '向下命中',
  left: '向左命中',
  right: '向右命中',
} as const;
export const directionArrow = { up: '↑', down: '↓', left: '←', right: '→' } as const;
export function intentRoutes(s: GameState, i: Intent) {
  if (i.kind !== 'select' || i.action.steps[i.index]?.kind !== 'direction') return [];
  const u = s.units.find((u) => u.id === i.draft.unitId),
    t = targets(s).find((t) => t.id === i.draft.targetId);
  return u && t ? selectableAttackRoutes(s, u, t) : [];
}
