import type { ActionSpec, Command, GameState, Point } from '../../engine';
import {
  actionError,
  allegiance,
  attackPath,
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
  if (step.kind === 'point') {
    draft.x = p.x;
    draft.y = p.y;
  }
  if (step.kind === 'row') draft.row = p.y;
  if (step.kind === 'column') draft.column = p.x;
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
