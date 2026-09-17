import { attackPath } from './geometry';
import { allegiance, asTarget, getStats, passive } from './state';
import type { GameState, Unit } from './types';

/** Older saves cannot recover which source paid the single aggregate protection.
 * Keep all then-existing friendly sources spent; a newly deployed source is still new. */
export function spentGuardSources(s: GameState, u: Unit): readonly string[] {
  return (
    u.guardSourceIds ??
    (u.guardUsed ? s.units.filter((v) => v.kind === 3 && v.owner === u.owner).map((v) => v.id) : [])
  );
}
export function normalizeLegacyGuards(s: GameState) {
  for (const u of s.units) {
    if (u.guardUsed && u.guardSourceIds === undefined)
      u.guardSourceIds = [...spentGuardSources(s, u)];
  }
}
/** Shared by damage, AI and the inspector. Array order breaks equal-deployment ties. */
export function guardProtections(s: GameState, u: Unit) {
  const used = spentGuardSources(s, u);
  return s.units
    .filter(
      (v) =>
        v.kind === 3 &&
        allegiance(s, v) === u.owner &&
        passive(s, v) &&
        attackPath(s, v, asTarget(u), getStats(s, v).range),
    )
    .sort((a, b) => a.deployedAt - b.deployedAt)
    .map((source) => ({ source, used: used.includes(source.id) }));
}
export function availableGuardians(s: GameState, u: Unit): Unit[] {
  return guardProtections(s, u)
    .filter((p) => !p.used)
    .map((p) => p.source);
}
