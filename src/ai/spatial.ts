/** Read-only spatial estimates. Final commands always use the engine's exact path/legality checks. */
import { cells, basePoint, inside, distance, neighbors } from '../engine/geometry';
import { allegiance, getStats, resetUnit } from '../engine/state';
import type { GameState, Player, Point, Stats, Target, Unit } from '../engine/types';
const index = (p: Point) => (p.y - 1) * 9 + p.x - 1;
interface Spatial {
  occupants: Unit[][];
  stats: Map<string, Stats>;
  reaches: Map<string, Int16Array>;
  windows: Map<string, GameState>;
}
const cache = new WeakMap<GameState, Spatial>();
function spatial(s: GameState): Spatial {
  let result = cache.get(s);
  if (!result) {
    const occupants = Array.from({ length: 117 }, () => [] as Unit[]);
    for (const u of s.units) for (const p of cells(u)) if (inside(p)) occupants[index(p)].push(u);
    result = { occupants, stats: new Map(), reaches: new Map(), windows: new Map() };
    cache.set(s, result);
  }
  return result;
}
export const occupantsAt = (s: GameState, p: Point) =>
  inside(p) ? spatial(s).occupants[index(p)] : [];
export function statsFor(s: GameState, u: Unit): Stats {
  const data = spatial(s);
  // Preview units may share an id but differ in coordinates/charge. Cache only state members.
  if (!s.units.includes(u)) return getStats(s, u);
  let result = data.stats.get(u.id);
  if (!result) {
    result = getStats(s, u);
    data.stats.set(u.id, result);
  }
  return result;
}
/** Availability-only projection, NOT a simulated turn: no invented draws or extra charges.
 * response = remaining current actions, or the opponent's next turn; next = next full owning turn.
 * Preserve the original objects; no PRNG, global tick or phantom move+attack is consumed. */
export function actionWindow(s: GameState, side: Player, next = false): GameState {
  const delta = s.active !== side ? 1 : next ? 2 : 0;
  if (!delta && s.phase === 'play') return s;
  const key = `${side}:${delta}`;
  const data = spatial(s);
  const old = data.windows.get(key);
  if (old) return old;
  const view: GameState = {
    ...s,
    ply: s.ply + delta,
    active: side,
    phase: 'play',
    pending: [],
    turns: { ...s.turns, [side]: s.turns[side] + (delta ? 1 : 0) },
    units: s.units
      .filter((u) => u.expiresAt === undefined || u.expiresAt > s.ply + delta)
      .map((u) => ({ ...u, effects: [...u.effects] })),
  };
  if (delta) for (const u of view.units) if (u.owner === side) resetUnit(view, u);
  data.windows.set(key, view);
  return view;
}
/** Enemy-occupied cells are terminal: a hit can arrive there but cannot pass through.
 * Equivalent reachability to attackPath for current target footprints, without allocating paths.
 * ignoreId removes a prospective defender's OLD footprint while ranking its alternative positions. */
export function attackField(s: GameState, u: Unit, range: number, ignoreId = ''): Int16Array {
  const data = spatial(s),
    k = `${u.id}:${u.x},${u.y}:${u.size}:${range}:${ignoreId}`;
  const old = data.reaches.get(k);
  if (old) return old;
  const distances = new Int16Array(117).fill(-1);
  const queue: Point[] = [];
  for (const p of cells(u))
    if (inside(p)) {
      distances[index(p)] = 0;
      queue.push(p);
    }
  const enemyBase = basePoint(u.owner === 1 ? 2 : 1);
  for (let i = 0; i < queue.length; i++) {
    const from = queue[i],
      depth = distances[index(from)];
    if (depth >= range) continue;
    for (const p of neighbors(from)) {
      const at = index(p);
      if (distances[at] >= 0) continue;
      distances[at] = depth + 1;
      const blocked =
        (p.x === enemyBase.x && p.y === enemyBase.y) ||
        data.occupants[at].some(
          (v) => v.id !== u.id && v.id !== ignoreId && allegiance(s, v) !== u.owner,
        );
      if (!blocked) queue.push(p);
    }
  }
  data.reaches.set(k, distances);
  return distances;
}
export function hitDistance(s: GameState, u: Unit, t: Target, ignoreId = ''): number {
  const st = statsFor(s, u),
    ends = t.unit ? cells(t.unit) : [t];
  if (u.equipment.includes('u28')) {
    let shortest = Infinity;
    for (const a of cells(u))
      for (const b of ends)
        if ((a.x === b.x || a.y === b.y) && distance(a, b) > 0)
          shortest = Math.min(shortest, distance(a, b));
    return shortest <= st.range ? shortest : Infinity;
  }
  const field = attackField(s, u, st.range, ignoreId);
  return Math.min(
    ...ends.filter(inside).map((p) => (field[index(p)] < 0 ? Infinity : field[index(p)])),
  );
}
export function isFrontHit(s: GameState, u: Unit, t: Target, ignoreId = ''): boolean {
  const d = hitDistance(s, u, t, ignoreId);
  if (!Number.isFinite(d) || d < 1) return false;
  const field = attackField(s, u, statsFor(s, u).range, ignoreId);
  return (t.unit ? cells(t.unit) : [t]).some((p) => {
    const before = { x: p.x, y: p.y + (t.owner === 1 ? 1 : -1) };
    return inside(before) && field[index(before)] === d - 1;
  });
}
