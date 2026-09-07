import { definition } from './catalog';
import { allegiance, has, passive } from './state';
import type { GameState, Player, Point, Target, Unit } from './types';
export const WIDTH = 9,
  HEIGHT = 13;
export const ALL_CELLS: Point[] = Array.from({ length: 117 }, (_, i) => ({
  x: (i % 9) + 1,
  y: Math.floor(i / 9) + 1,
}));
export const other = (p: Player): Player => (p === 1 ? 2 : 1);
export const basePoint = (p: Player): Point => ({ x: 5, y: p === 1 ? 1 : 13 });
export const inside = (p: Point) =>
  Number.isInteger(p.x) && Number.isInteger(p.y) && p.x >= 1 && p.x <= 9 && p.y >= 1 && p.y <= 13;
export const equal = (a: Point, b: Point) => a.x === b.x && a.y === b.y;
export const distance = (a: Point, b: Point) => Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
export const key = (p: Point) => `${p.x},${p.y}`;
export function cells(u: { kind: Unit['kind']; x: number; y: number; size?: number }): Point[] {
  const size = u.size ?? definition(u.kind).size ?? 1;
  return Array.from({ length: size * size }, (_, i) => ({
    x: u.x + (i % size),
    y: u.y + Math.floor(i / size),
  }));
}
export const occupants = (s: GameState, p: Point) =>
  s.units.filter((u) => cells(u).some((c) => equal(c, p)));
export const occupant = (s: GameState, p: Point) => occupants(s, p)[0];
export function targets(s: GameState): Target[] {
  return [
    ...s.units.map((u) => ({ id: u.id, owner: u.owner, x: u.x, y: u.y, unit: u })),
    ...([1, 2] as Player[]).map((p) => ({ id: `base-${p}`, owner: p, ...basePoint(p) })),
  ];
}
export function targetAt(s: GameState, p: Point): Target | undefined {
  return targets(s).find((t) => (t.unit ? cells(t.unit).some((c) => equal(c, p)) : equal(t, p)));
}
export function topTarget(s: GameState, t: Target) {
  if (!t.unit) return true;
  return occupant(s, t)?.id === t.id;
}
export function adjacent(a: Unit, b: Unit) {
  return cells(a).some((c) => cells(b).some((d) => distance(c, d) === 1));
}
export function ring(a: Unit, b: Point | Unit) {
  const bc = 'kind' in b ? cells(b) : [b];
  return (
    cells(a).some((c) =>
      bc.some((d) => Math.max(Math.abs(c.x - d.x), Math.abs(c.y - d.y)) === 1),
    ) && !cells(a).some((c) => bc.some((d) => equal(c, d)))
  );
}
export function canPlace(
  s: GameState,
  u: Unit,
  at: Point,
  deployment = false,
  ignore: string[] = [],
): boolean {
  const moved = { ...u, ...at },
    footprint = cells(moved);
  if (footprint.some((p) => !inside(p) || equal(p, basePoint(1)) || equal(p, basePoint(2))))
    return false;
  if (deployment && u.kind !== 'u27' && footprint.some((p) => !s.deployRows[u.owner].includes(p.y)))
    return false;
  const others = s.units.filter((v) => v.id !== u.id && !ignore.includes(v.id));
  if (
    others.some(
      (v) =>
        cells(v).some((c) => footprint.some((p) => equal(c, p))) &&
        !(
          u.kind === 'u25' &&
          v.kind === 'u25' &&
          u.owner === v.owner &&
          u.size === 1 &&
          v.size === 1 &&
          !has(s, v, 'freeze')
        ),
    )
  )
    return false;
  if (
    others.some(
      (v) =>
        allegiance(s, v) === u.owner &&
        ((v.kind === 23 && passive(s, v)) || (u.kind === 23 && passive(s, u))) &&
        cells(v).some((c) =>
          footprint.some((p) => Math.max(Math.abs(c.x - p.x), Math.abs(c.y - p.y)) <= 1),
        ),
    )
  )
    return false;
  return true;
}
export const neighbors = (p: Point): Point[] =>
  [
    { x: p.x, y: p.y + 1 },
    { x: p.x, y: p.y - 1 },
    { x: p.x + 1, y: p.y },
    { x: p.x - 1, y: p.y },
  ].filter(inside);
export function movementPath(
  s: GameState,
  u: Unit,
  to: Point,
  limit: number,
  straight = false,
): Point[] | null {
  if (!inside(to) || equal(u, to) || !canPlace(s, u, to)) return null;
  if (straight) {
    if (distance(u, to) !== 3 || (u.x !== to.x && u.y !== to.y)) return null;
    const path = Array.from({ length: 4 }, (_, i) => ({
      x: u.x + Math.sign(to.x - u.x) * i,
      y: u.y + Math.sign(to.y - u.y) * i,
    }));
    return path.slice(1).every((p) => canPlace(s, u, p)) ? path : null;
  }
  const queue: Point[][] = [[{ x: u.x, y: u.y }]],
    seen = new Set([key(u)]);
  for (let i = 0; i < queue.length; i++) {
    const path = queue[i];
    if (path.length - 1 >= limit) continue;
    for (const p of neighbors(path.at(-1)!)) {
      if (seen.has(key(p)) || !canPlace(s, u, p)) continue;
      const next = [...path, p];
      if (equal(p, to)) return next;
      seen.add(key(p));
      queue.push(next);
    }
  }
  return null;
}
export function attackPath(
  s: GameState,
  u: Unit,
  target: Target | Point,
  limit: number,
): Point[] | null {
  const t = 'id' in target ? target : undefined,
    ends = t?.unit ? cells(t.unit) : [target],
    blocked = new Set<string>();
  for (const v of s.units)
    if (v.id !== u.id && v.id !== t?.id && allegiance(s, v) !== u.owner)
      for (const p of cells(v)) blocked.add(key(p));
  // When attacking a stack, other members on its target square are not intervening blockers.
  for (const end of ends) blocked.delete(key(end));
  if (t?.id !== `base-${other(u.owner)}`) blocked.add(key(basePoint(other(u.owner))));
  const queue = cells(u).map((p) => [p]),
    seen = new Set(cells(u).map(key));
  let fallback: Point[] | null = null;
  for (let i = 0; i < queue.length; i++) {
    const path = queue[i],
      depth = path.length - 1;
    if (ends.some((p) => equal(p, path[depth]))) return path;
    if (depth >= limit || (fallback && depth >= fallback.length - 1)) continue;
    for (const p of neighbors(path[depth])) {
      if (blocked.has(key(p))) continue;
      const next = [...path, p];
      if (ends.some((e) => equal(e, p))) {
        fallback ??= next;
        if (t?.unit?.kind !== 24 || t.unit.silenced || frontal(next, t.owner)) return next;
        continue;
      }
      if (!seen.has(key(p))) {
        seen.add(key(p));
        queue.push(next);
      }
    }
  }
  return fallback;
}
export function frontal(path: Point[], owner: Player) {
  if (path.length < 2) return false;
  const a = path.at(-2)!,
    b = path.at(-1)!;
  return owner === 1 ? b.y < a.y : b.y > a.y;
}
export function refreshDeployment(s: GameState, owner: Player) {
  s.deployRows[owner] = [];
  for (let y = 1; y <= 13; y++) {
    const count = (p: Player) =>
      s.units.filter((u) => allegiance(s, u) === p && cells(u).some((c) => c.y === y)).length;
    if ((owner === 1 ? y <= 8 : y >= 6) || count(owner) - count(other(owner)) >= 2)
      s.deployRows[owner].push(y);
  }
}
export function inSquare(u: Unit, p: Point, radius = 5) {
  return inside(p) && Math.abs(p.x - u.x) <= radius && Math.abs(p.y - u.y) <= radius;
}
