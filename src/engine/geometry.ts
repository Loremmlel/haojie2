import { definition } from './catalog';
import type { GameState, Player, Point, Target, Unit } from './types';
export const WIDTH = 9, HEIGHT = 13;
export const ALL_CELLS: Point[] = Array.from({ length: WIDTH * HEIGHT }, (_, i) => ({ x: i % WIDTH + 1, y: Math.floor(i / WIDTH) + 1 }));
export const other = (p: Player): Player => p === 1 ? 2 : 1;
export const basePoint = (p: Player): Point => ({ x: 5, y: p === 1 ? 1 : 13 });
export const inside = (p: Point) => Number.isInteger(p.x) && Number.isInteger(p.y) && p.x >= 1 && p.x <= WIDTH && p.y >= 1 && p.y <= HEIGHT;
export const equal = (a: Point, b: Point) => a.x === b.x && a.y === b.y;
export const distance = (a: Point, b: Point) => Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
export const key = (p: Point) => `${p.x},${p.y}`;
export function cells(u: Pick<Unit, 'kind' | 'x' | 'y'>): Point[] {
  const size = definition(u.kind).size ?? 1;
  return Array.from({ length: size * size }, (_, i) => ({ x: u.x + i % size, y: u.y + Math.floor(i / size) }));
}
export function occupant(s: GameState, p: Point): Unit | undefined { return s.units.find(u => cells(u).some(c => equal(c, p))); }
export function targets(s: GameState): Target[] {
  return [...s.units.map(u => ({ id: u.id, x: u.x, y: u.y, owner: u.owner, unit: u })), ...([1,2] as Player[]).map(p => ({ id: `base-${p}`, owner: p, ...basePoint(p) }))];
}
export function targetAt(s: GameState, p: Point): Target | undefined { return targets(s).find(t => t.unit ? cells(t.unit).some(c => equal(c,p)) : equal(t,p)); }
export function adjacent(a: Unit, b: Unit): boolean { return cells(a).some(c => cells(b).some(d => distance(c,d) === 1)); }
const near = (a: Unit, b: Unit) => cells(a).some(c => cells(b).some(d => Math.max(Math.abs(c.x-d.x),Math.abs(c.y-d.y)) <= 1));
export function canPlace(s: GameState, u: Unit, at: Point, deployment = false): boolean {
  const moved = { ...u, ...at }, footprint = cells(moved);
  if (footprint.some(p => !inside(p) || equal(p,basePoint(1)) || equal(p,basePoint(2)))) return false;
  if (deployment && footprint.some(p => !s.deployRows[u.owner].includes(p.y))) return false;
  if (s.units.some(v => v.id !== u.id && cells(v).some(c => footprint.some(p => equal(c,p))))) return false;
  if (s.units.some(v => v.id !== u.id && v.owner === u.owner && (v.kind === 23 || u.kind === 23) && near(moved,v))) return false;
  return true;
}
const neighbors = (p: Point): Point[] => [{ x:p.x, y:p.y+1 }, { x:p.x, y:p.y-1 }, { x:p.x+1, y:p.y }, { x:p.x-1, y:p.y }].filter(inside);
export function movementPath(s: GameState, u: Unit, to: Point, limit: number, straight = false): Point[] | null {
  if (!inside(to) || equal(u,to) || !canPlace(s,u,to)) return null;
  if (straight) {
    if (distance(u,to) !== 3 || (u.x !== to.x && u.y !== to.y)) return null;
    const path = Array.from({ length:4 }, (_,i) => ({ x:u.x+Math.sign(to.x-u.x)*i, y:u.y+Math.sign(to.y-u.y)*i }));
    return path.slice(1).every(p => canPlace(s,u,p)) ? path : null;
  }
  const queue: Point[][] = [[{x:u.x,y:u.y}]], visited = new Set([key(u)]);
  for (let i=0; i<queue.length; i++) {
    const path = queue[i];
    if (path.length-1 >= limit) continue;
    for (const p of neighbors(path[path.length-1])) {
      if (visited.has(key(p)) || !canPlace(s,u,p)) continue;
      const next = [...path,p];
      if (equal(p,to)) return next;
      visited.add(key(p)); queue.push(next);
    }
  }
  return null;
}
export function attackPath(s: GameState, u: Unit, target: Target | Point, limit: number): Point[] | null {
  const isTarget = (v: Target | Point): v is Target => 'id' in v;
  const t = isTarget(target) ? target : undefined;
  const ends = t?.unit ? cells(t.unit) : [target];
  const blocked = new Set<string>();
  for (const enemy of s.units) if (enemy.owner !== u.owner && enemy.id !== t?.id) for (const p of cells(enemy)) blocked.add(key(p));
  if (`base-${other(u.owner)}` !== t?.id) blocked.add(key(basePoint(other(u.owner))));
  const starts = cells(u), queue = starts.map(p => [p]), visited = new Set(starts.map(key));
  let fallback: Point[] | null = null;
  for (let i=0; i<queue.length; i++) {
    const path = queue[i], depth = path.length-1;
    if (ends.some(p => equal(p,path[depth]))) return path;
    if (depth >= limit || (fallback && depth >= fallback.length-1)) continue;
    for (const p of neighbors(path[depth])) {
      if (blocked.has(key(p))) continue;
      const next = [...path,p];
      if (ends.some(e => equal(e,p))) {
        if (!fallback) fallback = next;
        if (t?.unit?.kind !== 24 || frontal(next,t.owner)) return next;
        continue;
      }
      if (!visited.has(key(p))) { visited.add(key(p)); queue.push(next); }
    }
  }
  return fallback;
}
export function frontal(path: Point[], owner: Player): boolean {
  if (path.length < 2) return false;
  const prev = path[path.length-2], last = path[path.length-1];
  return owner === 1 ? last.y < prev.y : last.y > prev.y;
}
