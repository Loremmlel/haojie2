import { allPieces, hasTrait, isLandmark } from './traits';
import { landmarkAt, landmarkSquare, liveLandmark } from '../setup/shrines';
import { definition } from '../catalog';
import { pieceById } from '../runtime/position';
import { allegiance, getStats, has, passive, piercing, hasWeapon } from './state';
import type { AttackDirection, GamePosition, Player, Point, Target, Unit } from '../types';
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
  // 常见占位直接构造新坐标，保留逐行顺序与调用方可独立修改返回值的约定。
  if (size === 1) return [{ x: u.x, y: u.y }];
  if (size === 2)
    return [
      { x: u.x, y: u.y },
      { x: u.x + 1, y: u.y },
      { x: u.x, y: u.y + 1 },
      { x: u.x + 1, y: u.y + 1 },
    ];
  return Array.from({ length: size * size }, (_, i) => ({
    x: u.x + (i % size),
    y: u.y + Math.floor(i / size),
  }));
}
/** 只读占位查询，不分配坐标数组；整数偏移保留逐格匹配，不能把格间位置当作占位。 */
function covers(u: Unit, p: Point): boolean {
  const size = u.size ?? definition(u.kind).size ?? 1;
  const dx = p.x - u.x,
    dy = p.y - u.y;
  return (
    Number.isInteger(dx) && Number.isInteger(dy) && dx >= 0 && dy >= 0 && dx < size && dy < size
  );
}
export const occupants = (s: GamePosition, p: Point) => s.units.filter((u) => covers(u, p));
export const occupant = (s: GamePosition, p: Point) => s.units.find((u) => covers(u, p));
export function targets(s: GamePosition): Target[] {
  return [
    ...allPieces(s).map((u) => ({ id: u.id, owner: u.owner, x: u.x, y: u.y, unit: u })),
    ...([1, 2] as Player[]).map((p) => ({ id: `base-${p}`, owner: p, ...basePoint(p) })),
  ];
}
/** 按普通棋子、有效地标、基地的原顺序定位，只为命中目标创建包装；unit 保持原引用。 */
export function targetById(s: GamePosition, id?: string): Target | undefined {
  const unit = pieceById(s, id);
  if (unit) return { id: unit.id, owner: unit.owner, x: unit.x, y: unit.y, unit };
  const owner = id === 'base-1' ? 1 : id === 'base-2' ? 2 : undefined;
  return owner ? { id: `base-${owner}`, owner, ...basePoint(owner) } : undefined;
}
export function targetAt(s: GamePosition, p: Point): Target | undefined {
  return targets(s).find((t) => (t.unit ? covers(t.unit, p) : equal(t, p)));
}
export function topTarget(s: GamePosition, t: Target) {
  if (!t.unit) return true;
  if (isLandmark(t.unit)) {
    const over = occupant(s, t);
    return !over || allegiance(s, over) !== t.owner;
  }
  return occupant(s, t)?.id === t.id;
}
export function adjacent(a: Unit, b: Unit) {
  const as = a.size ?? definition(a.kind).size ?? 1,
    bs = b.size ?? definition(b.kind).size ?? 1;
  // 与 cells 的逐行格序及坐标运算相同；纯判定直接遍历，不为每对实体构造格子数组。
  for (let i = 0; i < Math.floor(as * as); i++) {
    const x = a.x + (i % as),
      y = a.y + Math.floor(i / as);
    for (let j = 0; j < Math.floor(bs * bs); j++)
      if (Math.abs(x - (b.x + (j % bs))) + Math.abs(y - (b.y + Math.floor(j / bs))) === 1)
        return true;
  }
  return false;
}
export function ring(a: Unit, b: Point | Unit) {
  const bc = 'kind' in b ? cells(b) : [b];
  return (
    cells(a).some((c) =>
      bc.some((d) => Math.max(Math.abs(c.x - d.x), Math.abs(c.y - d.y)) === 1),
    ) && !cells(a).some((c) => bc.some((d) => equal(c, d)))
  );
}
interface PlacementContext {
  occupants: Unit[][];
  friends: Record<Player, Unit[]>;
  loners: Record<Player, Unit[]>;
  rows: Partial<Record<Player, number[]>>;
}
/**
 * 为同步只读查询批次创建索引。调用期间不得修改 s 或其棋子；新局面必须重新创建。
 * 不挂载全局缓存、不改变权威 canPlace 的默认入口，不保存到局面或消耗随机数。
 */
export function createPlacementQuery(s: GamePosition) {
  const context: PlacementContext = {
    occupants: Array.from({ length: 117 }, () => []),
    friends: { 1: [], 2: [] },
    loners: { 1: [], 2: [] },
    rows: {},
  };
  for (const v of s.units) {
    for (const p of cells(v)) if (inside(p)) context.occupants[(p.y - 1) * 9 + p.x - 1].push(v);
    const owner = allegiance(s, v);
    if (owner) {
      context.friends[owner].push(v);
      if (hasTrait(v, 23) && passive(s, v)) context.loners[owner].push(v);
    }
  }
  const place = (u: Unit, at: Point, deployment = false, ignore: string[] = []) =>
    placement(s, u, at, deployment, ignore, context);
  return {
    canPlace: place,
    movementPath: (u: Unit, to: Point, limit: number, straight = false) =>
      movement(s, u, to, limit, straight, place),
    movementField: (u: Unit, limit: number, straight = false) =>
      movementField(u, limit, straight, (p) => place(u, p)),
  };
}
export function canPlace(
  s: GamePosition,
  u: Unit,
  at: Point,
  deployment = false,
  ignore: string[] = [],
): boolean {
  return placement(s, u, at, deployment, ignore);
}
function placement(
  s: GamePosition,
  u: Unit,
  at: Point,
  deployment: boolean,
  ignore: string[],
  context?: PlacementContext,
): boolean {
  const occupied = (p: Point) =>
    context ? context.occupants[(p.y - 1) * 9 + p.x - 1] : occupants(s, p);
  const footprint = cells({ kind: u.kind, size: u.size, x: at.x, y: at.y });
  if (footprint.some((p) => !inside(p) || equal(p, basePoint(1)) || equal(p, basePoint(2))))
    return false;
  if (isLandmark(u)) {
    if (!deployment) return equal(u, at);
    if (!landmarkSquare(u.kind, at) || landmarkAt(s, at)) return false;
    const over = occupied(at).filter((v) => !ignore.includes(v.id));
    return over.length <= 1 && over.every((v) => allegiance(s, v) === u.owner);
  }
  const rows = deployment
    ? context
      ? (context.rows[u.owner] ??= deploymentRows(s, u.owner))
      : deploymentRows(s, u.owner)
    : [];
  if (
    deployment &&
    !hasTrait(u, 'u27') &&
    footprint.some((p) => {
      const land = landmarkAt(s, p);
      return !rows.includes(p.y) && !(liveLandmark(land) && allegiance(s, land) === u.owner);
    })
  )
    return false;
  for (const p of footprint) {
    const land = landmarkAt(s, p);
    if (!land) continue;
    if (u.size > 1) return false;
    if (deployment && liveLandmark(land) && allegiance(s, land) !== u.owner) return false;
    // 地标即使休眠也只能承载一个随从，不能承载克隆叠放。
    if (occupied(p).some((v) => v.id !== u.id && !ignore.includes(v.id))) return false;
  }
  // 候选落点的重叠查询只读对应格；默认权威路径仍扫描原始单位数组。
  const others = context
    ? footprint.flatMap(occupied).filter((v) => v.id !== u.id && !ignore.includes(v.id))
    : s.units.filter((v) => v.id !== u.id && !ignore.includes(v.id));
  if (
    others.some(
      (v) =>
        footprint.some((p) => covers(v, p)) &&
        !(
          u.kind === 'u25' &&
          v.kind === 'u25' &&
          u.owner === v.owner &&
          u.size === 1 &&
          v.size === 1
        ),
    )
  )
    return false;
  const isolation = context
    ? (hasTrait(u, 23) && passive(s, u)
        ? context.friends[u.owner]
        : context.loners[u.owner]
      ).filter((v) => v.id !== u.id && !ignore.includes(v.id))
    : others;
  if (
    isolation.some(
      (v) =>
        allegiance(s, v) === u.owner &&
        ((hasTrait(v, 23) && passive(s, v)) || (hasTrait(u, 23) && passive(s, u))) &&
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

/**
 * 同一来源的所有落点共用一棵广搜树；格号、前驱和距离是紧凑数值缓冲。
 * 保留旧的下/上/右/左入队顺序及分数距离上界，只在取出可行路径时分配坐标。
 * 占位约束在这次同步查询内不变；冲撞、多格逐步移动由调用者保留专用规则。
 */
function movementField(u: Unit, limit: number, straight: boolean, place: (p: Point) => boolean) {
  const parent = new Int16Array(117).fill(-1);
  const depth = new Int16Array(117);
  const queue = new Int16Array(117);
  const source = (u.y - 1) * WIDTH + u.x - 1;
  const pointAt = (cell: number): Point => ({
    x: (cell % WIDTH) + 1,
    y: Math.floor(cell / WIDTH) + 1,
  });
  if (inside(u)) {
    parent[source] = source;
    queue[0] = source;
    let tail = 1;
    for (let head = 0; head < tail; head++) {
      const cell = queue[head];
      if (straight ? depth[cell] >= 3 : depth[cell] >= limit) continue;
      const at = pointAt(cell);
      for (const p of neighbors(at)) {
        if (straight && p.x !== u.x && p.y !== u.y) continue;
        const next = (p.y - 1) * WIDTH + p.x - 1;
        if (parent[next] !== -1 || !place(p)) continue;
        parent[next] = cell;
        depth[next] = depth[cell] + 1;
        queue[tail++] = next;
      }
    }
  }
  return (to: Point): Point[] | null => {
    if (!inside(to) || equal(u, to)) return null;
    let cell = (to.y - 1) * WIDTH + to.x - 1;
    if (parent[cell] === -1 || (straight && depth[cell] !== 3)) return null;
    const path: Point[] = [];
    while (cell !== source) {
      path.push(pointAt(cell));
      cell = parent[cell];
    }
    path.push({ x: u.x, y: u.y });
    path.reverse();
    return path;
  };
}
export function movementPath(
  s: GamePosition,
  u: Unit,
  to: Point,
  limit: number,
  straight = false,
): Point[] | null {
  return movement(s, u, to, limit, straight, (unit, p) => canPlace(s, unit, p));
}
function movement(
  s: GamePosition,
  u: Unit,
  to: Point,
  limit: number,
  straight: boolean,
  place: (unit: Unit, p: Point) => boolean,
): Point[] | null {
  if (!inside(to) || equal(u, to) || !place(u, to)) return null;
  if (straight) {
    if (distance(u, to) !== 3 || (u.x !== to.x && u.y !== to.y)) return null;
    const path = Array.from({ length: 4 }, (_, i) => ({
      x: u.x + Math.sign(to.x - u.x) * i,
      y: u.y + Math.sign(to.y - u.y) * i,
    }));
    return path.slice(1).every((p) => place(u, p)) ? path : null;
  }
  // 原广搜在分数上限时仍能走到 ceil(limit)；下界剪枝须保留这个边界。
  if (distance(u, to) > Math.max(0, Math.ceil(limit))) return null;
  const queue: Point[][] = [[{ x: u.x, y: u.y }]],
    seen = new Set([key(u)]);
  for (let i = 0; i < queue.length; i++) {
    const path = queue[i];
    if (path.length - 1 >= limit) continue;
    for (const p of neighbors(path.at(-1)!)) {
      if (seen.has(key(p)) || !place(u, p)) continue;
      const next = [...path, p];
      if (equal(p, to)) return next;
      seen.add(key(p));
      queue.push(next);
    }
  }
  return null;
}
export function attackPath(
  s: GamePosition,
  u: Unit,
  target: Target | Point,
  limit: number,
  direction?: AttackDirection,
  pierce = false,
): Point[] | null {
  if (direction !== undefined)
    return (
      attackRoutes(s, u, target, limit, pierce).find((r) => r.direction === direction)?.path ?? null
    );
  return attackSearch(s, u, target, limit, pierce, false).best;
}

export function frontal(path: Point[], owner: Player) {
  if (path.length < 2) return false;
  const a = path.at(-2)!,
    b = path.at(-1)!;
  return owner === 1 ? b.y < a.y : b.y > a.y;
}
/** 从当前占位派生部署行；每枚棋子在接触行计一次，中立与独立地标不计数。 */
export function deploymentRows(s: GamePosition, owner: Player): number[] {
  const counts = Array<number>(14).fill(0);
  for (const u of s.units) {
    const side = allegiance(s, u);
    if (!side) continue;
    // 正方形每个覆盖行只计一枚；无需先展开所有格子再去重。
    const size = u.size ?? definition(u.kind).size ?? 1;
    for (let offset = 0; offset < size; offset++) counts[u.y + offset] += side === owner ? 1 : -1;
  }
  return Array.from({ length: 13 }, (_, i) => i + 1).filter(
    (y) => (owner === 1 ? y <= 8 : y >= 6) || counts[y] >= 2,
  );
}
/** 保留 v2 序列化字段，但它只镜像当前局面，不能作为权限来源。 */
export function refreshDeployment(s: GamePosition, owner: Player) {
  s.deployRows[owner] = deploymentRows(s, owner);
}
export function inSquare(u: Unit, p: Point, radius = 5) {
  return inside(p) && Math.abs(p.x - u.x) <= radius && Math.abs(p.y - u.y) <= radius;
}

/** 最后一步行进方向，供战斗、预览和 AI 共用。 */
export function pathDirection(path: Point[]): AttackDirection | undefined {
  if (path.length < 2) return undefined;
  const a = path.at(-2)!,
    b = path.at(-1)!;
  return b.y < a.y ? 'up' : b.y > a.y ? 'down' : b.x < a.x ? 'left' : 'right';
}
export interface AttackRoute {
  direction: AttackDirection;
  path: Point[];
}
/** 为每个入射方向各保留一条最短合法路径，而非只保留全局最短路。目标格为终点，攻击不能穿过目标再击中背面。不消耗随机数、不改局面，也不接收客户端路径坐标。 */
export function attackRoutes(
  s: GamePosition,
  u: Unit,
  target: Target | Point,
  limit: number,
  pierce = false,
): AttackRoute[] {
  return attackSearch(s, u, target, limit, pierce, true).routes;
}

/** 用数字前驱替代每个 BFS 节点的拥有型路径；到达结果时才展开坐标，保留精确遍历顺序。 */
function attackSearch(
  s: GamePosition,
  u: Unit,
  target: Target | Point,
  limit: number,
  pierce: boolean,
  allRoutes: boolean,
) {
  const result: { best: Point[] | null; routes: AttackRoute[] } = { best: null, routes: [] };
  const t = 'id' in target ? target : undefined,
    endCells = t?.unit ? cells(t.unit) : [target],
    starts = cells(u);
  if (
    endCells.every((to) =>
      starts.every((from) => distance(from, to) > Math.max(0, Math.ceil(limit))),
    )
  )
    return result;
  const index = (p: Point) => (p.y - 1) * WIDTH + p.x - 1;
  const point = (cell: number): Point => ({
    x: (cell % WIDTH) + 1,
    y: Math.floor(cell / WIDTH) + 1,
  });
  const ends = new Uint8Array(117),
    blocked = new Uint8Array(117);
  for (const p of endCells) if (inside(p)) ends[index(p)] = 1;
  for (const v of allPieces(s))
    if (!pierce && v.id !== u.id && v.id !== t?.id && allegiance(s, v) !== u.owner)
      for (const p of cells(v)) if (inside(p)) blocked[index(p)] = 1;
  for (const p of endCells) if (inside(p)) blocked[index(p)] = 0;
  if (t?.id !== `base-${other(u.owner)}`) blocked[index(basePoint(other(u.owner)))] = 1;
  const parent = new Int16Array(117).fill(-1),
    depth = new Uint8Array(117),
    queue = new Uint8Array(117);
  let tail = 0,
    directions = 0,
    fallbackDepth = Infinity;
  for (const p of starts) {
    const cell = index(p);
    if (allRoutes && ends[cell]) continue;
    parent[cell] = cell;
    queue[tail++] = cell;
  }
  const path = (cell: number, end?: number) => {
    const points: Point[] = end === undefined ? [] : [point(end)];
    for (;;) {
      points.push(point(cell));
      if (parent[cell] === cell) break;
      cell = parent[cell];
    }
    return points.reverse();
  };
  for (let head = 0; head < tail; head++) {
    const cell = queue[head],
      n = depth[cell];
    if (!allRoutes && ends[cell]) {
      result.best = path(cell);
      return result;
    }
    if (n >= limit || (!allRoutes && n >= fallbackDepth)) continue;
    for (let dir = 0; dir < 4; dir++) {
      if (
        (dir === 0 && cell >= 108) ||
        (dir === 1 && cell < 9) ||
        (dir === 2 && cell % 9 === 8) ||
        (dir === 3 && cell % 9 === 0)
      )
        continue;
      const next = cell + (dir === 0 ? 9 : dir === 1 ? -9 : dir === 2 ? 1 : -1);
      if (blocked[next]) continue;
      if (ends[next]) {
        if (allRoutes) {
          if (!(directions & (1 << dir))) {
            directions |= 1 << dir;
            result.routes.push({
              direction: dir === 0 ? 'down' : dir === 1 ? 'up' : dir === 2 ? 'right' : 'left',
              path: path(cell, next),
            });
            if (directions === 15) return result;
          }
        } else {
          if (result.best === null) {
            result.best = path(cell, next);
            fallbackDepth = n + 1;
          }
          if (
            !t?.unit ||
            !hasTrait(t.unit, 24) ||
            t.unit.silenced ||
            (t.owner === 1 ? dir === 1 : dir === 0)
          ) {
            result.best = path(cell, next);
            return result;
          }
        }
      } else if (parent[next] === -1) {
        parent[next] = cell;
        depth[next] = n + 1;
        queue[tail++] = next;
      }
    }
  }
  return result;
}
/** 方向敏感能力在界面与 AI 中共用同一有界候选集合。 */
export function selectableAttackRoutes(s: GamePosition, u: Unit, t: Target): AttackRoute[] {
  if (piercing(u) && !hasWeapon(u, 'u28')) return [];
  if (
    !(
      piercing(u) ||
      (hasTrait(u, 'u20') && !u.silenced) ||
      (t.unit && hasTrait(t.unit, 24) && !t.unit.silenced)
    )
  )
    return [];
  return attackRoutes(s, u, t, getStats(s, u).range, piercing(u));
}

/** 围绕原格扩展，最终覆盖格在两层都必须无占位。 */
export function expansionAnchors(s: GamePosition, u: Unit): Point[] {
  if (u.size !== 1 || isLandmark(u)) return [];
  return [
    { x: u.x, y: u.y },
    { x: u.x - 1, y: u.y },
    { x: u.x, y: u.y - 1 },
    { x: u.x - 1, y: u.y - 1 },
  ].filter((p) => canPlace(s, { ...u, size: 2 }, p));
}
/** 棋盘预览与命令验证共用；敌方基地只能作为路径终点。 */
export function validAttackRoute(s: GamePosition, u: Unit, path: Point[], limit: number): boolean {
  if (!Array.isArray(path) || !path.length || path.length > Math.min(117, limit + 1)) return false;
  if (path.some((p) => !p || !inside(p))) return false;
  if (!covers(u, path[0]) || new Set(path.map(key)).size !== path.length) return false;
  for (let i = 1; i < path.length; i++) {
    if (distance(path[i - 1], path[i]) !== 1 || covers(u, path[i])) return false;
    if (i < path.length - 1 && equal(path[i], basePoint(other(u.owner)))) return false;
  }
  return true;
}
/** 选定时捕获目标快照，每次齐射对每个栈顶及地标只命中一次，不按格重复。AOE 另按规则处理；路径前缀保留各目标的真实入射方向。 */
export function piercingTargets(s: GamePosition, u: Unit, path: Point[]) {
  const victims: { target: Target; path: Point[] }[] = [];
  const seen = new Set<string>();
  const available = targets(s).filter(
    (t) =>
      t.id !== u.id &&
      (t.unit ? allegiance(s, t.unit) : t.owner) !== u.owner &&
      (topTarget(s, t) || (t.unit && isLandmark(t.unit))),
  );
  for (let i = 1; i < path.length; i++)
    for (const t of available)
      if (!seen.has(t.id) && (t.unit ? cells(t.unit) : [t]).some((p) => equal(p, path[i]))) {
        seen.add(t.id);
        victims.push({ target: t, path: path.slice(0, i + 1) });
      }
  return victims;
}
