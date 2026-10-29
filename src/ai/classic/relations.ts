import { classicPosition, createClassicCommandInspector } from './options';
import { decisionOwner } from '../observation';
import {
  basePoint,
  cells,
  distance,
  createAttackQuery,
  attackPath,
} from '../../engine/core/geometry';
import { getStats, piercing } from '../../engine/core/state';
import type { Observation } from '../types';
import type { Player, Target } from '../../engine/types';

/** 与 Rust 紧凑实体顺序相同的几何输入；完整规则查询决定射程、阻挡与当前可攻击性。 */
export function classicRelations(observation: Observation) {
  const s = classicPosition(observation),
    viewer = decisionOwner(observation),
    other = (3 - viewer) as Player;
  const entries: { owner: number; target?: Target }[] = [
    { owner: 0 },
    ...[viewer, other].map((p) => ({
      owner: p,
      target: { id: `base-${p}`, owner: p, ...basePoint(p as Player) },
    })),
    ...s.units.map((u) => ({
      owner: u.owner,
      target: { id: u.id, owner: u.owner, x: u.x, y: u.y, unit: u },
    })),
    ...[viewer, other].flatMap((p) => s.hands[p as Player].map(() => ({ owner: p }))),
    ...s.pending.map((r) => ({
      owner: r.source.owner,
      target: {
        id: r.source.id,
        owner: r.source.owner,
        x: r.source.x,
        y: r.source.y,
        unit: r.source,
      },
    })),
  ];
  const attacks = createAttackQuery(s),
    inspect = createClassicCommandInspector(observation, viewer),
    open = { ...s, units: [] };
  const relations: number[] = [];
  for (let i = 0; i < entries.length; i++)
    for (let j = 0; j < entries.length; j++) {
      const a = entries[i],
        b = entries[j],
        row = [0, 0, 0, 0, 0, 0, Number(a.owner !== 0 && a.owner === b.owner), 0];
      if (a.target && b.target) {
        const from = a.target,
          to = b.target;
        row[0] = (to.x - from.x) / 9;
        row[1] = (to.y - from.y) / 13;
        const d = Math.min(
          ...(from.unit ? cells(from.unit) : [from]).flatMap((p) =>
            (to.unit ? cells(to.unit) : [to]).map((q) => distance(p, q)),
          ),
        );
        row[2] = d / 22;
        if (from.unit) {
          const u = from.unit,
            range = getStats(s, u).range;
          row[3] = range / 100;
          if (i !== j && d <= range) {
            const reachable = attacks.path(u, to, range, undefined, piercing(u)) !== null;
            row[4] = Number(reachable);
            row[5] = Number(
              !reachable && attackPath(open, u, to, range, undefined, piercing(u)) !== null,
            );
            row[7] = Number(
              inspect({ type: 'attack', unitId: u.id, targetId: to.id }).status !== 'invalid',
            );
          }
        }
      }
      relations.push(...row.map(Math.fround));
    }
  return { entities: entries.length, relations };
}
