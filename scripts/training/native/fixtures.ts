import { CATALOG } from '../../../src/engine/catalog';
import { ALL_CELLS } from '../../../src/engine/core/geometry';
import { add, fixture } from '../../../tests/helpers';
import type { Command, GameState, Unit } from '../../../src/engine/types';

export interface Job {
  state: GameState;
  probes: Command[];
  command?: Command;
}
export function moveProbes(unitId: string): Command[] {
  return [
    ...ALL_CELLS.map((p): Command => ({ type: 'move', unitId, ...p })),
    { type: 'move', unitId, x: 0, y: 7 },
    { type: 'move', unitId, x: 4.5, y: 7 },
    { type: 'move', unitId },
    { type: 'move', unitId: 'missing', x: 4, y: 7 },
    { type: 'finish-mode', unitId },
  ];
}

/** 专项只构造规则边界，不存逐步快照；每个落点都交给正式 TS 入口作参照。 */
export function fixtures(): { name: string; job: Job }[] {
  const cases: { name: string; job: Job }[] = [];
  const push = (name: string, state: GameState, u: Unit) =>
    cases.push({ name, job: { state, probes: moveProbes(u.id) } });
  for (const d of CATALOG.filter(
    (d) => d.spell === undefined && d.weapon === undefined && !d.aura,
  )) {
    const s = fixture(),
      u = add(s, d.id, 1, 4, 6);
    if (d.landmark) {
      s.units = [];
      s.landmarks = [u];
    }
    u.readyCharge = u.charge = 2;
    u.equipment = ['u16'];
    push(`catalog-${d.id}`, s, u);
  }
  for (const name of [
    'enemy',
    'freeze-local',
    'freeze-global',
    'expired-freeze',
    'stun',
    'sleep',
    'extra',
    'used',
    'other-mode',
    'silenced',
    'inherited',
    'giant',
    'giant-continuing',
    'clone',
    'loner',
    'landmark',
    'banner',
    'legacy-guard',
    'pending',
    'synthesis',
    'transit',
    'siphon',
    'finish-attack',
    'finish-bonus',
  ]) {
    const s = fixture(),
      u = add(
        s,
        name === 'clone'
          ? 'u25'
          : name === 'giant-continuing'
            ? 3
            : name === 'inherited'
              ? 'formless'
              : 12,
        1,
        4,
        6,
      );
    add(s, 1, 2, 4, 7);
    if (name === 'enemy') u.owner = 2;
    if (['freeze-local', 'freeze-global', 'expired-freeze', 'stun'].includes(name)) {
      u.offset = 2;
      u.effects = [
        {
          type: name === 'stun' ? 'stun' : 'freeze',
          from: 5,
          until: name === 'expired-freeze' ? 7 : 8,
          owner: 2,
          global: name === 'freeze-global',
        },
      ];
    }
    if (name === 'sleep') u.born = 3;
    if (name === 'extra') {
      u.extraOperations = 1;
      u.operations = 1;
    }
    if (name === 'used') u.operations = 1;
    if (name === 'other-mode') u.mode = 'attack';
    if (name === 'silenced') {
      u.traits = [23, 'u12'];
      u.silenced = true;
    }
    if (name === 'inherited') {
      u.traits = [3, 13];
      u.abilityCharges = { 3: { charge: 2, readyCharge: 1, chargeType: 'move', lastCharge: 3 } };
    }
    if (name === 'giant' || name === 'giant-continuing') {
      u.size = 2;
      s.units.pop();
    }
    if (name === 'giant-continuing') {
      u.mode = 'move';
      u.moves = 2;
      u.charge = u.readyCharge = 0;
    }
    if (name === 'clone') add(s, 'u25', 1, 3, 6);
    if (name === 'loner') add(s, 23, 1, 2, 6);
    if (name === 'landmark' || name === 'banner') {
      const l = add(s, name === 'banner' ? 's8' : 's1', 1, 4, 7);
      s.units.pop();
      s.landmarks = [l];
      u.bannerHp = name === 'banner' ? 0 : 10;
      u.overMaxFromBanner = true;
    }
    if (name === 'legacy-guard') {
      u.guardUsed = true;
      add(s, 3, 1, 2, 6);
    }
    if (name === 'pending')
      s.pending.push({ kind: 'death-shot', source: structuredClone(u), owner: 1, amount: 0 });
    if (name === 'synthesis') s.phase = 'synthesis';
    if (name === 'transit') {
      const runner = add(s, 'u12p', 1, 4, 7);
      runner.mode = 'move';
      runner.moves = 2;
    }
    if (name === 'siphon')
      s.siphons = [{ id: 'link', sourceId: u.id, owner: 1, fromId: 'base-1', toId: 'base-2' }];
    if (name.startsWith('finish')) {
      u.mode = 'attack';
      u.shots = 1;
      u.bonusSequence = name === 'finish-bonus';
    }
    push(name, s, u);
  }
  return cases;
}
