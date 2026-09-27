import { CATALOG } from '../../../src/engine/catalog';
import { ALL_CELLS } from '../../../src/engine/core/geometry';
import { add, fixture } from '../../../tests/helpers';
import type { Command, GameState, Unit } from '../../../src/engine/types';
import type { Job } from './fixtures';

/** 攻击分别验证不致死、致死、随机免疫/暴击和排队反应，不把专项样本算作对局。 */
export function combatFixtures(): { name: string; job: Job }[] {
  const cases: { name: string; job: Job }[] = [];
  const push = (name: string, state: GameState, u: Unit, target: Unit) => {
    const ids = [target.id, u.id, 'base-1', 'base-2', 'missing'];
    const probes: Command[] = ids.flatMap((targetId) =>
      [undefined, 'heal', 'damage'].map(
        (mode): Command => ({
          type: 'attack',
          unitId: u.id,
          targetId,
          ...(mode ? { mode } : {}),
        }),
      ),
    );
    probes.push(
      ...(['up', 'down', 'left', 'right'] as const).map(
        (direction): Command => ({
          type: 'attack',
          unitId: u.id,
          targetId: target.id,
          direction,
        }),
      ),
    );
    cases.push({ name, job: { state, probes } });
  };
  for (const d of CATALOG.filter(
    (d) => d.spell === undefined && d.weapon === undefined && !d.aura && !d.landmark,
  )) {
    for (const lethal of [false, true]) {
      const s = fixture(),
        u = add(s, d.id, 1, 4, 6),
        victim = add(s, 14, 2, 4, 8);
      u.charge = u.readyCharge = 5;
      u.chargeType = 'attack';
      victim.hp = victim.maxHp = lethal ? 5 : 1000;
      push(`attack-${d.id}-${lethal ? 'death' : 'live'}`, s, u, victim);
    }
  }
  for (const victimKind of [
    2,
    3,
    11,
    12,
    16,
    20,
    24,
    '17p',
    'u10',
    'u18',
    'u21',
    'u25',
    'sage',
    'slayer',
    's8',
  ] as const) {
    const s = fixture(),
      u = add(s, 26, 1, 4, 6),
      victim = add(s, victimKind, 2, 4, 7);
    victim.hp = 5;
    if (victimKind === 's8') {
      s.units.pop();
      s.landmarks = [victim];
    }
    if (victimKind === 16) victim.owner = 1;
    push(`victim-${victimKind}`, s, u, victim);
  }
  for (const name of [
    'guard',
    'guardian-order',
    'mark',
    'death-hut',
    'death-city',
    'weapon-return',
    'drain',
    'freeze',
    'burn',
    'inherit-charge',
    'retaliation',
    'reflection',
    'base-win',
    'mixed-damage',
    'immune',
    'friendly-reflect',
    'hut-sacrifice',
  ]) {
    const s = fixture(),
      u = add(
        s,
        name === 'inherit-charge' ? 'formless' : name === 'retaliation' ? 'u18' : 26,
        1,
        4,
        6,
      ),
      victim = add(
        s,
        name === 'retaliation'
          ? 'u18'
          : name === 'reflection'
            ? 'slayer'
            : name === 'friendly-reflect'
              ? 16
              : 2,
        2,
        4,
        7,
      );
    u.hp = 20;
    u.attackBonus = 10;
    victim.hp = 10;
    if (name.startsWith('guard')) {
      const guard = add(s, 3, 2, 3, 7);
      guard.deployedAt = 3;
      if (name === 'guardian-order') {
        const earlier = add(s, 3, 2, 5, 7);
        earlier.deployedAt = 1;
      }
    }
    if (name === 'mark')
      victim.effects = [
        {
          type: 'mark',
          from: 0,
          until: 99,
          owner: 1,
          sourceId: u.id,
          global: true,
        },
        { type: 'mark', from: 0, until: 99, owner: 1, global: true },
      ];
    if (name.startsWith('death-')) add(s, name === 'death-hut' ? 'u22' : 'citadel', 2, 3, 7);
    if (name === 'weapon-return') {
      victim.equipment = ['s2'];
      victim.equipmentIds = { s2: 'original-card' };
    }
    if (name === 'drain') u.equipment = ['u11'];
    if (name === 'freeze') {
      u.equipment = ['u5'];
      victim.hp = 100;
    }
    if (name === 'burn') {
      u.traits = ['u6'];
      victim.hp = 100;
    }
    if (name === 'inherit-charge') {
      u.traits = [15, 4];
      u.abilityCharges = {
        15: { charge: 2, readyCharge: 2, chargeType: 'attack', lastCharge: 0 },
        4: { charge: 5, readyCharge: 5, chargeType: 'attack', lastCharge: 0 },
      };
    }
    if (name === 'retaliation') {
      u.hp = 200;
      victim.hp = 200;
    }
    if (name === 'mixed-damage') {
      u.traits = ['s6'];
      u.equipment = ['s15'];
      u.receivedDamage = [{ ply: s.ply, amount: 3 }];
      victim.hp = 100;
      victim.maxHp = 120;
    }
    if (name === 'immune') victim.effects = [{ type: 'immune', owner: 2, from: 0, until: 99 }];
    if (name === 'friendly-reflect') {
      victim.owner = 1;
      victim.hp = 100;
    }
    if (name === 'base-win') {
      u.x = 5;
      u.y = 12;
      s.bases[2] = 5;
    }
    if (name === 'hut-sacrifice') {
      const hut = add(s, 'u22', 1, 3, 6);
      hut.hp = hut.maxHp = 10;
      s.pending = [
        {
          kind: 'hut-spawn',
          owner: 1,
          source: structuredClone(hut),
          amount: 0,
        },
      ];
    }
    push(name, s, u, victim);
    if (s.pending.length)
      cases.push({
        name: `${name}-queued`,
        job: { state: s, probes: reactionProbes(s) },
      });
  }
  // 真实回放发现：致死后 TS 仍产生标记施加事件，不能使用旧满血快照跳过它。
  for (const victimKind of [12, 14, 's8'] as const) {
    const s = fixture(),
      u = add(s, 10, 1, 4, 6),
      victim = add(s, victimKind, 2, 4, 7);
    victim.hp = victim.maxHp = 1;
    if (victimKind === 's8') {
      s.units.pop();
      s.landmarks = [victim];
    }
    push(`catapult-lethal-full-${victimKind}`, s, u, victim);
  }
  {
    const s = fixture(),
      u = add(s, 's1', 1, 4, 6);
    s.units.pop();
    s.landmarks = [u];
    // 金晔射程为0；敌方移动到地标同格后才会发生这里的攻击。
    const victim = add(s, 14, 2, 4, 6);
    victim.hp = 1;
    push('landmark-kill-no-unit-reward', s, u, victim);
    cases.at(-1)!.job.command = {
      type: 'attack',
      unitId: u.id,
      targetId: victim.id,
    };
  }
  for (const kind of [1, 'u1', 'u8'] as const)
    for (let seed = 1; seed <= 48; seed++) {
      const s = fixture(),
        u = add(s, kind, 1, 4, 6),
        victim = add(s, '17p', 2, 4, 7);
      s.rng = (seed * 2654435761) >>> 0;
      victim.hp = victim.maxHp = 1000;
      u.kills = seed % 6;
      cases.push({
        name: `rng-${kind}-${seed}`,
        job: {
          state: s,
          probes: [{ type: 'attack', unitId: u.id, targetId: victim.id }],
          command: { type: 'attack', unitId: u.id, targetId: victim.id },
        },
      });
    }
  return cases;
}

export function reactionProbes(s: GameState): Command[] {
  if (!s.pending.length) return [];
  const r = s.pending[0];
  if (r.kind === 'bounce' || r.kind === 'hut-spawn')
    return [{ type: 'react' }, ...ALL_CELLS.map((p): Command => ({ type: 'react', ...p }))];
  if (r.kind === 'hit-pull') return [{ type: 'react' }, { type: 'react', mode: 'pull' }];
  return [
    { type: 'react' },
    ...[
      ...s.units.map((u) => u.id),
      ...(s.landmarks ?? []).map((u) => u.id),
      'base-1',
      'base-2',
      'missing',
    ].map((targetId): Command => ({ type: 'react', targetId })),
  ];
}
