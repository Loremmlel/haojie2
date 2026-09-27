import { CATALOG } from '../../../src/engine/catalog';
import { ALL_CELLS } from '../../../src/engine/core/geometry';
import { createGame, applyCommand } from '../../../src/engine/commands/game';
import { SYNTHESIS_RECIPES } from '../../../src/engine/setup/synthesis';
import { captureClockFrame } from '../../../src/engine/setup/shrines';
import { add, card, fixture } from '../../../tests/helpers';
import type { Command, GameState, Kind } from '../../../src/engine/types';
import type { Job } from './fixtures';

/** 比较真实入口的可观察结果，覆盖历史随机采样很少触及的时钟、继承、暗选与延迟结算。 */
export function completeFixtures(): { name: string; job: Job }[] {
  const cases: { name: string; job: Job }[] = [];
  const push = (name: string, state: GameState, probes: Command[]) =>
    cases.push({ name: `complete-${name}`, job: { state, probes } });
  for (const k of [5, 6, 7, 14, 19, 21, 'u6', 'u7', 'u14', 'u19', 'u21', 'u23', 'u24'] as Kind[]) {
    for (const mode of ['native', 'inherited', 'frozen', 'protected', 'last-life']) {
      const s = fixture(),
        u = add(s, mode === 'inherited' ? 's5' : k, 1, 4, 6);
      if (mode === 'inherited') {
        u.traits = [k];
        u.abilityCharges = {
          [k]: {
            charge: 2,
            readyCharge: 2,
            chargeType: 'skill' as const,
            lastCharge: -1,
          },
        };
      }
      u.charge = u.readyCharge = 2;
      u.chargeType = 'skill';
      u.hookReadyAt = s.ply;
      u.hookExpiresAt = s.ply + 2;
      if (mode === 'frozen') u.effects.push({ type: 'freeze', owner: 2, from: 0, until: 20 });
      if (mode === 'last-life') u.hp = u.maxHp = 5;
      const v = add(s, 14, 1, 5, 6),
        enemy = add(s, 26, 2, 4, 8);
      enemy.hp = enemy.maxHp = 100;
      if (mode === 'protected') add(s, 'u15', 2, 5, 8);
      s.deaths.push({
        id: 'dead-fixture',
        kind: 'u25',
        owner: 1,
        ply: s.ply - 2,
        revived: false,
      });
      const base: Command = {
        type: 'skill',
        unitId: u.id,
        ability: k,
        targetId: enemy.id,
        x: 4,
        y: 7,
        column: 4,
        secondId: v.id,
        deathId: 'dead-fixture',
      };
      push(`skill-${k}-${mode}`, s, [
        ...[u.id, v.id, enemy.id, 'base-2', 'missing'].map((targetId) => ({
          ...base,
          targetId,
        })),
        ...ALL_CELLS.map((p) => ({ ...base, ...p })),
        { ...base, targetId: v.id, mode: 'summon' },
        { ...base, x: undefined, y: undefined },
      ]);
    }
  }
  for (const d of CATALOG.filter((d) => d.spell !== undefined)) {
    for (const mode of ['plain', 'counter', 'frozen-counter', 'tower', 'inherited-counter']) {
      const s = fixture(),
        cardId = card(s, d.id),
        a = add(s, 26, 1, 4, 6),
        b = add(s, 14, 1, 5, 6),
        enemy = add(s, 'slayer', 2, 4, 8);
      if (mode.includes('counter')) {
        const mage = add(s, mode === 'inherited-counter' ? 's5' : 'archmage', 2, 7, 7);
        if (mode === 'inherited-counter') mage.traits = ['archmage'];
        if (mode === 'frozen-counter')
          mage.effects.push({ type: 'freeze', owner: 1, from: 0, until: 20 });
      }
      if (mode === 'tower') add(s, 'u15', 2, 5, 8);
      enemy.size = 2;
      const command: Command = {
        type: 'cast',
        cardId,
        targetId: a.id,
        x: 4,
        y: 8,
        mode: 'row',
        row: 8,
      };
      push(`cast-${d.id}-${mode}`, s, [
        command,
        { ...command, targetId: enemy.id },
        { ...command, targetId: 'base-2' },
        { ...command, targetId: 'missing' },
        { ...command, mode: 'double', sacrificeIds: [a.id, b.id] },
      ]);
    }
  }
  for (const recipe of SYNTHESIS_RECIPES) {
    const s = fixture();
    s.phase = 'synthesis';
    const ids = Array.from({ length: 3 }, (_, i) =>
      recipe.source === 'board'
        ? add(s, recipe.material, 1, 2 + i, 5).id
        : card(s, recipe.material),
    );
    push(`synthesis-${recipe.id}`, s, [
      ...ALL_CELLS.map(
        (p): Command => ({
          type: 'synthesize',
          recipeId: recipe.id,
          materialIds: ids,
          ...p,
        }),
      ),
      {
        type: 'synthesize',
        recipeId: recipe.id,
        materialIds: [ids[0], ids[0], ids[1]],
        x: 3,
        y: 5,
      },
    ]);
  }
  for (let seed = 1; seed <= 24; seed++) {
    let s = createGame(seed, 'shrine');
    const one: Command = {
      type: 'choose-shrine',
      player: 1,
      shrineKind: s.shrineDraft!.offers[1][0],
      parity: 'odd',
    };
    push(`draft-${seed}-first`, s, [one, { ...one, player: 2 }, { ...one, shrineKind: 1 }]);
    s = applyCommand(s, one);
    const two: Command = {
      type: 'choose-shrine',
      player: 2,
      shrineKind: s.shrineDraft!.offers[2][0],
      parity: 'even',
    };
    push(`draft-${seed}-reveal`, s, [two]);
    s = applyCommand(s, two);
    push(`setup-${seed}-first`, s, [{ type: 'finish-shrine-setup' }]);
    s = applyCommand(s, { type: 'finish-shrine-setup' });
    push(`setup-${seed}-begin`, s, [{ type: 'finish-shrine-setup' }]);
  }
  for (const mode of [
    'plain',
    'burn',
    'freeze',
    'firelord',
    'shrine',
    'hut',
    'hand',
    'clock',
    'synthesis',
  ]) {
    const s = fixture(),
      u = add(s, mode === 'firelord' ? 'firelord' : 26, 1, 4, 6),
      enemy = add(s, mode === 'hut' ? 2 : 26, 2, 5, 7);
    if (mode === 'burn' || mode === 'freeze' || mode === 'hut')
      enemy.effects.push({
        type: mode === 'freeze' ? 'freeze' : 'burn',
        owner: 1,
        from: 0,
        until: 30,
        amount: 50,
        sourceId: u.id,
      });
    if (mode === 'hand') card(s, 26);
    if (mode === 'synthesis') for (let i = 0; i < 3; i++) add(s, 'u23', 2, 2 + i, 10);
    if (mode === 'shrine') {
      const l = add(s, 's8', 2, 5, 7);
      s.units = s.units.filter((v) => v.id !== l.id);
      s.landmarks = [{ ...l, hp: 0, dormantSince: s.ply - 2, rebuildTicks: 9 }];
    }
    if (mode === 'clock') {
      s.auras = { 1: [{ kind: 's10' }], 2: [] };
      captureClockFrame(s);
    }
    s.hazards.push({
      id: 'hazard-fixture',
      owner: 1,
      axis: 'row',
      line: 7,
      due: s.ply + 1,
    });
    s.iceMarks.push({
      id: 'ice-fixture',
      owner: 1,
      sourceId: u.id,
      x: 5,
      y: 7,
      due: s.ply + 1,
    });
    push(`end-${mode}`, s, [{ type: 'end' }]);
  }
  for (const mode of ['previous', 'new-enemy', 'occupied', 'tower', 'used']) {
    const s = fixture(),
      u = add(s, 26, 1, 3, 6),
      enemy = add(s, 14, 2, 5, 7);
    s.auras = { 1: [{ kind: 's10' }], 2: [] };
    if (mode !== 'new-enemy') {
      captureClockFrame(s);
      s.ply += 2;
      s.turns[1]++;
      captureClockFrame(s);
    }
    enemy.deployedAt = s.ply;
    u.hp = 5;
    u.x = 4;
    if (mode === 'occupied') add(s, 26, 1, 3, 6);
    if (mode === 'tower') add(s, 'u15', 2, 6, 7);
    if (mode === 'used') s.auras[1][0].usedPly = s.ply;
    push(`clock-${mode}`, s, [
      { type: 'clock', targetId: u.id },
      { type: 'clock', targetId: enemy.id },
    ]);
  }
  for (const k of [1, 2, 20, 'u25', 's5'] as Kind[]) {
    const s = fixture(),
      u = add(s, k, 1, 4, 6),
      enemy = add(s, 'slayer', 2, 4, 7);
    s.auras = { 1: [{ kind: 's9', parity: 'odd' }], 2: [] };
    push(`shatter-${k}`, s, [{ type: 'shatter', unitId: u.id, targetId: enemy.id }]);
  }
  {
    const s = fixture(),
      u = add(s, 14, 1, 4, 6),
      sacrificed = add(s, 20, 1, 3, 6);
    add(s, 26, 2, 4, 8);
    push('sacrifice-reflect-exclusion', s, [
      { type: 'skill', unitId: u.id, targetId: sacrificed.id, column: 4 },
    ]);
  }
  {
    const s = fixture(),
      u = add(s, 's5', 1, 4, 6),
      victim = add(s, 2, 2, 4, 7);
    victim.hp = 1;
    victim.equipment = ['s2'];
    victim.equipmentIds = { s2: 'weapon-return-identity' };
    push('steal-before-death-reaction-snapshot', s, [
      { type: 'attack', unitId: u.id, targetId: victim.id },
    ]);
  }
  return cases;
}
