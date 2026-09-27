import { CATALOG, SUMMON_POOL, ULTIMATE_POOL } from '../../../src/engine/catalog';
import { ALL_CELLS } from '../../../src/engine/core/geometry';
import { applyCommand } from '../../../src/engine/commands/game';
import { add, card, fixture, seedFor } from '../../../tests/helpers';
import type { Command, GameState, Kind } from '../../../src/engine/types';
import type { Job } from './fixtures';

/** 由正式 TS 入口判断边界，不在断言中重写召唤概率、装备或蓄力规则。 */
export function preparationFixtures(): { name: string; job: Job }[] {
  const cases: { name: string; job: Job }[] = [];
  const push = (name: string, state: GameState, probes: Command[]) =>
    cases.push({ name: `preparation-${name}`, job: { state, probes } });
  for (const d of CATALOG) {
    const s = fixture(),
      cardId = card(s, d.id);
    push(`deploy-${d.id}`, s, [
      ...(d.spell === undefined && d.weapon === undefined && !d.aura
        ? ALL_CELLS
        : [{ x: 4, y: 6 }]
      ).map((p): Command => ({ type: 'deploy', cardId, ...p })),
      { type: 'deploy', cardId, x: 4, y: 6, charge: true },
      { type: 'deploy', cardId, x: 4.5, y: 6 },
      { type: 'deploy', cardId },
      { type: 'deploy', cardId: 'missing', x: 4, y: 6 },
    ]);
    if (d.spell !== undefined || d.weapon !== undefined || d.aura) continue;
    const cs = fixture(),
      u = add(cs, d.id, 1, 4, 6);
    if (d.landmark) {
      cs.units = [];
      cs.landmarks = [u];
    }
    push(
      `charge-${d.id}`,
      cs,
      ['attack', 'move', 'skill'].map((mode): Command => ({ type: 'charge', unitId: u.id, mode })),
    );
  }
  for (const name of [
    'friend-landmark',
    'enemy-landmark',
    'dormant',
    'silenced',
    'banner',
    'clone',
    'loner',
    'live-row',
    'closed-row',
    'shrine-setup',
  ]) {
    for (const k of [1, 5, 'u5', 'u12', 'u25', 'u27'] as Kind[]) {
      const s = fixture(),
        cardId = card(s, k);
      if (name === 'shrine-setup') s.phase = 'shrine-setup';
      else if (name === 'clone') add(s, 'u25', 1, 4, 7);
      else if (name === 'loner') add(s, 23, 1, 3, 7);
      else if (name.endsWith('row')) {
        add(s, 1, 1, 1, 10);
        add(s, 1, 1, 2, 10);
        if (name === 'closed-row') add(s, 1, 2, 3, 10);
        s.deployRows[1] = [13];
      } else {
        const l = add(s, name === 'banner' ? 's8' : 's1', name === 'enemy-landmark' ? 2 : 1, 4, 7);
        s.units = [];
        s.landmarks = [l];
        if (name === 'dormant') {
          l.hp = 0;
          s.landmarks[0].dormantSince = s.ply;
        }
        if (name === 'silenced') l.silenced = true;
      }
      push(`placement-${name}-${k}`, s, [
        { type: 'deploy', cardId, x: 4, y: 7 },
        { type: 'deploy', cardId, x: 4, y: 7, charge: true },
        { type: 'deploy', cardId, x: 4, y: 10 },
      ]);
    }
  }
  for (const name of [
    'inherited',
    'frozen',
    'sleeping',
    'silenced',
    'full',
    'other-reserve',
    'other-mode',
    'used',
    'once',
    'extra-operation',
  ]) {
    for (const k of [3, 4, 15, 21, 'u2', 'u6'] as Kind[]) {
      const s = fixture(),
        u = add(s, name === 'inherited' ? 1 : k, 1, 4, 6);
      if (name === 'inherited') {
        u.traits = [k];
        u.charge = 3;
        u.readyCharge = 2;
        u.abilityCharges = {
          [k]: {
            charge: 1,
            readyCharge: 0,
            chargeType: 'attack' as const,
            lastCharge: -1,
          },
        };
      }
      if (name === 'frozen') u.effects.push({ type: 'freeze', owner: 2, from: 0, until: 20 });
      if (name === 'sleeping') u.born = s.turns[1];
      if (name === 'silenced') u.silenced = true;
      if (name === 'full') u.charge = 5;
      if (name === 'other-reserve') {
        u.charge = 1;
        u.chargeType = 'move';
      }
      if (name === 'other-mode') u.mode = 'attack';
      if (name === 'used') u.operations = 1;
      if (name === 'extra-operation') {
        u.operations = 1;
        u.extraOperations = 1;
      }
      if (name === 'once') u.onceUsed = true;
      push(`charge-${name}-${k}`, s, [
        ...['attack', 'move', 'skill'].map(
          (mode): Command => ({
            type: 'charge',
            unitId: u.id,
            ability: k,
            mode,
          }),
        ),
        { type: 'charge', unitId: u.id, ability: 26, mode: 'attack' },
      ]);
    }
  }
  for (const d of CATALOG.filter((d) => d.weapon !== undefined)) {
    for (const k of [1, 10, 'u3', 's7', 'grave'] as Kind[]) {
      const s = fixture(),
        u = add(s, k, 1, 4, 6),
        cardId = card(s, d.id);
      u.equipment = ['u11'];
      u.equipmentIds = { u11: 'old-weapon' };
      u.maxHp += 25;
      u.hp += 25;
      u.overMaxFromBanner = true;
      u.effects.push({ type: 'freeze', owner: 2, from: 0, until: 10 });
      push(`equip-${d.id}-${k}`, s, [
        { type: 'equip', cardId, targetId: u.id },
        { type: 'equip', cardId, targetId: 'missing' },
      ]);
    }
  }
  for (const d of CATALOG.filter((d) => d.aura)) {
    for (const parity of [undefined, 'odd'] as const) {
      const s = fixture(),
        cardId = card(s, d.id);
      if (parity) s.hands[1][0].parity = parity;
      push(`aura-${d.id}-${parity ?? 'missing'}`, s, [{ type: 'activate-aura', cardId }]);
    }
  }
  for (const ultimate of [false, true]) {
    const pool = ultimate ? ULTIMATE_POOL : SUMMON_POOL;
    for (const [i, chosenKind] of pool.entries()) {
      const s = fixture();
      s.phase = 'summon';
      s.summonSlots = 2;
      s.rng = seedFor((i + 0.1) / pool.length, (i + 0.9) / pool.length);
      push(`draw-${chosenKind}`, s, [{ type: 'summon', ultimate }]);
      const selected = structuredClone(s);
      selected.auras = { 1: [{ kind: 'laoqian' }], 2: [] };
      push(`chosen-${chosenKind}`, selected, [
        { type: 'summon', ultimate, chosenKind },
        { type: 'summon', ultimate, chosenKind: ultimate ? 1 : 'u1' },
        { type: 'summon', ultimate, chosenKind: 'u12p' },
      ]);
    }
  }
  // 固定不同种子覆盖同编号的派生概率；不把穷举随机结果混作 AI 可见的未来随机数。
  for (const seed of [1, 19, 97, 3011, 19777, 57381, 90210, 183746, 290001, 350009]) {
    for (const shrine of [false, true]) {
      const s = fixture();
      s.phase = 'summon';
      s.summonSlots = 2;
      s.rng = seed;
      if (shrine) {
        s.mode = 'shrine';
        s.regularSummons = 2;
        s.auras = { 1: [{ kind: 's13' }], 2: [] };
      }
      push(`summon-${seed}-${shrine}`, s, [
        { type: 'summon' },
        { type: 'summon', ultimate: true },
        { type: 'summon', ultimate: false, mode: 'normal' },
        { type: 'summon', chosenKind: 1 },
        { type: 'extra-summon', ultimate: false },
        { type: 'extra-summon', ultimate: true },
      ]);
      if (shrine) {
        const offer = applyCommand(s, { type: 'summon' });
        push(
          `offer-${seed}`,
          offer,
          [[0, 1], [1, 0], [0, 0], [0, 5], [0.5, 1], []].map(
            (offerIndices): Command => ({
              type: 'choose-summons',
              offerIndices,
            }),
          ),
        );
      }
    }
  }
  // 定向覆盖3/17/u12的两个派生分支，以及老千K候选中整批克隆的选择顺序。
  const variants = new Set<string>();
  for (let i = 1; variants.size < 7 && i <= 2000; i++) {
    for (const ultimate of [false, true]) {
      const s = fixture();
      s.phase = 'summon';
      s.summonSlots = 2;
      s.rng = Math.imul(i, 2654435761) >>> 0;
      const next = applyCommand(s, { type: 'summon', ultimate });
      const k = next.hands[1][0].kind,
        label = String(k);
      if (![3, '3p', 17, '17p', 'u12', 'u12p', 'u25'].includes(k) || variants.has(label)) continue;
      variants.add(label);
      push(`variant-${label}`, s, [{ type: 'summon', ultimate }]);
      if (k === 'u25') {
        next.summonOffer = {
          owner: 1,
          groups: [next.hands[1], [{ id: 'offer-other', kind: 1, drawnAt: 3, summonedPly: s.ply }]],
          count: 2,
        };
        next.hands[1] = [];
        push('offer-clone-batch', next, [{ type: 'choose-summons', offerIndices: [1, 0] }]);
      }
    }
  }
  if (variants.size !== 7) throw new Error('派生召唤专项缺少分支');
  for (const name of [
    'self',
    'repeat',
    'late-self',
    'mage',
    'used',
    'legacy-used',
    'inherited',
    'silenced',
    'frozen',
    'clone',
    'partial-clone',
    'old-card',
    'unspent',
    'no-pool',
  ]) {
    const s = fixture(),
      cardId = card(s, name.includes('clone') ? 'u25' : name === 'no-pool' ? 's2' : 'u13');
    const c = s.hands[1][0];
    if (name.includes('clone')) {
      c.group = 'batch';
      for (let i = 1; i < (name === 'clone' ? 8 : 7); i++)
        s.hands[1].push({ ...c, id: `${cardId}-${i}` });
    }
    if (name === 'repeat') c.rerolled = true;
    if (name === 'late-self') s.turns[1] = 6;
    if (name === 'old-card') c.summonedPly = s.ply - 1;
    if (name === 'unspent') s.summonSlots = 1;
    const u = add(s, name === 'inherited' ? 1 : 'u13', 1, 4, 6);
    if (name === 'inherited') u.traits = ['u13'];
    if (name === 'used') u.rerollUsedPly = s.ply;
    if (name === 'legacy-used') u.freeUsed = s.ply;
    if (name === 'silenced') u.silenced = true;
    if (name === 'frozen') u.effects.push({ type: 'freeze', owner: 2, from: 0, until: 20 });
    s.auras = { 1: [{ kind: 'laoqian' }], 2: [] };
    push(`reroll-${name}`, s, [
      { type: 'reroll', cardId },
      { type: 'reroll', cardId, unitId: u.id },
      { type: 'reroll', cardId, unitId: u.id, chosenKind: 'u25' },
      { type: 'reroll', cardId, unitId: 'missing' },
    ]);
  }
  for (const phase of ['play', 'synthesis', 'summon', 'shrine-setup', 'shrine-draft'] as const) {
    for (const slots of [0, 1, -1]) {
      const s = fixture();
      s.phase = phase;
      s.summonSlots = slots;
      s.heads[1] = 0;
      push(`stage-${phase}-${slots}`, s, [
        { type: 'begin' },
        { type: 'skip-synthesis' },
        { type: 'summon', ultimate: true },
        { type: 'extra-summon' },
        { type: 'choose-summons' },
        { type: 'activate-aura' },
        { type: 'deploy' },
        { type: 'charge' },
        { type: 'equip' },
        { type: 'reroll' },
      ]);
    }
  }
  return cases;
}
