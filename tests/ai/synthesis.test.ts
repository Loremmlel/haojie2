import test from 'node:test';
import assert from 'node:assert/strict';
import {
  SYNTHESIS_RECIPES,
  applyCommand,
  asTarget,
  getStats,
  isLegal,
  firelordStrike,
  type GameState,
} from '../../src/engine';
import { candidateGroups } from '../../src/ai/planning/candidates';
import { observe, imagined, fingerprint } from '../../src/ai/observation';
import { decide } from '../../src/ai/planning/search';
import { attackPressure } from '../../src/ai/evaluation/threats';
import { hitDistance } from '../../src/ai/evaluation/spatial';
import { add, card, fixture } from '../helpers';
import type { Difficulty } from '../../src/ai/types';
const levels: Difficulty[] = ['easy', 'medium', 'hard'];
const options = (s: GameState, d: Difficulty) =>
  candidateGroups(imagined(observe(s)), d).flatMap((g) => g.commands);

test('2.5 every AI difficulty enumerates every recipe, legal post-removal destinations and an explicit keep-materials alternative', () => {
  for (const recipe of SYNTHESIS_RECIPES) {
    const s = fixture();
    s.phase = 'synthesis';
    s.summonSlots = 2;
    for (let i = 0; i < 3; i++)
      recipe.source === 'board' ? add(s, recipe.material, 1, 2 + i, 3) : card(s, recipe.material);
    const before = structuredClone(s);
    for (const d of levels) {
      const commands = options(s, d);
      assert.ok(commands.some((c) => c.type === 'skip-synthesis'));
      const synth = commands.filter((c) => c.type === 'synthesize' && c.recipeId === recipe.id);
      assert.ok(synth.length, `${d}/${recipe.id}`);
      assert.ok(commands.every((c) => isLegal(s, c)));
      assert.ok(synth.every((c) => new Set(c.materialIds).size === 3));
    }
    assert.deepEqual(s, before);
    s.phase = 'play';
    s.summonSlots = 0;
    assert.ok(options(s, 'hard').every((c) => c.type !== 'synthesize' && c.type !== 'craft'));
  }
});

test('2.5 fusion search compares deterministic pre-draw outcomes: merge depleted huts but retain three highly upgraded fighters', () => {
  for (const d of levels) {
    const s = fixture();
    s.phase = 'synthesis';
    s.summonSlots = 2;
    for (let i = 0; i < 3; i++) {
      const u = add(s, 'u22', 1, 2 + i, 3);
      u.hp = 1;
      u.maxHp = 5;
    }
    const before = fingerprint(s);
    const chosen = decide(observe(s), 1, d, { simulations: 400, milliseconds: 100000 });
    assert.equal(chosen.command?.type, 'synthesize', JSON.stringify(chosen.command));
    assert.equal(chosen.command?.recipeId, 'citadel');
    assert.equal(fingerprint(s), before);
    const hidden = structuredClone(s);
    hidden.seed = 123456;
    hidden.rng = 987654;
    assert.deepEqual(
      decide(observe(hidden), 1, d, { simulations: 400, milliseconds: 100000 }).command,
      chosen.command,
    );
    const keep = fixture();
    keep.phase = 'synthesis';
    keep.summonSlots = 2;
    for (let i = 0; i < 3; i++) {
      const u = add(keep, 'u8', 1, 2 + i, 3);
      u.maxHp = u.hp = 150;
      u.attackBonus = 80;
      u.kills = 10;
    }
    const retained = decide(observe(keep), 1, d, { simulations: 400, milliseconds: 100000 });
    assert.equal(retained.command?.type, 'skip-synthesis', JSON.stringify(retained.command));
  }
});

test('2.5 AI shares innate piercing, half-speed charge and automatic firelord target geometry with the engine', () => {
  const s = fixture(),
    killer = add(s, 'slayer', 1, 3, 4),
    screen = add(s, 'grave', 2, 3, 5),
    behind = add(s, 'grave', 2, 3, 7),
    side = add(s, 'grave', 2, 4, 6);
  assert.ok(Number.isFinite(hitDistance(s, killer, asTarget(behind))));
  assert.equal(Number.isFinite(hitDistance(s, killer, asTarget(side))), false);
  const attacks = options(s, 'medium').filter((c) => c.type === 'attack' && c.unitId === killer.id);
  assert.ok(attacks.some((c) => c.targetId === behind.id));
  assert.ok(attacks.every((c) => isLegal(s, c)));
  const slow = fixture(),
    hook = add(slow, 'formless', 1, 3, 4),
    victim = add(slow, 'grave', 2, 3, 9);
  for (const d of levels) {
    assert.ok(
      options(slow, d).some(
        (c) => c.type === 'charge' && c.mode === 'attack' && c.unitId === hook.id,
      ),
    );
    assert.ok(!options(slow, d).some((c) => c.type === 'attack' && c.unitId === hook.id));
  }
  const passive = fixture(),
    lord = add(passive, 'firelord', 1, 3, 4),
    primary = add(passive, 'grave', 2, 5, 7),
    splash = add(passive, 'grave', 2, 5, 8),
    diagonal = add(passive, 'grave', 2, 6, 8);
  primary.maxHp = primary.hp = 120;
  assert.equal(getStats(passive, lord).remaining, 0);
  assert.equal(firelordStrike(passive, lord)?.target.id, primary.id);
  assert.equal(attackPressure(passive, lord, asTarget(primary)), 80);
  assert.equal(attackPressure(passive, lord, asTarget(splash)), 10);
  assert.equal(attackPressure(passive, lord, asTarget(diagonal)), 0);
});

test('2.5 AI owns optional on-hit pulls and mandatory citadel placement without emitting an illegal skip', () => {
  for (const d of levels) {
    const s = fixture(),
      hook = add(s, 'formless', 1, 3, 4),
      target = add(s, 'grave', 2, 3, 9);
    hook.charge = hook.readyCharge = 1;
    hook.chargeType = 'attack';
    const pending = applyCommand(s, { type: 'attack', unitId: hook.id, targetId: target.id });
    const commands = options(pending, d);
    assert.ok(commands.some((c) => c.type === 'react' && c.mode === 'pull'));
    assert.ok(commands.some((c) => c.type === 'react' && !c.mode));
    assert.ok(commands.every((c) => isLegal(pending, c)));
    const city = fixture(),
      source = add(city, 'citadel', 1, 3, 4);
    city.pending = [{ kind: 'hut-spawn', owner: 1, source: structuredClone(source), amount: 0 }];
    const spawns = options(city, d);
    assert.ok(spawns.length);
    assert.ok(spawns.every((c) => c.x !== undefined && isLegal(city, c)));
  }
});
