import test from 'node:test';
import assert from 'node:assert/strict';
import { applyCommand, createSession, parseSession } from '../../src/engine';
import { addEffect, emit } from '../../src/engine/state';
import { eventActor, withEventFacts } from '../../src/engine/event-facts';
import {
  planEffects,
  center,
  appendBatch,
  MAX_BATCHES,
  MAX_CUES,
  type EffectBatch,
} from '../../src/ui/board/vfx/plan';
import { vfxScenarios } from '../../scripts/vfx/scenarios';
import { add, fixture } from '../helpers';
const scenarios = vfxScenarios();
const scenario = (name: string) => scenarios.find((s) => s.name === name)!;
const effects = (name: string) => planEffects(scenario(name).after.events);

test('attack identity separates contact, arrows, heavy shots and friendly healing without rule guesses', () => {
  for (const [name, family] of [
    ['近身斩击', 'slash'],
    ['射手箭矢', 'arrow'],
    ['赤焰重炮', 'cannon'],
    ['治疗流光', 'mend'],
  ])
    assert.equal(effects(name)[0].family, family);
  const ranged = fixture(),
    a = add(ranged, 26, 1, 3, 5),
    b = add(ranged, 13, 2, 3, 8);
  assert.equal(
    planEffects(applyCommand(ranged, { type: 'attack', unitId: a.id, targetId: b.id }).events)[0]
      .family,
    'bolt',
  );
  const arrow = effects('射手箭矢')[0];
  for (const p of scenario('射手箭矢').after.events.find((e) => e.type === 'attack')!.path!)
    assert.ok(arrow.route.some((v) => v.x === center(p).x && v.y === center(p).y));
  assert.deepEqual(arrow.to, center(scenario('射手箭矢').before.units[1], 2));
});

test('effect facts are bounded identity snapshots and survive source death, movement and conversion', () => {
  const hook = structuredClone(scenario('钩子牵引')),
    move = hook.after.events.find((e) => e.type === 'move')!;
  assert.equal(move.actor!.kind, 7);
  assert.deepEqual(move.from, { x: 5, y: 5 });
  assert.deepEqual(move.to, { x: 4, y: 6 });
  assert.equal(move.subject!.x, 5);
  assert.equal('hp' in move.subject!, false);
  hook.after.units[1].x = 8;
  assert.equal(move.subject!.x, 5);
  const death = scenario('处决触发').after.events.find((e) => e.type === 'death')!;
  assert.equal(death.subject!.size, 2);
  assert.equal(death.action, 'execution');
  const converted = scenario('策反兑现').after.events.find((e) => e.action === 'conversion')!;
  assert.equal(converted.stage, 'trigger');
  assert.equal(converted.subject!.owner, 1);
});

test('spell areas are exact board cells, including delayed storm, and immunity never reports success', () => {
  assert.equal(effects('爆弹区域')[0].area.length, 4);
  assert.equal(effects('烈焰风暴')[0].area.length, 9);
  assert.equal(effects('风暴再临')[0].area.length, 13);
  const cross = scenario('十字浩劫').after.events.find((e) => e.action === 'cross' && e.area)!;
  assert.ok(cross.area!.every((p) => p.x === 5 || p.y === 7));
  assert.ok(cross.area!.every((p) => p.x >= 1 && p.x <= 9 && p.y >= 1 && p.y <= 10));
  assert.ok(effects('法术反制').every((c) => c.family === 'counter' && !c.area.length));
  assert.ok(effects('金身挡下').some((c) => c.family === 'ward' && c.stage === 'blocked'));
  assert.ok(!effects('金身挡下').some((c) => c.family === 'damage'));
  const blocked = fixture(),
    mage = add(blocked, 'u14', 1, 3, 5),
    protectedUnit = add(blocked, 26, 2, 5, 5),
    friend = add(blocked, 13, 1, 3, 7);
  addEffect(blocked, protectedUnit, 'immune', 2, 0, 2);
  const link = applyCommand(blocked, {
    type: 'skill',
    unitId: mage.id,
    targetId: protectedUnit.id,
    secondId: friend.id,
  });
  assert.equal(link.siphons.length, 0);
  assert.ok(!planEffects(link.events).some((c) => c.family === 'siphon'));
  for (const [apply, trigger, family] of [
    ['策反施加', '策反兑现', 'conversion'],
    ['处决施加', '处决触发', 'execution'],
  ]) {
    assert.ok(effects(apply).some((c) => c.family === family && c.stage === 'apply'));
    assert.ok(effects(trigger).some((c) => c.family === family && c.stage === 'trigger'));
  }
});

test('hit numbers share a causal impact, drain follows actual healing and reaction attacks remain nested', () => {
  const shot = effects('赤焰重炮'),
    main = shot.find((c) => c.family === 'cannon')!;
  assert.ok(shot.filter((c) => c.family === 'damage').every((c) => c.start === main.impact));
  const drain = effects('吸血回流'),
    flow = drain.find((c) => c.family === 'siphon')!,
    heal = drain.find((c) => c.family === 'heal')!;
  assert.equal(flow.impact, heal.start);
  assert.notDeepEqual(flow.from, flow.to);
  const s = fixture(),
    a = add(s, 26, 1, 4, 6),
    king = add(s, 'u18', 2, 5, 6);
  const next = applyCommand(s, { type: 'attack', unitId: a.id, targetId: king.id });
  const attacks = next.events.filter((e) => e.type === 'attack');
  assert.equal(attacks.length, 2);
  assert.equal(attacks[1].parentId, attacks[0].causeId);
  const plans = planEffects(next.events).filter((c) => c.family === 'slash');
  assert.ok(plans[1].start >= plans[0].impact);
});

test('presentation planning is pure, deterministic, uses no RNG and preserves old saves', () => {
  for (const sample of scenarios) {
    const before = JSON.stringify(sample.after);
    assert.deepEqual(planEffects(sample.after.events), planEffects(sample.after.events));
    assert.equal(JSON.stringify(sample.after), before);
    assert.doesNotThrow(() => parseSession(JSON.stringify(createSession(sample.after))));
  }
  const old = {
    id: 'old',
    type: 'attack' as const,
    from: { x: 3, y: 5 },
    to: { x: 3, y: 8 },
    owner: 1 as const,
  };
  assert.equal(planEffects([old])[0].family, 'bolt');
  const s = fixture(),
    a = add(s, 26, 1, 3, 5),
    serial = s.serial,
    rng = s.rng;
  assert.throws(() =>
    withEventFacts(s, { action: 'bomb', actor: eventActor(a) }, () => {
      throw new Error('abort');
    }),
  );
  emit(s, { type: 'skill', to: a });
  assert.equal(s.events.at(-1)!.action, undefined);
  assert.equal(s.serial, serial + 1);
  assert.equal(s.rng, rng);
});

test('effect batches coexist, expire independently and bound both batches and cues', () => {
  const make = (id: number, born: number, expires: number): EffectBatch => ({
    id,
    born,
    expires,
    cues: effects('近身斩击'),
  });
  let batches = appendBatch([], make(1, 0, 1000), 0);
  batches = appendBatch(batches, make(2, 300, 1300), 300);
  assert.deepEqual(
    batches.map((b) => b.id),
    [1, 2],
  );
  assert.notEqual(
    batches[0].cues.find((c) => c.family === 'damage')!.numberSlot,
    batches[1].cues.find((c) => c.family === 'damage')!.numberSlot,
  );
  batches = appendBatch(batches, make(3, 1050, 2000), 1050);
  assert.deepEqual(
    batches.map((b) => b.id),
    [2, 3],
  );
  for (let i = 4; i < 30; i++) batches = appendBatch(batches, make(i, 1100, 2400), 1100);
  assert.ok(batches.length <= MAX_BATCHES);
  assert.ok(batches.reduce((n, b) => n + b.cues.length, 0) <= MAX_CUES);
  const storm = Array.from({ length: 400 }, (_, i) => ({
    id: String(i),
    type: 'damage' as const,
    to: { x: 3, y: 5 },
    amount: 1,
  }));
  assert.ok(planEffects(storm).length <= MAX_CUES);
});
