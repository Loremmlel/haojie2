/** Synthetic diagnostic positions, not a reconstruction of the author's missing match replay. */
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { createGame, applyCommand, definition } from '../../src/engine';
import type { GameState } from '../../src/engine';
import { template } from '../../src/engine/state';
import { observe, decide } from '../../src/ai';
import type { Difficulty } from '../../src/ai';
import { allocateBudget, emptyBudget } from '../../src/ai/budget';
import { cardValue, explainEvaluation, materialValue } from '../../src/ai/evaluate';
import { analyzePayload, baseThreat } from '../../src/ai/threats';
import { actionWindow } from '../../src/ai/spatial';

function position(): GameState {
  const s = createGame(19);
  s.phase = 'play';
  s.summonSlots = 0;
  s.hands[1] = [
    {
      id: 'bomb',
      kind: 8,
      drawnAt: s.turns[1],
      summonedPly: s.ply,
      expiresAt: s.turns[1] + definition(8).spell!,
    },
  ];
  return s;
}
function decisions(s: GameState) {
  return (['easy', 'medium', 'hard'] as Difficulty[]).map((difficulty) => {
    const limits = allocateBudget(s, difficulty, emptyBudget());
    const result = decide(observe(s), 1, difficulty, { ...limits, trace: true });
    return { difficulty, limits, ...result };
  });
}
const bombs = ['opening', 'catapult', 'lethal', 'expiry'].map((name) => {
  const s = position();
  if (name === 'lethal') s.bases[2] = 20;
  if (name === 'expiry') s.hands[1][0].expiresAt = s.turns[1] + 1;
  if (name === 'catapult') s.units.push(template(10, 2, 0, { x: 5, y: 9 }, 'enemy-catapult'));
  const before = explainEvaluation(s, 1);
  const afterBase = explainEvaluation(
    applyCommand(s, { type: 'cast', cardId: 'bomb', x: 4, y: 12 }),
    1,
  );
  const afterCatapult =
    name === 'catapult'
      ? explainEvaluation(applyCommand(s, { type: 'cast', cardId: 'bomb', x: 4, y: 8 }), 1)
      : null;
  return {
    name,
    observation: observe(s),
    heldValue: cardValue(s, s.hands[1][0], 1),
    before,
    afterBase,
    afterCatapult,
    decisions: decisions(s),
  };
});
const immediate = position();
immediate.ply = 5;
immediate.turns = { 1: 3, 2: 2 };
immediate.hands[1][0] = { id: 'bomb', kind: 8, drawnAt: 3, summonedPly: 5, expiresAt: 7 };
immediate.bases[1] = 20;
immediate.units.push(template(11, 2, 0, { x: 5, y: 4 }, 'invader'));
const approaching = structuredClone(immediate);
approaching.hands[1] = [];
approaching.units[0].y = 6;
// Check the estimator's horizon. Moving spends the operation; it does NOT also attack now.
const move = { type: 'move' as const, unitId: 'invader', x: 5, y: 5 };
const moved = applyCommand(actionWindow(approaching, 2), move);
const afterAdvance = applyCommand(moved, { type: 'end' });
const defense = {
  immediate: observe(immediate),
  immediateThreat: baseThreat(immediate, 1),
  decisions: decisions(immediate),
  approaching: observe(approaching),
  currentThreat: baseThreat(approaching, 1),
  move,
  nextThreatAfterAdvance: baseThreat(afterAdvance, 1),
};
const conversion = position();
conversion.hands[1] = [];
const carrier = template(10, 1, 0, { x: 3, y: 4 }, 'carrier');
const victim = template('grave', 2, 0, { x: 4, y: 6 }, 'victim');
carrier.effects.push({
  type: 'convert',
  owner: 1,
  from: conversion.ply,
  until: conversion.ply + 2,
});
conversion.units.push(carrier, victim);
const estimate = analyzePayload(conversion, carrier, 'convert', (u) =>
  materialValue(conversion, u),
);
const conversionAfter = applyCommand(conversion, {
  type: 'attack',
  unitId: carrier.id,
  targetId: victim.id,
});
const conversionEstimate = {
  observation: observe(conversion),
  estimate,
  actualOwner: conversionAfter.units.find((u) => u.id === victim.id)!.owner,
  actualHp: conversionAfter.units.find((u) => u.id === victim.id)!.hp,
};
const output = process.argv[2] ?? 'artifacts/feedback-ai-20260916.json';
mkdirSync(dirname(output), { recursive: true });
writeFileSync(
  output,
  JSON.stringify(
    {
      label: 'Synthetic diagnostics; no win-rate or exact user-match reproduction claim',
      node: process.version,
      bombs,
      defense,
      conversionEstimate,
    },
    null,
    2,
  ) + '\n',
);
for (const c of bombs)
  console.log(
    JSON.stringify({
      name: c.name,
      heldValue: c.heldValue,
      baseGain: c.afterBase.total - c.before.total,
      catapultGain: c.afterCatapult ? c.afterCatapult.total - c.before.total : null,
      choices: c.decisions.map((d) => ({
        difficulty: d.difficulty,
        command: d.command,
        simulations: d.stats.simulations,
      })),
    }),
  );
console.log(
  JSON.stringify({
    immediateThreat: defense.immediateThreat,
    choices: defense.decisions.map((d) => ({ difficulty: d.difficulty, command: d.command })),
    approach: [defense.currentThreat, defense.nextThreatAfterAdvance],
    conversionProbability: estimate.targets[0]?.probability,
    actualOwner: conversionEstimate.actualOwner,
    output,
  }),
);
