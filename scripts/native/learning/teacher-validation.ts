import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { nativeClient } from '../engine/validation/client';
import { createGame, applyRuntimeCommand } from '../../../src/engine/commands/game';
import { observe, decisionOwner } from '../../../src/ai/observation';
import { decide } from '../../../src/ai/planning/search';
import { allocateBudget, emptyBudget } from '../../../src/ai/budget';
import { cachedDecision } from '../../../src/ai/planning/plan-cache';
import type { Player } from '../../../src/engine/types';
import type { PlanStep } from '../../../src/ai/types';

const native = await nativeClient(
  process.argv[2] ??
    resolve(
      'artifacts/native-target/release',
      process.platform === 'win32' ? 'haojie-engine.exe' : 'haojie-engine',
    ),
);
const steps = Number(process.argv[3] ?? 64);
let checked = 0;
try {
  for (const difficulty of ['easy', 'medium', 'hard'] as const) {
    if (process.argv[4] && process.argv[4] !== difficulty) continue;
    for (const seed of [731, 719]) {
      if (process.argv[5] && Number(process.argv[5]) !== seed) continue;
      let state = createGame(seed),
        record: unknown = null;
      const budgets = { 1: emptyBudget(), 2: emptyBudget() };
      const plans: Record<Player, PlanStep[]> = { 1: [], 2: [] };
      for (let step = 0; step < steps && !state.winner; step++) {
        const observation = observe(state),
          side = decisionOwner(observation);
        if (budgets[side].ply !== state.ply) {
          budgets[side] = { ...emptyBudget(), ply: state.ply };
          plans[side] = [];
        }
        const left =
          cachedDecision(observation, plans[side]) ??
          decide(observation, side, difficulty, allocateBudget(state, difficulty, budgets[side]));
        const right = await native.request({
          op: 'classic-teacher',
          observation,
          difficulty,
          record,
        });
        assert.deepEqual(
          right.decision,
          left,
          `${difficulty} seed=${seed} step=${step} ply=${state.ply}`,
        );
        record = right.record;
        plans[side] = left.plan.slice(1);
        budgets[side].nodes += left.stats.simulations;
        budgets[side].commands++;
        state = applyRuntimeCommand(state, left.command!);
        state.log = [];
        state.events = [];
        checked++;
      }
      console.log(
        JSON.stringify({ difficulty, seed, checked, ply: state.ply, winner: state.winner ?? null }),
      );
    }
  }
} finally {
  native.close();
}
