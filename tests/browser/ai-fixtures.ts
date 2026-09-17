import { mkdirSync, writeFileSync } from 'node:fs';
import { fixture, add, card } from '../helpers';
import { createDemoGame, createSession } from '../../src/engine';
import type { GameState } from '../../src/engine';
import type { MatchSettings } from '../../src/match/settings';
mkdirSync('artifacts/ai-fixtures', { recursive: true });
const save = (name: string, state: GameState, match: MatchSettings) =>
  writeFileSync(`artifacts/ai-fixtures/${name}.json`, JSON.stringify(createSession(state, match)));
const hard: MatchSettings = { mode: 'ai', human: 2, difficulty: 'hard' };
save('thinking', createDemoGame(), hard);
const s = fixture();
s.heads = { 1: 0, 2: 0 };
add(s, 9, 2, 5, 4);
add(s, 1, 1, 2, 3);
save('response', s, { mode: 'ai', human: 1, difficulty: 'medium' });
const r = fixture(),
  healer = add(r, 2, 2, 3, 5);
r.units = [];
add(r, 1, 1, 3, 4);
r.pending.push({ kind: 'death-shot', owner: 2, source: healer, amount: 20 });
save('human-reaction', r, hard);

// A predictable repeated attack sequence, used to observe public board pacing and cancellation.
const paced = fixture();
paced.heads = { 1: 0, 2: 0 };
const archer = add(paced, 23, 1, 4, 7);
archer.born = 1; // Independent of deployment fatigue: this is its active six-shot turn.
add(paced, 'grave', 2, 4, 8);
save('pacing', paced, { mode: 'ai', human: 2, difficulty: 'medium' });

// Exact sanitized CLI failure position, before its mistaken deploy -> cached END plan.
import endTurnPosition from '../fixtures/ai/end-turn-20260915.json';
import { imagined } from '../../src/ai/observation';
import type { Observation } from '../../src/ai/types';
save('end-turn-value', imagined(endTurnPosition as Observation), {
  mode: 'ai',
  human: 1,
  difficulty: 'hard',
});
const fusion = fixture();
fusion.phase = 'synthesis';
fusion.summonSlots = 2;
for (let i = 0; i < 3; i++) {
  const hut = add(fusion, 'u22', 1, 2 + i, 3);
  hut.hp = 1;
  hut.maxHp = 5;
}
save('synthesis', fusion, hard);
