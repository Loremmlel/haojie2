import { mkdirSync, writeFileSync } from 'node:fs';
import { fixture, add, card } from '../../helpers';
import { createDemoGame, createSession } from '../../../src/engine';
import type { GameState } from '../../../src/engine';
import type { MatchSettings } from '../../../src/match/settings';
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

// 可预测的重复攻击序列，用于观察公开棋盘节奏和取消行为。
const paced = fixture();
paced.heads = { 1: 0, 2: 0 };
const archer = add(paced, 23, 1, 4, 7);
archer.born = 1; // 与部署疲劳无关，这是其能攻击六次的实际行动回合。
const barricade = add(paced, 'wall', 2, 4, 8);
barricade.hp = barricade.maxHp = 70;
paced.log = ['5 · 独行侠准备攻击路障'];
save('pacing', paced, { mode: 'ai', human: 2, difficulty: 'medium' });

// 精确脱敏的 CLI 失败局面，位于错误的“部署后缓存结束回合”计划之前。
import endTurnPosition from '../../fixtures/ai/end-turn-20260915.json';
import { imagined } from '../../../src/ai/observation';
import type { Observation } from '../../../src/ai/types';
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
