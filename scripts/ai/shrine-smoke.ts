/** 神龛无界面验收使用真实引擎 RNG，AI 只读取公开观察。 */
import assert from 'node:assert/strict';
import { writeFileSync, mkdirSync } from 'node:fs';
import { createGame, createSession, validState, type Kind, type Command } from '../../src/engine';
import { Arena } from '../../src/match/arena';
import { observe, decisionOwner } from '../../src/ai/observation';
import { decide } from '../../src/ai/planning/search';
import { prepareTranscript, appendEntry } from './cli/transcript';
import type { Difficulty } from '../../src/ai/types';
const record = process.argv.includes('--record');
mkdirSync('artifacts', { recursive: true });
const report = [];
let seed = 1;
while (
  !createGame(seed, 'shrine').shrineDraft!.offers[1].includes('s8') ||
  !createGame(seed, 'shrine').shrineDraft!.offers[2].includes('s13')
)
  seed++;
for (const difficulty of ['easy', 'medium', 'hard'] as Difficulty[]) {
  const arena = new Arena(createSession(createGame(seed, 'shrine')));
  const path = 'docs/playtests/cli-feedback3-20260920.jsonl';
  if (record && difficulty === 'medium') prepareTranscript(path, arena.session, true);
  let commands = 0,
    simulations = 0,
    ended = 0;
  const act = (c: Command, ai = false) => {
    const entry = arena.play(c);
    assert.ok(validState(arena.session.present), `${difficulty}/${commands}: ${JSON.stringify(c)}`);
    commands++;
    if (c.type === 'end') ended++;
    if (record && difficulty === 'medium')
      appendEntry(path, { ...entry, actor: ai ? 'ai' : 'human' });
  };
  // 双方候选均由真实抽取得到；明确使用作者更正后的举旗坐标。
  act({ type: 'choose-shrine', player: 1, shrineKind: 's8' });
  act({ type: 'choose-shrine', player: 2, shrineKind: 's13' });
  act({ type: 'deploy', cardId: arena.session.present.hands[1][0].id, x: 5, y: 7 });
  act({ type: 'finish-shrine-setup' });
  act({ type: 'activate-aura', cardId: arena.session.present.hands[2][0].id });
  act({ type: 'finish-shrine-setup' });
  while (arena.session.present.ply < 9 && !arena.session.present.winner && commands < 180) {
    const s = arena.session.present,
      d = decide(observe(s), decisionOwner(s), difficulty, { mode: 'work', simulations: 240 });
    assert.ok(d.command, `${difficulty}: no command at ${s.ply}/${s.phase}`);
    simulations += d.stats.simulations;
    act(d.command, true);
  }
  assert.ok(arena.session.present.ply >= 9 || arena.session.present.winner, 'match stalled');
  report.push({
    difficulty,
    seed,
    commands,
    simulations,
    ended,
    ply: arena.session.present.ply,
    winner: arena.session.present.winner ?? null,
    valid: true,
  });
  console.log(JSON.stringify(report.at(-1)));
}
writeFileSync(
  'artifacts/shrine-smoke-report.json',
  JSON.stringify(
    {
      scope:
        '8 plies, fixed work budget, scripted public-offer choices; not a strength/win-rate evaluation',
      games: report,
    },
    null,
    2,
  ),
);
