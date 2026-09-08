/** Small reproducible qualification tournament, not a rating claim. */
import { writeFileSync, mkdirSync } from 'node:fs';
import { createGame, applyCommand } from '../../src/engine';
import { observe, fingerprint, decisionOwner } from '../../src/ai/observation';
import { decide } from '../../src/ai/search';
import type { Difficulty, PlanStep } from '../../src/ai/types';
const pair = (process.argv[2] ?? 'medium,easy').split(',') as [Difficulty, Difficulty];
if (pair.length !== 2 || pair.some((d) => !['easy', 'medium', 'hard'].includes(d)))
  throw new Error('Use easy,medium,hard: npm run bench:ai -- hard,medium');
const maxPlies = Number(process.env.AI_BENCH_PLIES ?? 40);
if (!Number.isSafeInteger(maxPlies) || maxPlies < 2 || maxPlies > 200)
  throw new Error('AI_BENCH_PLIES must be 2–200');
const games = [];
for (const seed of process.env.AI_BENCH_QUICK ? [7] : [7, 42])
  for (const swap of [false, true]) {
    const players = swap ? [pair[1], pair[0]] : pair;
    let s = createGame(seed),
      commands = 0,
      cache: PlanStep[] = [],
      simulations = 0,
      decisions = 0,
      replySearches = 0,
      maxMs = 0;
    const start = performance.now();
    while (!s.winner && s.ply <= maxPlies && commands < 700) {
      const side = decisionOwner(s),
        d = players[side - 1];
      let command;
      if (cache[0]?.before === fingerprint(s)) {
        command = cache.shift()!.command;
      } else {
        const t = performance.now();
        const result = decide(observe(s), side, d, {
          simulations: d === 'hard' ? 1400 : d === 'medium' ? 600 : 200,
          milliseconds: 100000,
          mode: 'work',
        });
        maxMs = Math.max(maxMs, performance.now() - t);
        simulations += result.stats.simulations;
        replySearches += result.stats.replies;
        decisions++;
        command = result.command;
        cache = result.plan.slice(1);
      }
      if (!command) throw new Error(`No legal command at seed${seed}, ply${s.ply}`);
      s = applyCommand(s, command);
      commands++;
    }
    const record = {
      seed,
      players,
      winner: s.winner ?? 'ply-limit',
      winningLevel: s.winner && s.winner !== 'draw' ? players[s.winner - 1] : null,
      plies: s.ply,
      commands,
      bases: s.bases,
      simulations,
      decisions,
      replySearches,
      maxDecisionMs: Math.round(maxMs),
      elapsedMs: Math.round(performance.now() - start),
    };
    games.push(record);
    console.log(JSON.stringify(record));
  }
mkdirSync('artifacts', { recursive: true });
writeFileSync(
  `artifacts/benchmark-${pair.join('-')}.json`,
  JSON.stringify(
    { mode: 'fixed-node qualification, seeds swapped; not an Elo claim', maxPlies, games },
    null,
    2,
  ),
);
