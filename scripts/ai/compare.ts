/** Paired fixed-seed baseline comparison. The old planner is an explicit frozen ESM bundle.
 * Never adjudicate an unfinished match by health/material. Transcript includes actual commands. */
import { pathToFileURL } from 'node:url';
import { mkdirSync, writeFileSync, readFileSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { createGame, createSession, applyCommand, commandError } from '../../src/engine';
import { decide } from '../../src/ai/search';
import { allocateBudget, emptyBudget } from '../../src/ai/budget';
import { observe, fingerprint, decisionOwner } from '../../src/ai/observation';
import type { Decision, Difficulty, PlanStep } from '../../src/ai/types';
const path = process.argv[2];
if (!path) throw new Error('Usage: tsx scripts/ai/compare.ts /absolute/path/to/baseline.mjs');
const baseline = (await import(pathToFileURL(path).href)) as {
  decide: typeof decide;
  allocateBudget?: typeof allocateBudget;
};
const production = process.env.AI_BUDGET === 'production';
if (process.env.AI_BUDGET && !['production', 'fixed'].includes(process.env.AI_BUDGET))
  throw new Error('AI_BUDGET must be production or fixed');
if (production && !baseline.allocateBudget)
  throw new Error('Production baseline must export its own allocateBudget');
const seeds = (process.env.AI_SEEDS ?? '7,42,20260907').split(',').map(Number);
const nodes = Number(process.env.AI_NODES ?? 700),
  maxPlies = Number(process.env.AI_PLIES ?? 60);
const prefix = process.env.AI_REPORT ?? 'artifacts/comparison';
const levels = (process.env.AI_LEVELS ?? 'hard,hard').split(',') as [Difficulty, Difficulty];
if (
  !seeds.every(Number.isSafeInteger) ||
  !Number.isInteger(nodes) ||
  nodes < 40 ||
  !Number.isInteger(maxPlies) ||
  maxPlies < 2 ||
  levels.length !== 2 ||
  !levels.every((v) => ['easy', 'medium', 'hard'].includes(v))
)
  throw new Error('Invalid comparison configuration');
mkdirSync('artifacts', { recursive: true });
const digest = createHash('sha256');
for (const f of readdirSync('src/ai')
  .filter((f) => f.endsWith('.ts'))
  .sort())
  digest.update(f).update(readFileSync('src/ai/' + f));
const sourceDigest = digest.digest('hex');
const results: unknown[] = [];
for (const seed of seeds)
  for (const newSide of [1, 2] as const) {
    let s = createGame(seed),
      cache: Record<1 | 2, PlanStep[]> = { 1: [], 2: [] },
      count = 0,
      turn = -1,
      used = 0,
      maxDecisionMs = 0;
    const budgets = { 1: emptyBudget(), 2: emptyBudget() };
    const transcript: unknown[] = [{ format: 'haojie-cli-v1', initial: createSession(s) }];
    const started = performance.now();
    let failure: string | undefined;
    while (!s.winner && s.ply <= maxPlies && count < 2000) {
      if (turn !== s.ply) {
        turn = s.ply;
        used = 0;
        if (turn % 10 === 0)
          console.log(
            JSON.stringify({
              progress: true,
              seed,
              newSide,
              ply: s.ply,
              units: s.units.length,
              bases: s.bases,
              commands: count,
            }),
          );
      }
      const owner = decisionOwner(s),
        before = fingerprint(s),
        isNew = owner === newSide;
      if (budgets[owner].ply !== s.ply) budgets[owner] = { ...emptyBudget(), ply: s.ply };
      const timed = performance.now(),
        predicted = cache[owner][0];
      let result: Decision;
      try {
        if (predicted?.before === before)
          result = {
            command: predicted.command,
            plan: cache[owner],
            stats: {
              simulations: 0,
              candidates: 0,
              depth: 0,
              replies: 0,
              sampled: 0,
              exhausted: false,
            },
          };
        else
          result = (isNew ? decide : baseline.decide)(
            observe(s),
            owner,
            levels[isNew ? 0 : 1],
            production
              ? (isNew ? allocateBudget : baseline.allocateBudget!)(
                  s,
                  levels[isNew ? 0 : 1],
                  budgets[owner],
                )
              : {
                  simulations: Math.max(100, Math.min(nodes, nodes * 6 - used)),
                  milliseconds: 100000,
                  mode: 'work',
                },
          );
        const elapsed = performance.now() - timed;
        budgets[owner].nodes += result.stats.simulations;
        budgets[owner].ms += elapsed;
        budgets[owner].commands++;
        maxDecisionMs = Math.max(maxDecisionMs, performance.now() - timed);
        used += result.stats.simulations;
        if (!result.command) throw new Error(`No command at ${s.ply}`);
        const err = commandError(s, result.command);
        if (err) throw new Error(err);
        cache[owner] = result.plan.slice(1);
        const next = applyCommand(s, result.command);
        transcript.push({
          actor: isNew ? 'new' : 'baseline',
          owner,
          ply: s.ply,
          command: result.command,
          before,
          after: fingerprint(next),
          events: next.events,
          stats: result.stats,
          milliseconds: Math.round(elapsed),
        });
        s = next;
        count++;
      } catch (e) {
        failure = String(e);
        break;
      }
    }
    const result = {
      sourceDigest,
      seed,
      newSide,
      levels,
      nodes,
      maxPlies,
      budgetMode: production ? 'production' : 'fixed',
      winner: s.winner ?? 'unresolved',
      newWon: s.winner === newSide,
      ply: s.ply,
      commands: count,
      bases: s.bases,
      heads: s.heads,
      units: s.units.length,
      milliseconds: Math.round(performance.now() - started),
      maxDecisionMs: Math.round(maxDecisionMs),
      ...(failure ? { failure } : {}),
    };
    if (failure) process.exitCode = 1;
    results.push(result);
    writeFileSync(
      `${prefix}-${seed}-${newSide}.jsonl`,
      transcript.map((r) => JSON.stringify(r)).join('\n') + '\n',
    );
    writeFileSync(
      `${prefix}-report.json`,
      JSON.stringify(
        {
          note: 'Paired diagnostic with explicit budget mode, not calibrated Elo or human-author parity. Unfinished games are unresolved.',
          results,
        },
        null,
        2,
      ),
    );
    console.log(JSON.stringify(result));
  }
