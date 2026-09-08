/** Production-path telemetry. Replays and cache executions are not fresh depth-zero searches. */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { replayTranscript } from './cli/transcript';
import { gunzipSync } from 'node:zlib';
import { createGame, applyCommand } from '../../src/engine';
import type { GameState } from '../../src/engine';
import { decide as currentDecide } from '../../src/ai/search';
import { observe, decisionOwner, fingerprint } from '../../src/ai/observation';
import { allocateBudget as currentBudget, emptyBudget } from '../../src/ai/budget';
import type { Difficulty, PlanStep } from '../../src/ai/types';
const output = process.argv[2] ?? 'artifacts/production-audit.json';
// Optional trusted frozen bundle must export BOTH the planner and its original production allocator.
const baseline = process.argv[3]
  ? ((await import(pathToFileURL(resolve(process.argv[3])).href)) as {
      decide: typeof currentDecide;
      allocateBudget: typeof currentBudget;
    })
  : undefined;
if (
  baseline &&
  (typeof baseline.decide !== 'function' || typeof baseline.allocateBudget !== 'function')
)
  throw new Error(
    'Audit baseline must export decide AND allocateBudget; never silently use new budgets for old code.',
  );
const decide = baseline?.decide ?? currentDecide,
  allocateBudget = baseline?.allocateBudget ?? currentBudget;
const rows = gunzipSync(readFileSync('docs/playtests/cli-human-20260907.jsonl.gz'))
  .toString()
  .trim()
  .split('\n')
  .map((x) => JSON.parse(x));
replayTranscript(rows.map((r) => JSON.stringify(r)).join('\n'));
let state: GameState = rows[0].initial.present;
const snapshots: GameState[] = [];
for (const row of rows.slice(1)) {
  if (
    state.phase === 'play' &&
    state.active === 2 &&
    !state.pending.length &&
    !snapshots.some((s) => s.ply === state.ply)
  )
    snapshots.push(state);
  state = applyCommand(state, row.command);
}
const positions = [];
for (const s of snapshots.slice(0, 9))
  for (const difficulty of ['medium', 'hard'] as Difficulty[]) {
    const limits = allocateBudget(s, difficulty, emptyBudget());
    const start = performance.now();
    const decision = decide(observe(s), decisionOwner(s), difficulty, { ...limits, trace: true });
    const ms = performance.now() - start;
    const record = {
      ply: s.ply,
      units: s.units.length,
      difficulty,
      limits,
      ms: Math.round(ms),
      command: decision.command,
      selectedDepth: decision.plan.length,
      ...decision.stats,
      adopted: decision.trace?.alternatives.filter((a) => a.stage === 'reply').length ?? 0,
    };
    positions.push(record);
    console.log(JSON.stringify(record));
  }
const commands = [];
if (!process.env.AI_AUDIT_POSITIONS_ONLY) {
  let s = createGame(7),
    caches: Record<number, PlanStep[]> = { 1: [], 2: [] },
    budgets = { 1: emptyBudget(), 2: emptyBudget() };
  while (!s.winner && s.ply <= 12 && commands.length < 300) {
    const owner = decisionOwner(s),
      before = fingerprint(s);
    if (budgets[owner].ply !== s.ply) budgets[owner] = { ...emptyBudget(), ply: s.ply };
    const cached = caches[owner][0]?.before === before;
    const limits = allocateBudget(s, 'hard', budgets[owner]);
    const start = performance.now();
    const d = cached
      ? {
          command: caches[owner][0].command,
          plan: caches[owner],
          stats: {
            simulations: 0,
            candidates: 0,
            depth: 0,
            replies: 0,
            sampled: 0,
            exhausted: false,
          },
        }
      : decide(observe(s), owner, 'hard', { ...limits, trace: true });
    const ms = performance.now() - start;
    budgets[owner].nodes += d.stats.simulations;
    budgets[owner].ms += ms;
    budgets[owner].commands++;
    caches[owner] = d.plan.slice(1);
    if (!d.command) throw new Error('No command at ' + s.ply);
    commands.push({
      ply: s.ply,
      phase: s.phase,
      units: s.units.length,
      cached,
      selectedDepth: d.plan.length,
      ms: Math.round(ms),
      limits,
      ...d.stats,
      adopted:
        'trace' in d ? (d.trace?.alternatives.filter((a) => a.stage === 'reply').length ?? 0) : 0,
      command: d.command,
    });
    s = applyCommand(s, d.command);
  }
}
mkdirSync(dirname(output), { recursive: true });
writeFileSync(
  output,
  JSON.stringify(
    {
      label: 'Same replay positions plus seed-7 production self-play; timings are machine-specific',
      baseline: process.argv[3] ?? null,
      node: process.version,
      platform: process.platform,
      positions,
      commands,
    },
    null,
    2,
  ),
);
console.log(JSON.stringify({ output, positions: positions.length, commands: commands.length }));
