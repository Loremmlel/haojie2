import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { createHash } from 'node:crypto';
import { nativeClient } from '../../native/client';
import { freeze } from '../../native/sampling/artifacts';
import { fixtures } from '../../native/fixtures';
import { combatFixtures } from '../../native/combat-fixtures';
import { completeFixtures } from '../../native/complete-fixtures';
import { preparationFixtures } from '../../native/preparation-fixtures';
import type * as API from '../api';
import type { GameState, Player } from '../../../../src/engine/types';

const { values } = parseArgs({
  options: {
    output: { type: 'string' },
    baseline: { type: 'string' },
    executable: { type: 'string' },
    historical: { type: 'string' },
  },
});
assert.ok(values.output && values.baseline && values.executable && values.historical);
const output = resolve(values.output);
const old: typeof API = await import(pathToFileURL(resolve(values.baseline)).href);
const { executable, save } = await freeze(
  'scripts/training/performance/runtime/collect.ts',
  output,
  values.executable,
);
const canonical = <T>(v: T): T => JSON.parse(JSON.stringify(v));
const workset: any[] = [];
const started = performance.now();
const client = await nativeClient(executable, 900_000);
try {
  for (const [rules, seed, historical] of [
    ['classic', 731270001, true],
    ['shrine', 741270001, true],
    ['classic', 731270017, false],
    ['shrine', 741270019, false],
  ] as const) {
    const start = performance.now();
    let decisions: any[] = [],
      native: any;
    if (historical) {
      for await (const row of old.readTrainingRecords(join(values.historical, `${rules}.jsonl.gz`)))
        if (row.type === 'decision') decisions.push({ actor: row.actor, command: row.command });
    } else {
      native = await client.request({
        op: 'sample-game',
        seed,
        rules,
        maxCommands: 12000,
        maxPlies: 500,
        policy: 'tiny',
      });
      save(`${rules}-${seed}-native.json`, native);
      assert.equal(native.error, null);
      decisions = native.commands;
    }
    const positions = new Set([0, 10, 50, 100, 250, 500, 750, 1000]);
    for (let i = 0; i < 16; i++) positions.add(Math.floor(((i + 0.5) * decisions.length) / 16));
    let state = old.createGame(seed, rules),
      largest: any;
    state.log = [];
    state.events = [];
    const take = (index: number) => ({
      path: `${rules}:${seed}`,
      index,
      rules,
      seed,
      actor: decisions[index].actor,
      state: canonical(state),
      observation: old.observe(state, decisions[index].actor),
    });
    for (const [index, { actor, command }] of decisions.entries()) {
      if (positions.has(index)) workset.push(take(index));
      if (
        !largest ||
        state.units.length + state.deaths.length >
          largest.state.units.length + largest.state.deaths.length
      )
        largest = take(index);
      state = old.applyPlayerCommand(state, actor, command);
      state.log = [];
      state.events = [];
    }
    if (largest && !positions.has(largest.index)) workset.push(largest);
    if (native) {
      assert.deepEqual(canonical(state), native.state);
      assert.deepEqual(
        [1, 2].map((p) => canonical(old.observe(state, p as Player))),
        native.observations,
      );
    }
    save(`${rules}-${seed}.json`, {
      seed,
      rules,
      decisions,
      state,
      collectionMs: performance.now() - start,
    });
    console.log(
      JSON.stringify({
        rules,
        seed,
        commands: decisions.length,
        ply: state.ply,
        winner: state.winner ?? null,
        collectionMs: performance.now() - start,
      }),
    );
    assert.ok(performance.now() - started < 7_200_000, '采集累计超时');
  }
} finally {
  client.close();
}
const cases = [...fixtures(), ...combatFixtures(), ...completeFixtures(), ...preparationFixtures()];
const selected = new Map<string, GameState>();
for (const [label, accept] of Object.entries({
  reaction: (s: GameState) => s.pending.length > 0,
  stack: (s: GameState) =>
    s.units.some((u, i) => s.units.slice(i + 1).some((v) => u.x === v.x && u.y === v.y)),
  giant: (s: GameState) => s.units.some((u) => u.size > 1),
  landmark: (s: GameState) => !!s.landmarks?.length,
  death: (s: GameState) => s.deaths.length > 0,
  clock: (s: GameState) => !!s.clockFrames,
})) {
  let count = 0;
  for (const c of cases)
    if (accept(c.job.state) && count++ < 2) selected.set(`${label}:${c.name}`, c.job.state);
}
for (const [name, state] of selected) {
  const actor = old.decisionOwner(state);
  workset.push({
    path: `fixture:${name}`,
    index: 0,
    seed: state.seed,
    rules: state.mode ?? 'classic',
    actor,
    state,
    observation: old.observe(state, actor),
  });
}
save('workset.json', workset);
const hash = (p: string) => createHash('sha256').update(readFileSync(p)).digest('hex');
save('workset-manifest.json', {
  baseline: resolve(values.baseline),
  baselineSha256: hash(values.baseline),
  worksetSha256: hash(join(output, 'workset.json')),
  cases: workset.map((r) => ({
    path: r.path,
    index: r.index,
    ply: r.state.ply,
    units: r.state.units.length,
    deaths: r.state.deaths.length,
  })),
  collectionMs: performance.now() - started,
  resource: process.resourceUsage(),
});
