import assert from 'node:assert/strict';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { createHash } from 'node:crypto';
import { nativeClient } from '../../native/client';
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
  },
});
assert.ok(values.output && values.baseline && values.executable);
const output = resolve(values.output);
mkdirSync(output, { recursive: true });
const save = (name: string, value: unknown) =>
  writeFileSync(join(output, name), JSON.stringify(value), { flag: 'wx' });
const old: typeof API = await import(pathToFileURL(resolve(values.baseline)).href);
const canonical = <T>(value: T): T => JSON.parse(JSON.stringify(value));
const workset: any[] = [];
const client = await nativeClient(resolve(values.executable), 1_800_000);
try {
  // 种子在实施前固定；超限保留真实前缀，不用换种子获得更短的终局。
  for (const [rules, seed] of [
    ['classic', 731270001],
    ['shrine', 741270001],
    ['classic', 731270031],
    ['shrine', 741270037],
  ] as const) {
    const native = await client.request({
      op: 'sample-game',
      seed,
      rules,
      maxCommands: 12000,
      maxPlies: 500,
      policy: 'tiny',
    });
    save(`${rules}-${seed}-native.json`, native);
    assert.equal(native.error, null);
    const decisions = native.commands;
    const positions = new Set([0, 10, 50, 100, 250, 500, 750, 1000]);
    for (let i = 0; i < 16; i++) positions.add(Math.floor(((i + 0.5) * decisions.length) / 16));
    let state = old.createGame(seed, rules);
    state.log = [];
    state.events = [];
    let largest: any;
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
    assert.deepEqual(canonical(state), native.state);
    assert.deepEqual(
      [1, 2].map((p) => canonical(old.observe(state, p as Player))),
      native.observations,
    );
    save(`${rules}-${seed}.json`, { rules, seed, decisions, state });
    console.log(
      JSON.stringify({
        rules,
        seed,
        commands: decisions.length,
        ply: state.ply,
        winner: state.winner ?? null,
      }),
    );
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
save('manifest.json', {
  baselineSha256: hash(values.baseline),
  executableSha256: hash(values.executable),
  worksetSha256: hash(join(output, 'workset.json')),
  cases: workset.map((r) => ({
    path: r.path,
    index: r.index,
    ply: r.state.ply,
    units: r.state.units.length,
    deaths: r.state.deaths.length,
  })),
});
