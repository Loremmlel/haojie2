/** 原失败局面上的 TS/Rust 采样差分；只用均匀替身，不计入模型产能。 */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { sampleCommand, emptyMetrics } from '../../economics/sample';
import { randomStream } from '../../economics/policy';
import { observe } from '../../../../src/ai/observation';
import { allPieces, hasTrait } from '../../../../src/engine/core/traits';
import { applyRuntimeCommand } from '../../../../src/engine/commands/game';
import { RuleError } from '../../../../src/engine/core/state';
import type { GameState, Player, Command } from '../../../../src/engine/types';
import { nativeClient } from '../../native/client';
import { nativeHash } from '../../native/pipeline/hash';

const [fixture, executable, output] = process.argv.slice(2);
const initial: GameState = JSON.parse(readFileSync(fixture, 'utf8')).state;
function run(seed: number) {
  let state = initial;
  const random = randomStream(seed),
    metrics = emptyMetrics();
  const commands: { actor: Player; command: Command }[] = [];
  let offer = true;
  for (let i = 0; i < 8 && state.winner === undefined; i++) {
    let actor = (state.pending[0]?.owner ?? state.active) as Player;
    let view = observe(state, actor);
    if (view.phase === 'shrine-draft' && view.shrineDraft?.committed[actor]) {
      actor = (3 - actor) as Player;
      view = observe(state, actor);
    }
    const select = (player: Player, optional: boolean) =>
      sampleCommand(
        observe(state, player),
        player,
        null,
        random,
        metrics,
        optional,
        (command, status) => {
          const before = nativeHash(state);
          try {
            state = applyRuntimeCommand(state, command);
            state.events = [];
            state.log = [];
            return true;
          } catch (error) {
            if (!(error instanceof RuleError) || status !== 'uncertain') throw error;
            assert.equal(nativeHash(state), before);
            return false;
          }
        },
      );
    const other = (3 - actor) as Player;
    let selected: ReturnType<typeof sampleCommand> | undefined;
    if (
      offer &&
      !view.pending.length &&
      view.phase !== 'shrine-draft' &&
      allPieces(state).some((u) => u.owner === other && hasTrait(u, 'u7'))
    ) {
      selected = select(other, true);
      if (selected.command) {
        actor = other;
        offer = false;
      }
    }
    if (!selected?.command) {
      selected = select(actor, false);
      offer = true;
    }
    assert.ok(selected.command, selected.error);
    commands.push({ actor, command: selected.command });
  }
  return { commands, metrics, state };
}
let seed = 1,
  reference = run(seed);
while (!reference.metrics.rejected && seed < 2048) reference = run(++seed);
assert.ok(reference.metrics.rejected, 'fixture must exercise authority rejection');
const client = await nativeClient(executable);
try {
  const result = await client.request({
    op: 'sample-game',
    seed: 1,
    rules: 'shrine',
    policy: 'uniform',
    samplerSeed: seed,
    initialState: initial,
    maxCommands: 8,
    maxPlies: 1000,
  });
  assert.equal(result.error, null);
  assert.deepEqual(result.commands, reference.commands);
  assert.equal(result.metrics.rejected, reference.metrics.rejected);
  assert.equal(nativeHash(result.state), nativeHash(reference.state));
  writeFileSync(
    output,
    JSON.stringify(
      {
        seed,
        rejected: result.metrics.rejected,
        commands: result.commands,
        finalHash: nativeHash(result.state),
      },
      null,
      2,
    ),
    { flag: 'wx' },
  );
  console.log(
    JSON.stringify({ seed, rejected: result.metrics.rejected, commands: result.commands.length }),
  );
} finally {
  client.close();
}
