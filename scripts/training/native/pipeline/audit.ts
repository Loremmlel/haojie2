import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createGame, applyRuntimeCommand } from '../../../../src/engine/commands/game';
import { parseCommand, actorCommandError } from '../../../../src/engine/online/authority';
import { observe } from '../../../../src/ai/observation';
import { TrainingActionTree } from '../../../../src/ai/training/action-tree';
import { encodeDecision } from '../../../../src/ai/training/encoding/decision';
import type { GameState, Player } from '../../../../src/engine/types';
import { nativeHash } from './hash';

const [input, output] = process.argv.slice(2);
assert.ok(input && output);
const lines = readFileSync(input, 'utf8')
  .trim()
  .split('\n')
  .map((line) => JSON.parse(line));
let state: GameState | undefined,
  previous = '',
  commands = 0;
const examples: unknown[] = [];
const clean = (s: GameState) => {
  s.log = [];
  s.events = [];
  return s;
};
function step(actor: Player, command: unknown) {
  assert.ok(state);
  const c = parseCommand(command);
  assert.equal(actorCommandError(state, actor, c), null);
  state = clean(applyRuntimeCommand(state, c));
}
for (const envelope of lines) {
  const { sha256, ...content } = envelope;
  assert.equal(content.previous, previous);
  assert.equal(nativeHash(content), sha256);
  previous = sha256;
  const row = content.body;
  if (row.type === 'game') {
    state = clean(createGame(row.start.seed, row.start.rules));
    for (const p of row.start.prelude) step(p.actor, p.command);
    assert.equal(nativeHash(state), row.initialHash, 'initial state');
  } else if (row.type === 'sample') {
    assert.ok(state);
    assert.equal(row.index, commands);
    assert.equal(nativeHash(state), row.before, 'before');
    const observation = observe(state, row.actor),
      tree = new TrainingActionTree(observation, row.actor);
    for (let depth = 0; depth < row.path.length; depth++) {
      const node = tree.node(row.path.slice(0, depth));
      assert.ok(node.choices[row.path[depth]]);
      if (depth === row.path.length - 1)
        assert.deepEqual(node.choices[row.path[depth]].command, row.command);
      const encoded = encodeDecision(observation, row.actor, node);
      if (row.optional && depth === 0) {
        const pass = Array(64).fill(0);
        pass[63] = 1;
        encoded.candidates.push(pass);
        encoded.candidate_mask.push(true);
        encoded.sources.push(-1);
        encoded.targets.push(-1);
      }
      examples.push({ index: commands, step: depth, input: encoded });
    }
    step(row.actor, row.command);
    assert.equal(nativeHash(state), row.after, 'after');
    commands++;
  } else {
    assert.ok(state);
    assert.equal(row.commands, commands);
    assert.equal(row.terminated, state.winner !== undefined);
    assert.equal(row.winner, state.winner ?? null);
  }
}
writeFileSync(output, JSON.stringify({ commands, examples }), { flag: 'wx' });
console.log(JSON.stringify({ commands, examples: examples.length, finalHash: nativeHash(state) }));
