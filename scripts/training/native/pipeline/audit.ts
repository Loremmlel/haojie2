import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createGame, applyRuntimeCommand } from '../../../../src/engine/commands/game';
import { parseCommand, actorCommandError } from '../../../../src/engine/online/authority';
import { observe } from '../../../../src/ai/observation';
import { TrainingActionTree } from '../../../../src/ai/training/action-tree';
import { encodeDecision } from '../../../../src/ai/training/encoding/decision';
import type { GameState, Player } from '../../../../src/engine/types';
import { cells } from '../../../../src/engine/core/geometry';
import { RuleError } from '../../../../src/engine/core/state';
import { nativeHash } from './hash';

const [input, output] = process.argv.slice(2);
const summaryOnly = process.argv.includes('--summary-only');
assert.ok(input && output);
const lines = readFileSync(input, 'utf8')
  .trim()
  .split('\n')
  .map((line) => JSON.parse(line));
let state: GameState | undefined,
  previous = '',
  commands = 0;
const examples: unknown[] = [];
const coverage = {
  reactions: 0,
  maxPending: 0,
  deaths: 0,
  personalClocks: 0,
  multiCell: 0,
  stacked: 0,
  optional: 0,
};
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
    // 只统计正式采样命令前的实局，不把预置前缀或成功回放次数算作采样工作量。
    coverage.reactions += Number(state.pending.length > 0);
    coverage.maxPending = Math.max(coverage.maxPending, state.pending.length);
    coverage.deaths += Number(state.deaths.length > 0);
    coverage.personalClocks += Number(state.units.some((u) => u.offset !== 0));
    coverage.multiCell += Number(state.units.some((u) => u.size > 1));
    coverage.optional += Number(row.optional);
    const occupied = state.units.flatMap((u) => cells(u).map((p) => `${p.x},${p.y}`));
    coverage.stacked += Number(new Set(occupied).size < occupied.length);
    const observation = observe(state, row.actor),
      tree = new TrainingActionTree(observation, row.actor);
    for (let depth = 0; depth < row.path.length; depth++) {
      const node = tree.node(row.path.slice(0, depth));
      assert.ok(node.choices[row.path[depth]]);
      if (depth === row.path.length - 1)
        assert.deepEqual(node.choices[row.path[depth]].command, row.command);
      if (summaryOnly) continue;
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
  } else if (row.type === 'rejected') {
    assert.ok(state);
    assert.equal(row.index, commands);
    assert.equal(nativeHash(state), row.before);
    assert.equal(row.after, row.before);
    const tree = new TrainingActionTree(observe(state, row.actor), row.actor);
    const choice = tree.node(row.path.slice(0, -1)).choices[row.path.at(-1)];
    assert.equal(choice.status, 'uncertain');
    assert.equal(choice.next, undefined);
    assert.deepEqual(choice.command, row.command);
    assert.equal(actorCommandError(state, row.actor, choice.command), null);
    assert.throws(() => applyRuntimeCommand(state!, choice.command), RuleError);
    assert.equal(nativeHash(state), row.before);
  } else {
    assert.ok(state);
    assert.equal(row.commands, commands);
    assert.equal(row.terminated, state.winner !== undefined);
    assert.equal(row.winner, state.winner ?? null);
  }
}
writeFileSync(output, JSON.stringify({ commands, examples, coverage }), { flag: 'wx' });
console.log(
  JSON.stringify({ commands, examples: examples.length, coverage, finalHash: nativeHash(state) }),
);
