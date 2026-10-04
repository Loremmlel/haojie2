/** 从失败前缀重放，独立比较公开树的可尝试动作与权威规则拒绝；不使用模型。 */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import {
  createGame,
  applyRuntimeCommand,
  inspectCommand,
  commandError,
} from '../../../../src/engine/commands/game';
import { observe } from '../../../../src/ai/observation';
import { TrainingActionTree } from '../../../../src/ai/training/action-tree';
import { nativeClient } from '../../native/client';
import { nativeHash } from '../../native/pipeline/hash';
import type { Command, Player } from '../../../../src/engine/types';

const [record, executable, output] = process.argv.slice(2);
const rows = readFileSync(record, 'utf8')
  .trim()
  .split('\n')
  .map((v) => JSON.parse(v).body);
const clean = <T extends { log: unknown[]; events: unknown[] }>(s: T) => {
  s.log = [];
  s.events = [];
  return s;
};
let state = clean(createGame(rows[0].start.seed, rows[0].start.rules));
for (const row of rows.filter((v) => v.type === 'sample')) {
  state = clean(applyRuntimeCommand(state, row.command));
  assert.equal(nativeHash(state), row.after);
}
const actor = (state.pending[0]?.owner ?? state.active) as Player;
const observation = observe(state, actor);
const tree = new TrainingActionTree(observation, actor);
const rejected: { command: Command; status: string; error: string }[] = [];
function visit(cursor: number[]) {
  tree.node(cursor).choices.forEach((choice, index) => {
    if (!cursor.length && choice.command.type !== 'move') return;
    if (choice.next) visit([...cursor, index]);
    else {
      const error = commandError(state, choice.command);
      if (error) rejected.push({ command: choice.command, status: choice.status, error });
    }
  });
}
visit([]);
const client = await nativeClient(executable);
try {
  const native = await client.request({
    op: 'run',
    jobs: rejected.map((v) => ({ state, command: v.command, probes: [v.command] })),
  });
  assert.equal(rejected.length, 1);
  assert.equal(native[0].inspections[0].status, 'uncertain');
  assert.equal(native[0].result.status, 'invalid');
  assert.equal(native[0].result.message, rejected[0].error);
  const reset = await client.request({ op: 'reset', state });
  const result = await client.request({
    op: 'step',
    revision: reset.revision,
    commands: [rejected[0].command],
  });
  assert.equal(result.results[0].status, 'invalid');
  assert.equal(result.revision, reset.revision);
  const after = await client.request({ op: 'export' });
  assert.equal(nativeHash(after.state), nativeHash(state));
  writeFileSync(
    output,
    JSON.stringify(
      {
        state,
        observation,
        rejected,
        private_inspections: rejected.map((v) => inspectCommand(state, v.command)),
        native,
        failure_atomic_hash: nativeHash(after.state),
      },
      null,
      2,
    ),
    { flag: 'wx' },
  );
  console.log(JSON.stringify({ rejected, native }));
} finally {
  client.close();
}
