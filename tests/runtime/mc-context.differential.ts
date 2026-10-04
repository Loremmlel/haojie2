import assert from 'node:assert/strict';
import test from 'node:test';
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import {
  mcSearch,
  sampleMc,
  type McDecision,
  type McNode,
} from '../../src/ai/training/sampling/mc';
import { randomStream } from '../../scripts/training/economics/policy';
import { createGame, applyCommand, type Player, type Command } from '../../src/engine';
import { hasTrait } from '../../src/engine/core/traits';
import { observe, decisionOwner } from '../../src/ai/observation';
import { TrainingActionTree } from '../../src/ai/training/action-tree';
import { nativeHash } from '../../scripts/training/native/pipeline/hash';

function node(path: number[]): McNode {
  const count = !path.length ? 3 : path[0] === 0 ? 0 : 2;
  const globals = Array<number>(32).fill(0);
  globals[1] = 1;
  globals[4] = path[0] ?? -1;
  return {
    stage: path.length ? 'target' : 'action',
    choices: Array.from({ length: count }, (_, i) => ({
      command: { type: 'end', x: i },
      next: !path.length,
      status: 'available',
    })),
    input: {
      globals,
      entities: [Array<number>(64).fill(0)],
      kinds: [1],
      entity_mask: [true],
      candidates: Array.from({ length: count }, (_, i) => {
        const row = Array<number>(64).fill(0);
        row[i] = 1;
        row[48] = (path.length ? 4 : 1) / 16;
        return row;
      }),
      sources: Array<number>(count).fill(-1),
      targets: Array<number>(count).fill(-1),
      candidate_mask: Array<boolean>(count).fill(true),
    },
  };
}

test('MC v2 剩余候选、公开拒绝、回溯、强制选择与Pass输入逐张量TS/Rust一致', () => {
  const engine = process.env.HAOJIE_NATIVE || 'artifacts/native-target/release/haojie-engine.exe';
  if (!existsSync(engine)) throw Error('设置HAOJIE_NATIVE为正式构建；不静默跳过差分');
  const python =
    process.env.HAOJIE_PYTHON ||
    (process.platform === 'win32' ? 'training/.venv/Scripts/python.exe' : 'python');
  const result = spawnSync(python, ['training/tests/console/mc_trace.py', engine], {
    encoding: 'utf8',
    windowsHide: true,
  });
  assert.equal(result.status, 0, result.stderr);
  const traces = JSON.parse(result.stdout);
  for (const trace of traces) {
    const rows: McDecision[] = [];
    const selected = mcSearch(
      node,
      (input) => (input.candidates.length === 2 ? [0, 100] : trace.scores),
      randomStream(trace.seed),
      true,
      (_, path) => JSON.stringify(path) !== '[2,1]',
      (row) => rows.push(row),
    );
    assert.deepEqual(rows, trace.rows);
    assert.deepEqual(selected?.path, trace.path);
    assert.equal(!!selected?.pass, trace.pass);
  }
  assert.ok(
    traces.some((t: { rows: McDecision[] }) => t.rows.some((r) => r.input.kinds.includes(250))),
  );
  assert.ok(traces.some((t: { pass: boolean }) => t.pass));
});

test('自然历史前缀的回合外Pass：实际观察及完整输入相同，权威命令数和RNG哈希不增加', () => {
  const start = JSON.parse(
    gunzipSync(readFileSync('tests/fixtures/native/late-starts.json.gz')).toString(),
  )[0];
  let state = createGame(start.seed, start.rules);
  let index = 0;
  for (; index < start.prelude.length; index++) {
    const other = (3 - decisionOwner(state)) as Player;
    if (
      !state.pending.length &&
      state.units.some((u) => u.owner === other && hasTrait(u, 'u7')) &&
      new TrainingActionTree(observe(state, other), other).node([]).choices.length
    )
      break;
    state = applyCommand(state, start.prelude[index].command);
    state.log = [];
    state.events = [];
  }
  assert.ok(index < start.prelude.length, '登记前缀中必须找到真实可选窗口');
  const rows: McDecision[] = [],
    random = randomStream(871);
  const actor = decisionOwner(state),
    other = (3 - actor) as Player;
  const before = nativeHash(state);
  const logits = (input: McDecision['input']) =>
    input.candidates.map((row) => (row[63] === 1 ? 100 : 0));
  const record = (row: McDecision) => rows.push(structuredClone(row));
  const pass = sampleMc(
    observe(state, other),
    other,
    logits,
    random,
    true,
    () => {
      throw Error('Pass不得提交命令');
    },
    record,
  );
  assert.equal(pass?.pass, true);
  assert.equal(nativeHash(state), before);
  sampleMc(
    observe(state, actor),
    actor,
    logits,
    random,
    false,
    (command: Command, _, status) => {
      try {
        state = applyCommand(state, command);
        state.log = [];
        state.events = [];
        return true;
      } catch (error) {
        if (status !== 'uncertain') throw error;
        return false;
      }
    },
    record,
  );
  const engine = process.env.HAOJIE_NATIVE || 'artifacts/native-target/release/haojie-engine.exe';
  const python =
    process.env.HAOJIE_PYTHON ||
    (process.platform === 'win32' ? 'training/.venv/Scripts/python.exe' : 'python');
  const result = spawnSync(python, ['training/tests/console/mc_trace.py', engine, '--real'], {
    encoding: 'utf8',
    windowsHide: true,
    maxBuffer: 32 * 1024 * 1024,
    input: JSON.stringify({ ...start, prelude: start.prelude.slice(0, index) }),
  });
  assert.equal(result.status, 0, result.stderr);
  const actual = JSON.parse(result.stdout);
  const encoded = rows.map((row) => ({
    ...row,
    input: {
      ...row.input,
      entities: row.input.entities.map((r) => r.map(Math.fround)),
      globals: row.input.globals.map(Math.fround),
      candidates: row.input.candidates.map((r) => r.map(Math.fround)),
    },
  }));
  assert.deepEqual(actual.rows, encoded);
  assert.equal(actual.done.outcome.commands, 1);
  assert.equal(actual.done.finalHash, nativeHash(state));
  assert.equal(actual.rows[0].pass, true);
});
