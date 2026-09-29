import assert from 'node:assert/strict';
import { parseArgs } from 'node:util';
import { resolve } from 'node:path';
import { TrainingActionTree, type ActionNode } from '../../../../src/ai/training/action-tree';
import { createDecisionEncoder } from '../../../../src/ai/training/encoding/decision';
import { observe } from '../../../../src/ai/observation';
import { createTrainingInspector } from '../../../../src/ai/training/queries';
import { createGame } from '../../../../src/engine/commands/game';
import { TinyPolicy } from '../../economics/policy';
import { nativeClient } from '../client';
import { fixtures } from '../fixtures';
import { combatFixtures } from '../combat-fixtures';
import { preparationFixtures } from '../preparation-fixtures';
import { completeFixtures } from '../complete-fixtures';
import { freeze } from './artifacts';

const { values } = parseArgs({
  options: {
    output: { type: 'string' },
    executable: {
      type: 'string',
      default: 'artifacts/native-target/release/haojie-engine-prototype.exe',
    },
    match: { type: 'string' },
  },
});
assert.ok(values.output);
const { executable, save } = await freeze(
  'scripts/training/native/sampling/validate.ts',
  resolve(values.output),
  values.executable,
);
const client = await nativeClient(executable);
const canonical = <T>(v: T): T => JSON.parse(JSON.stringify(v));
const wire = (n: ActionNode) =>
  canonical({
    ...n,
    choices: n.choices.map((c) => ({
      key: c.key,
      command: c.command,
      status: c.status,
      subject: c.subject,
      next: !!c.next,
    })),
  });
let maxError = 0,
  maxLogitError = 0,
  nodes = 0,
  roots = 0;
function near(a: any, b: any, path: string, tolerance: number) {
  if (typeof a === 'number' && typeof b === 'number') {
    const error = Math.abs(a - b);
    if (path.startsWith('logits')) maxLogitError = Math.max(maxLogitError, error);
    else maxError = Math.max(maxError, error);
    assert.ok(error <= tolerance, `${path}: ${a} != ${b}`);
  } else if (Array.isArray(a)) {
    assert.equal(a.length, b.length, path);
    a.forEach((v, i) => near(v, b[i], `${path}[${i}]`, tolerance));
  } else if (a && typeof a === 'object') {
    assert.deepEqual(Object.keys(a).sort(), Object.keys(b).sort(), path);
    for (const k of Object.keys(a)) near(a[k], b[k], `${path}.${k}`, tolerance);
  } else assert.equal(a, b, path);
}
try {
  const policy = new TinyPolicy(73129);
  const cases = [
    ...fixtures(),
    ...combatFixtures(),
    ...preparationFixtures(),
    ...completeFixtures(),
  ].filter((c) => !values.match || c.name.includes(values.match));
  for (const { name, job } of cases)
    for (const actor of [1, 2] as const) {
      const observation = observe(job.state, actor);
      const tree = new TrainingActionTree(observation, actor);
      const encode = createDecisionEncoder(observation, actor);
      const cursors: number[][] = [[]];
      tree.node().choices.forEach((c, i) => {
        if (c.next) cursors.push([i]);
      });
      // 专项中的成功命令补齐深层路径、合成材料、反应与继承能力参数。
      if (
        job.command &&
        createTrainingInspector(observation, actor)(job.command).status !== 'invalid'
      ) {
        for (const { node } of tree.trace(job.command))
          if (!cursors.some((c) => JSON.stringify(c) === JSON.stringify(node.cursor)))
            cursors.push(node.cursor);
      }
      const expected = cursors.map((c) => tree.node(c));
      const got = await client.request({
        op: 'training-nodes',
        observation,
        actor,
        cursors,
        encode: true,
      });
      try {
        assert.deepEqual(got.nodes, expected.map(wire), `${name}: nodes`);
        expected.forEach((n, i) => {
          if (!n.choices.length) {
            assert.equal(got.encoded[i], null);
            assert.equal(got.logits[i], null);
            return;
          }
          const encoded = encode(n);
          near(got.encoded[i], encoded, `${name}:encoded-${i}`, 2e-15);
          near(got.logits[i], policy.logits(encoded), `logits:${name}:${i}`, 2e-8);
        });
      } catch (error) {
        save('failure.json', { name, observation, cursors, got, expected: expected.map(wire) });
        throw error;
      }
      roots++;
      nodes += cursors.length;
      if (roots % 100 === 0) console.log(JSON.stringify({ roots, nodes, maxError, maxLogitError }));
    }
  const base = canonical(observe(createGame(731270001)));
  const invalid = async (observation: unknown, pattern: RegExp) =>
    assert.rejects(
      client.request({ op: 'training-nodes', observation, actor: 1, cursors: [[]], encode: true }),
      pattern,
    );
  await invalid({ ...base, seed: 1 }, /contains seed/);
  await invalid({ ...base, rng: 1 }, /contains rng/);
  await invalid({ ...base, unknown: 1 }, /unencoded field/);
  const draft = canonical(observe(createGame(731270001, 'shrine'), 1));
  draft.shrineDraft!.choices[2] = { kind: 's9', parity: 'odd' };
  await invalid(draft, /choice leaked/);
  save('validation.json', {
    roots,
    nodes,
    maxError,
    maxLogitError,
    rejectedPrivacyAndSchemaCases: 4,
  });
  console.log(JSON.stringify({ roots, nodes, maxError, maxLogitError }));
} finally {
  client.close();
}
