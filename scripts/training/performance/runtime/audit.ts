import assert from 'node:assert/strict';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { fixtures } from '../../native/fixtures';
import { combatFixtures } from '../../native/combat-fixtures';
import { completeFixtures } from '../../native/complete-fixtures';
import { preparationFixtures } from '../../native/preparation-fixtures';
import type * as API from '../api';
import type { Command, GameState } from '../../../../src/engine/types';

const { values } = parseArgs({
  options: {
    baseline: { type: 'string' },
    candidate: { type: 'string' },
    output: { type: 'string' },
    match: { type: 'string' },
    workset: { type: 'string' },
  },
});
assert.ok(values.baseline && values.candidate && values.output);
const output = resolve(values.output);
mkdirSync(output);
const a: typeof API = await import(pathToFileURL(resolve(values.baseline)).href);
const b: typeof API = await import(pathToFileURL(resolve(values.candidate)).href);
const cases = [
  ...fixtures(),
  ...combatFixtures(),
  ...preparationFixtures(),
  ...completeFixtures(),
].filter((c) => !values.match || c.name.includes(values.match));
const counts = {
  fixtures: 0,
  probes: 0,
  invalid: 0,
  uncertain: 0,
  settled: 0,
  roots: 0,
  nodes: 0,
  encodings: 0,
  leaves: 0,
  sampled: 0,
};
const policies = [new a.TinyPolicy(73129), new b.TinyPolicy(73129)];
const result = (api: typeof API, s: GameState, c: Command) => {
  try {
    return { state: api.applyCommand(s, c) };
  } catch (e) {
    return { error: e instanceof Error ? e.message : String(e) };
  }
};
let current: unknown;
try {
  for (const { name, job } of cases) {
    const original = structuredClone(job.state);
    for (const c of job.probes) {
      current = { name, command: c, state: job.state };
      const x = a.inspectCommand(job.state, c),
        y = b.inspectCommand(job.state, c);
      assert.deepEqual(y, x, name);
      counts.probes++;
      if (x.status === 'invalid') counts.invalid++;
      if (x.status === 'uncertain') counts.uncertain++;
      const old = result(a, job.state, c),
        next = result(b, job.state, c);
      assert.deepEqual(next, old, name);
      assert.deepEqual(job.state, original, `${name}: 输入/失败原子性`);
      if ('state' in next && next.state) {
        structuredClone(next.state);
        counts.settled++;
      }
    }
    for (const actor of [1, 2] as const) {
      const observation = a.observe(job.state, actor),
        before = structuredClone(observation);
      const trees = [
        new a.TrainingActionTree(observation, actor),
        new b.TrainingActionTree(observation, actor),
      ];
      const encoders = [
        a.createDecisionEncoder(observation, actor),
        b.createDecisionEncoder(observation, actor),
      ];
      assert.deepEqual(trees[1].actions, trees[0].actions, name);
      const cursors: number[][] = [[]];
      trees[0].node().choices.forEach((c, i) => {
        if (c.next) cursors.push([i]);
      });
      if (job.command) {
        try {
          for (const step of trees[0].trace(job.command))
            if (!cursors.some((c) => c.join() === step.node.cursor.join()))
              cursors.push(step.node.cursor);
        } catch {
          /* 非操作者或非法夹具无成功路径。 */
        }
      }
      for (const cursor of cursors) {
        current = { name, actor, cursor, observation };
        const nodes = trees.map((t) => t.node(cursor));
        assert.deepEqual(nodes[1], nodes[0], `${name}:${actor}:${cursor}`);
        counts.nodes++;
        counts.leaves += nodes[0].choices.filter((c) => !c.next).length;
        if (nodes[0].choices.length) {
          const inputs = nodes.map((n, i) => encoders[i](n));
          assert.deepEqual(inputs[1], inputs[0]);
          assert.deepEqual(policies[1].logits(inputs[1]), policies[0].logits(inputs[0]));
          counts.encodings++;
        }
      }
      assert.deepEqual(observation, before, `${name}: 观察隔离`);
      counts.roots++;
    }
    counts.fixtures++;
    if (counts.fixtures % 200 === 0) console.log(JSON.stringify(counts));
  }
  if (values.workset)
    for (const [i, row] of JSON.parse(readFileSync(values.workset, 'utf8')).entries()) {
      current = { path: row.path, index: row.index };
      const randoms = [a.randomStream(982451653 + i), b.randomStream(982451653 + i)],
        calls = [0, 0];
      const metrics = [a.emptyMetrics(), b.emptyMetrics()];
      const old = a.sampleCommand(
        row.observation,
        row.actor,
        policies[0],
        () => {
          calls[0]++;
          return randoms[0]();
        },
        metrics[0],
      );
      const next = b.sampleCommand(
        row.observation,
        row.actor,
        policies[1],
        () => {
          calls[1]++;
          return randoms[1]();
        },
        metrics[1],
      );
      assert.deepEqual(next, old);
      assert.equal(calls[0], calls[1]);
      assert.equal(randoms[0](), randoms[1]());
      for (const key of [
        'nodes',
        'evaluations',
        'forced',
        'backtracks',
        'maxEntities',
        'maxCandidates',
        'offTurnCommands',
        'offTurnPasses',
      ] as const)
        assert.equal(metrics[1][key], metrics[0][key]);
      counts.sampled++;
    }
  writeFileSync(
    join(output, 'audit.json'),
    JSON.stringify({ equal: true, counts, resource: process.resourceUsage() }, null, 2),
    { flag: 'wx' },
  );
  console.log(JSON.stringify(counts));
} catch (e) {
  writeFileSync(
    join(output, 'failure.json'),
    JSON.stringify({ current, counts, error: String(e) }, null, 2),
    { flag: 'wx' },
  );
  throw e;
}
