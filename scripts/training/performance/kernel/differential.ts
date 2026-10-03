// 全规则夹具比较冻结旧 API 与新 API；失败原子性、候选顺序和编码属于外部语义。
import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { fixtures } from '../../native/fixtures';
import { combatFixtures } from '../../native/combat-fixtures';
import { completeFixtures } from '../../native/complete-fixtures';
import { preparationFixtures } from '../../native/preparation-fixtures';
import { actorCommandError } from '../../../../src/engine/online/authority';
import type * as API from '../api';

const { values } = parseArgs({
  options: {
    baseline: { type: 'string' },
    candidate: { type: 'string' },
    output: { type: 'string' },
  },
});
assert.ok(values.baseline && values.candidate && values.output);
const old: typeof API = await import(pathToFileURL(resolve(values.baseline)).href);
const next: typeof API = await import(pathToFileURL(resolve(values.candidate)).href);
const output = resolve(values.output);
mkdirSync(output);
const plain = (v: unknown) => JSON.parse(JSON.stringify(v));
const apply = (api: typeof API, s: any, c: any) => {
  try {
    return { state: plain(api.applyCommand(s, c)) };
  } catch (e) {
    return { error: (e as Error).message };
  }
};
const cases = [...fixtures(), ...combatFixtures(), ...preparationFixtures(), ...completeFixtures()];
let inspections = 0,
  nodes = 0,
  commands = 0,
  current = '';
try {
  for (const { name, job } of cases) {
    current = name;
    const before = JSON.stringify(job.state);
    for (const command of job.probes) {
      assert.deepEqual(
        next.inspectCommand(job.state, command),
        old.inspectCommand(job.state, command),
        name,
      );
      inspections++;
      assert.deepEqual(apply(next, job.state, command), apply(old, job.state, command), name);
      commands++;
    }
    if (job.command) {
      assert.deepEqual(
        apply(next, job.state, job.command),
        apply(old, job.state, job.command),
        name,
      );
      commands++;
    }
    for (const actor of [1, 2] as const) {
      const observation = old.observe(job.state, actor);
      assert.deepEqual(next.observe(job.state, actor), observation);
      const a = new old.TrainingActionTree(observation, actor),
        b = new next.TrainingActionTree(observation, actor);
      const cursors: number[][] = [[]];
      a.node().choices.forEach((c, i) => {
        if (c.next) cursors.push([i]);
      });
      if (
        job.command &&
        actorCommandError(job.state, actor, job.command) === null &&
        old.inspectCommand(job.state, job.command).status !== 'invalid'
      ) {
        // 当前操作者的权限与规则资格分开；不把无权限夹具变成可执行轨迹。
        for (const { node } of a.trace(job.command))
          if (!cursors.some((c) => JSON.stringify(c) === JSON.stringify(node.cursor)))
            cursors.push(node.cursor);
      }
      const encodeA = old.createDecisionEncoder(observation, actor),
        encodeB = next.createDecisionEncoder(observation, actor);
      for (const cursor of cursors) {
        const x = a.node(cursor),
          y = b.node(cursor);
        assert.deepEqual(plain(y), plain(x), name);
        if (x.choices.length) assert.deepEqual(encodeB(y), encodeA(x), name);
        nodes++;
      }
    }
    assert.equal(JSON.stringify(job.state), before, name);
  }
  const report = {
    fixtures: cases.length,
    inspections,
    commands,
    nodes,
    exactStates: true,
    exactCandidateOrder: true,
    exactEncoding: true,
  };
  writeFileSync(join(output, 'report.json'), JSON.stringify(report, null, 2), { flag: 'wx' });
  console.log(JSON.stringify(report));
} catch (error) {
  writeFileSync(join(output, 'failure.json'), JSON.stringify({ current, error: String(error) }), {
    flag: 'wx',
  });
  throw error;
}
