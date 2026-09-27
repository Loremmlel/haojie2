import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import type * as API from './api';
import type { Command } from '../../../src/engine/types';

const { values } = parseArgs({
  options: {
    baseline: { type: 'string' },
    current: { type: 'string' },
    output: { type: 'string' },
    cases: { type: 'string', default: '32' },
    nodes: { type: 'string', default: '128' },
  },
});
assert.ok(values.baseline && values.current && values.output);
const baseline: typeof API = await import(pathToFileURL(resolve(values.baseline)).href);
const current: typeof API = await import(pathToFileURL(resolve(values.current)).href);
const count = Number(values.cases),
  limit = Number(values.nodes);
assert.ok(Number.isSafeInteger(count) && count >= 4 && count <= 128 && count % 4 === 0);
assert.ok(Number.isSafeInteger(limit) && limit >= 1 && limit <= 4096);
const results = [];
for (const name of ['classic-tiny', 'shrine-tiny', 'classic-uniform', 'shrine-uniform']) {
  const path = `artifacts/training/economics-20260926/${name}/worker-0.jsonl.gz`;
  const rows: any[] = [];
  for await (const row of baseline.readTrainingRecords(path))
    if (row.type === 'decision') rows.push(row);
  for (let i = 0; i < count / 4; i++) {
    const row = rows[Math.floor(((i + 0.5) * rows.length) / (count / 4))];
    const observation = row.observation,
      before = structuredClone(observation);
    const a = new baseline.TrainingActionTree(observation, row.actor);
    const b = new current.TrainingActionTree(observation, row.actor);
    assert.deepEqual(b.actions, a.actions, `${name}:${row.index} 动作规格`);
    const queue: number[][] = [[]],
      visited = new Set<string>();
    let encoded = 0,
      choices = 0;
    const check = (cursor: number[]) => {
      if (visited.has(cursor.join(','))) return;
      visited.add(cursor.join(','));
      const oldNode = a.node(cursor),
        newNode = b.node(cursor);
      assert.deepEqual(newNode, oldNode, `${name}:${row.index}:${cursor} 候选/顺序/状态`);
      choices += oldNode.choices.length;
      if (oldNode.choices.length) {
        assert.deepEqual(
          current.encodeDecision(observation, row.actor, newNode),
          baseline.encodeDecision(observation, row.actor, oldNode),
          `${name}:${row.index}:${cursor} 完整网络输入`,
        );
        encoded++;
      }
      oldNode.choices.forEach((choice, j) => {
        if (choice.next) queue.push([...cursor, j]);
      });
    };
    // 广度遍历有明确上限；另强制走真实记录路径，不能将未展开的路径树称为全覆盖。
    const trace = a.trace(row.command);
    assert.deepEqual(b.trace(row.command), trace);
    for (const step of trace) check(step.node.cursor);
    for (let j = 0; j < queue.length && visited.size < limit; j++) check(queue[j]);
    const probes: Command[] = [{ type: 'begin' }, { type: 'end' }];
    const pieces = [...observation.units, ...(observation.landmarks ?? [])];
    const targets = [...pieces.map((u: any) => u.id), 'base-1', 'base-2'];
    for (const u of pieces) {
      for (const targetId of targets) probes.push({ type: 'attack', unitId: u.id, targetId });
      for (let n = 0; n < 117; n++)
        probes.push({ type: 'move', unitId: u.id, x: (n % 9) + 1, y: Math.floor(n / 9) + 1 });
      probes.push({ type: 'clock', targetId: u.id });
    }
    for (const card of observation.hands[row.actor])
      for (let n = 0; n < 117; n++)
        probes.push({ type: 'deploy', cardId: card.id, x: (n % 9) + 1, y: Math.floor(n / 9) + 1 });
    for (const c of probes)
      assert.deepEqual(
        current.inspectCommand(b.position, c),
        baseline.inspectCommand(a.position, c),
        `${name}:${row.index}:${JSON.stringify(c)} 预检及拒绝原因`,
      );
    assert.deepEqual(observation, before, '查询修改了公开观察');
    const result = {
      name,
      index: row.index,
      nodes: visited.size,
      encoded,
      choices,
      probes: probes.length,
      frontierRemaining: queue.filter((c) => !visited.has(c.join(','))).length,
    };
    results.push(result);
    console.log(JSON.stringify(result));
  }
}
const report = {
  baseline: values.baseline,
  current: values.current,
  equal: true,
  cases: results.length,
  results,
};
writeFileSync(values.output, JSON.stringify(report, null, 2), { flag: 'wx' });
