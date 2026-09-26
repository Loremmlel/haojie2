import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import type * as API from './api';

const { values } = parseArgs({
  options: {
    baseline: { type: 'string' },
    current: { type: 'string' },
    output: { type: 'string' },
  },
});
assert.ok(values.baseline && values.current && values.output);
const baseline: typeof API = await import(pathToFileURL(resolve(values.baseline)).href);
const current: typeof API = await import(pathToFileURL(resolve(values.current)).href);
const records = ['classic-tiny', 'shrine-tiny', 'classic-uniform', 'shrine-uniform'].flatMap(
  (name) => {
    const folder = `artifacts/training/economics-20260926/${name}`;
    return readdirSync(folder)
      .filter((p) => p.endsWith('.jsonl.gz'))
      .map((p) => join(folder, p));
  },
);
const results = [];
for (const path of records) {
  let oldState: ReturnType<typeof baseline.createGame> | undefined;
  let newState: typeof oldState;
  let commands = 0;
  let games = 0;
  for await (const row of baseline.readRecordLines(path)) {
    if (row.type === 'game') {
      oldState = baseline.createGame(row.seed, row.rules);
      newState = current.createGame(row.seed, row.rules);
      oldState.events = newState.events = [];
      oldState.log = newState.log = [];
      assert.deepEqual(newState, oldState);
      games++;
    } else if (row.type === 'decision') {
      assert.ok(oldState && newState);
      const before = JSON.stringify(newState);
      const a = baseline.applyPlayerCommand(oldState, row.actor, row.command);
      const b = current.applyPlayerCommand(newState, row.actor, row.command);
      assert.equal(JSON.stringify(newState), before, `${path}:${row.index} 修改输入`);
      // 比较完整局面：含正式 RNG、反应队列、时钟、身份序号和本步事件，不只比公开指纹。
      assert.deepEqual(b, a, `${path}:${row.index} 完整结算不同`);
      a.log = [];
      b.log = [];
      a.events = [];
      b.events = [];
      oldState = a;
      newState = b;
      commands++;
    }
  }
  results.push({
    path,
    sha256: createHash('sha256').update(readFileSync(path)).digest('hex'),
    games,
    commands,
  });
  console.log(JSON.stringify({ path, games, commands, equal: true }));
}
const report = {
  baseline: values.baseline,
  current: values.current,
  games: results.reduce((n, r) => n + r.games, 0),
  commands: results.reduce((n, r) => n + r.commands, 0),
  fullStatesEqual: true,
  results,
};
writeFileSync(values.output, JSON.stringify(report, null, 2), { flag: 'wx' });
console.log(JSON.stringify({ games: report.games, commands: report.commands, equal: true }));
