import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { readTrainingRecords } from '../../records/replay';
import { bootstrapValueDecision } from '../bootstrap/policy';
import { openValueModel } from '../value-cycle/model';

/** 重放审计发现的四个实际决策；初始权威状态仅用于回放，网络仍只接收公开张量。 */
const output = resolve(process.argv[2]);
const network = await openValueModel('artifacts/training/value-cycle-20260925/model.pt');
const rows = [];
try {
  for (const [part, indices] of [
    ['selfplay-00', [67, 93]],
    ['selfplay-01', [3, 23]],
  ] as const) {
    const path = `artifacts/training/continuous-night-20260925/round-000/${part}/attempt-00/game-0.jsonl.gz`;
    for await (const row of readTrainingRecords(path)) {
      if (row.type !== 'sample') continue;
      if (row.index > Math.max(...indices)) break;
      if (!(indices as readonly number[]).includes(row.index)) continue;
      const result = await bootstrapValueDecision(row.observation, row.searchSeed, network.value);
      const teacher = row.searchPolicy[0].command;
      assert.deepEqual(result.command, teacher);
      for (const constant of [-0.25, 0, 0.25]) {
        const control = await bootstrapValueDecision(
          row.observation,
          row.searchSeed,
          async () => constant,
        );
        assert.deepEqual(control.command, teacher);
      }
      rows.push({ path, index: row.index, previous: row.command, teacher, result });
    }
  }
} finally {
  network.model.close();
}
assert.equal(rows.length, 4);
mkdirSync(dirname(output), { recursive: true });
writeFileSync(output, JSON.stringify({ passed: true, rows }, null, 2), { flag: 'wx' });
console.log(JSON.stringify({ passed: true, cases: rows.length, output }));
