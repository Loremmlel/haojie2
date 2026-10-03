// 类型化规范流与 TS 独立规范实现比较；遍历规则夹具和冻结开中晚盘。
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { nativeClient } from '../../native/client';
import { nativeHash } from '../../native/pipeline/hash';
import { fixtures } from '../../native/fixtures';
import { combatFixtures } from '../../native/combat-fixtures';
import { preparationFixtures } from '../../native/preparation-fixtures';
import { completeFixtures } from '../../native/complete-fixtures';

const { values } = parseArgs({
  options: {
    executable: { type: 'string' },
    workset: { type: 'string' },
    output: { type: 'string' },
  },
});
assert.ok(values.executable && values.output);
const cases = [
  ...fixtures(),
  ...combatFixtures(),
  ...preparationFixtures(),
  ...completeFixtures(),
].map(({ name, job }) => ({ name, state: job.state }));
if (values.workset)
  cases.push(
    ...JSON.parse(readFileSync(values.workset, 'utf8')).map((r: any) => ({
      name: `${r.path}:${r.index}`,
      state: r.state,
    })),
  );
const client = await nativeClient(values.executable);
try {
  for (const row of cases) {
    const [hash] = await client.request({ op: 'hash-states', states: [row.state] });
    assert.equal(hash.typed, hash.value, row.name);
    assert.equal(hash.typed, nativeHash(row.state), row.name);
  }
  const report = { cases: cases.length, exactCanonicalHash: true };
  writeFileSync(values.output, JSON.stringify(report), { flag: 'wx' });
  console.log(JSON.stringify(report));
} finally {
  client.close();
}
