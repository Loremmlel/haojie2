// 只运行含 kernel-profile 的独立程序；保留累计分配与嵌套独占时间，不作为正式倍率。
import assert from 'node:assert/strict';
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { createHash } from 'node:crypto';
import { nativeClient } from '../../native/client';

const { values } = parseArgs({
  options: {
    executable: { type: 'string' },
    workset: { type: 'string' },
    output: { type: 'string' },
    commands: { type: 'string', default: '2' },
  },
});
assert.ok(values.executable && values.workset && values.output);
const output = resolve(values.output);
mkdirSync(output);
const executable = join(output, 'engine.exe');
copyFileSync(values.executable, executable);
const client = await nativeClient(executable);
const cases = JSON.parse(readFileSync(values.workset, 'utf8')),
  rows = [];
try {
  for (const row of cases) {
    const result = await client.request({
      op: 'sample-game',
      initialState: row.state,
      seed: row.seed,
      rules: row.rules,
      policy: 'tiny',
      maxCommands: Number(values.commands),
      maxPlies: 500,
    });
    assert.equal(result.error, null);
    assert.ok(result.kernelProfile, '需要 kernel-profile 构建');
    rows.push({
      path: row.path,
      index: row.index,
      commands: result.commands.length,
      metrics: result.metrics,
      profile: result.kernelProfile,
    });
  }
  writeFileSync(
    join(output, 'profile.json'),
    JSON.stringify(
      {
        rows,
        worksetSha256: createHash('sha256').update(readFileSync(values.workset)).digest('hex'),
      },
      null,
      2,
    ),
    { flag: 'wx' },
  );
  console.log(
    JSON.stringify({ cases: rows.length, commands: rows.reduce((n, r) => n + r.commands, 0) }),
  );
} finally {
  client.close();
}
