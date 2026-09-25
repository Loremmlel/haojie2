import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { encodingSourceHash } from '../../encode';
import { readTrainingRecords } from '../../records/replay';
import { hashRecordFile } from '../../records/io';

// 复用已验收的完整对局worker；仅分发独立新种子，不复制规则或采样策略。
const { values } = parseArgs({
  options: {
    output: { type: 'string' },
    seed: { type: 'string' },
    games: { type: 'string' },
    workers: { type: 'string', default: '4' },
  },
});
assert.ok(values.output && values.seed && values.games);
const output = values.output,
  seed = Number(values.seed),
  count = Number(values.games),
  workers = Number(values.workers);
assert.ok(Number.isSafeInteger(seed) && seed > 0 && seed + count <= 0x100000000);
assert.ok(Number.isSafeInteger(count) && count > 0);
assert.ok(Number.isInteger(workers) && workers > 0 && workers <= 4);
mkdirSync(output, { recursive: true });
const digest = (file: string) => createHash('sha256').update(readFileSync(file)).digest('hex');
const write = (name: string, value: unknown) =>
  writeFileSync(resolve(output, name), JSON.stringify(value, null, 2), { flag: 'wx' });
const jobs = Array.from({ length: count }, (_, game) => ({
  game,
  seed: seed + game,
  primary: 1,
  kind: 'selfplay',
  output,
}));
const scriptHashes = Object.fromEntries(
  [
    'scripts/training/search/puct.ts',
    'scripts/training/search/bootstrap/policy.ts',
    'scripts/training/search/bootstrap/games.ts',
    'scripts/training/search/bootstrap/sample.ts',
  ].map((f) => [f, digest(f)]),
);
const protocol = {
  format: 'haojie-bootstrap-selfplay-batch-v1',
  jobs,
  workers,
  scriptHashes,
  sourceSha256: encodingSourceHash(),
  limits: { maxPlies: 120, maxCommands: 1800 },
  note: 'teacher-assisted candidate search; actual terminal labels only, keep truncations and errors',
};
write('protocol.json', protocol);
const results: any[] = Array(count);
let next = 0;
const outcomes = await Promise.allSettled(
  Array.from({ length: Math.min(workers, count) }, async () => {
    while (next < count) {
      const index = next++;
      const child = spawn(
        process.execPath,
        ['--import', 'tsx', fileURLToPath(new URL('./games.ts', import.meta.url)), '--worker'],
        { windowsHide: true, stdio: ['ignore', 'ignore', 'inherit', 'ipc'] },
      );
      try {
        const result = new Promise<any>((accept, reject) => {
          child.on('error', reject);
          child.on('exit', (code) => reject(new Error(`对弈子进程提前退出：${code}`)));
          child.on('message', (message: any) => {
            if (message.progress) console.log(JSON.stringify(message));
            else if (message.error) reject(new Error(message.error));
            else accept(message.result);
          });
        });
        child.send!(jobs[index]);
        results[index] = await result;
        write(`result-${index}.json`, results[index]);
        console.log(JSON.stringify({ complete: results[index] }));
      } finally {
        child.kill();
      }
    }
  }),
);
const failures = outcomes.filter((r) => r.status === 'rejected');
if (failures.length) {
  write(
    'failure.json',
    failures.map((r) => String(r.reason)),
  );
  throw new Error('批次存在中断，保留全部产物，禁止当作成功采样');
}
for (const [file, sha] of Object.entries(scriptHashes)) assert.equal(digest(file), sha);
assert.equal(encodingSourceHash(), protocol.sourceSha256);
let replayed = 0,
  policyTargets = 0,
  fallbackTargets = 0;
for (const result of results) {
  assert.equal(await hashRecordFile(result.output), result.sha256);
  for await (const row of readTrainingRecords(result.output)) {
    if (row.type === 'outcome') assert.ok(!row.interrupted);
    if (row.type !== 'sample') continue;
    replayed++;
    if (row.policyMode === 'search') policyTargets++;
    else {
      assert.equal(row.searchPolicy, null);
      fallbackTargets++;
    }
  }
}
const summary = {
  complete: true,
  protocolSha256: digest(resolve(output, 'protocol.json')),
  results,
  replayed,
  policyTargets,
  fallbackTargets,
  selfplayTerminal: results.filter((r) => r.terminated).length,
  evaluationTerminal: 0,
  truncated: results.filter((r) => r.truncated).length,
  interrupted: 0,
};
write('summary.json', summary);
console.log(JSON.stringify(summary));
