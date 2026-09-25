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
    checkpoint: { type: 'string' },
    gate: { type: 'string' },
    'evaluation-seeds': { type: 'string', default: '0' },
  },
});
assert.ok(values.output && values.seed && values.games);
const output = values.output,
  seed = Number(values.seed),
  count = Number(values.games),
  workers = Number(values.workers);
const evaluationSeeds = Number(values['evaluation-seeds']);
assert.ok(Number.isSafeInteger(seed) && seed > 0 && seed + count <= 0x100000000);
assert.ok(Number.isSafeInteger(count) && count > 0);
assert.ok(Number.isInteger(workers) && workers > 0 && workers <= 4);
assert.ok(Number.isInteger(evaluationSeeds) && evaluationSeeds >= 0 && evaluationSeeds * 2 < count);
assert.ok(!evaluationSeeds || values.checkpoint, '换边评测须提供网络检查点');
const digest = (file: string) => createHash('sha256').update(readFileSync(file)).digest('hex');
const write = (name: string, value: unknown) =>
  writeFileSync(resolve(output, name), JSON.stringify(value, null, 2), { flag: 'wx' });
const jobs = Array.from({ length: count }, (_, game) => ({
  game,
  seed: seed + (game < evaluationSeeds * 2 ? Math.floor(game / 2) : game - evaluationSeeds),
  primary: game < evaluationSeeds * 2 ? (game % 2 ? 2 : 1) : 1,
  kind: game < evaluationSeeds * 2 ? 'evaluation' : 'selfplay',
  output,
  ...(values.checkpoint ? { checkpoint: resolve(values.checkpoint) } : {}),
}));
if (values.checkpoint) {
  assert.ok(values.gate, '网络回接须提供已通过的战术门槛目录');
  const gate = JSON.parse(readFileSync(resolve(values.gate, 'summary.json'), 'utf8'));
  const gateProtocol = JSON.parse(readFileSync(resolve(values.gate, 'protocol.json'), 'utf8'));
  assert.equal(
    gate.protocolSha256,
    digest(resolve(values.gate, 'protocol.json')),
    '门槛报告与协议不匹配',
  );
  assert.equal(gateProtocol.sourceSha256, encodingSourceHash(), '门槛规则与编码源码已变化');
  for (const [file, sha] of Object.entries(gateProtocol.scriptHashes))
    assert.equal(digest(file), sha, `门槛搜索源码已变化：${file}`);
  assert.equal(gate.fixtureOptimal, 60);
  assert.equal(gate.supportedNaturalWins, 10);
  assert.ok(gate.networkCalls > 0);
  assert.equal(gateProtocol.checkpoint.checkpoint_sha256, digest(values.checkpoint));
}
const scriptHashes = Object.fromEntries(
  [
    'scripts/training/search/puct.ts',
    'scripts/training/search/bootstrap/policy.ts',
    'scripts/training/search/bootstrap/games.ts',
    'scripts/training/search/bootstrap/sample.ts',
    'scripts/training/search/value-cycle/model.ts',
    'scripts/training/python-policy.ts',
    'training/haojie_training/inference.py',
    'training/haojie_training/model.py',
    'training/haojie_training/runtime.py',
    'training/haojie_training/data.py',
  ].map((f) => [f, digest(f)]),
);
const protocol = {
  format: values.checkpoint
    ? 'haojie-bootstrap-value-cycle-v1'
    : 'haojie-bootstrap-selfplay-batch-v1',
  jobs,
  workers,
  scriptHashes,
  sourceSha256: encodingSourceHash(),
  limits: { maxPlies: 120, maxCommands: 1800 },
  ...(values.checkpoint
    ? {
        checkpoint: {
          path: resolve(values.checkpoint),
          sha256: digest(values.checkpoint),
          scale: 0.25,
        },
        gateSha256: digest(resolve(values.gate!, 'summary.json')),
      }
    : {}),
  note: 'teacher-assisted candidate search; actual terminal labels only, keep truncations and errors',
};
mkdirSync(output, { recursive: true });
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
if (protocol.checkpoint) assert.equal(digest(protocol.checkpoint.path), protocol.checkpoint.sha256);
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
  selfplayTerminal: results.filter((r) => r.kind === 'selfplay' && r.terminated).length,
  evaluationTerminal: results.filter((r) => r.kind === 'evaluation' && r.terminated).length,
  truncated: results.filter((r) => r.truncated).length,
  interrupted: 0,
};
write('summary.json', summary);
console.log(JSON.stringify(summary));
