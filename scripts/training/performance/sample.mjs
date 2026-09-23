import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createWriteStream, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { once } from 'node:events';
import { cpus } from 'node:os';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { createGzip } from 'node:zlib';
import { pipeline } from 'node:stream/promises';

// 输入是冻结的 self-play ESM 构建；每次独立进程测量真实搜索、序列化和落盘。
// 固定 work 预算，计时不进入决策；摘要包含增量命令、指纹、统计和结果，排除耗时。
const { values } = parseArgs({
  options: {
    module: { type: 'string' },
    output: { type: 'string' },
    commands: { type: 'string', default: '600' },
    seed: { type: 'string', default: '2026092301' },
  },
});
assert.ok(values.module && values.output, '请提供 --module 和新的 --output 报告路径');
const options = {
  games: 2,
  seed: Number(values.seed),
  rules: 'classic',
  difficulty: 'hard',
  simulations: 800,
  opponent: { difficulty: 'medium', simulations: 320 },
  maxPlies: 40,
  maxCommands: Number(values.commands),
};
assert.ok(Number.isSafeInteger(options.maxCommands) && options.maxCommands > 0);
const { runSelfPlay } = await import(pathToFileURL(resolve(values.module)).href);
mkdirSync(dirname(values.output), { recursive: true });
const output = createWriteStream(`${values.output}.jsonl.gz`, { flags: 'wx' });
await once(output, 'open');
const stream = createGzip();
const written = pipeline(stream, output);
void written.catch(() => {});
const trajectoryFormat = 'haojie-training-record-v1';
const digest = createHash('sha256');
let commands = 0;
const started = new Date().toISOString();
const start = performance.now();
const result = await runSelfPlay(options, async (row) => {
  if (row.type === 'game') assert.equal(row.format, trajectoryFormat, '请重新构建增量教师模块');
  assert.ok(!('observation' in row), '教师模块仍输出旧快照，请重新构建');
  const { timing, elapsedMs, ...stable } = row;
  digest.update(JSON.stringify(stable) + '\n');
  if (!stream.write(JSON.stringify(row) + '\n')) await once(stream, 'drain');
  if (row.type === 'sample' && ++commands % 100 === 0)
    console.error(`${commands}条命令，${((performance.now() - start) / 1000).toFixed(1)}秒`);
});
stream.end();
await written;
const elapsedMs = performance.now() - start;
assert.ok(!result.cancelled && result.results.every((r) => !r.interrupted));
const report = {
  started,
  runtime: { node: process.version, cpu: cpus()[0].model },
  moduleSha256: createHash('sha256').update(readFileSync(values.module)).digest('hex'),
  options,
  elapsedMs,
  commands,
  commandsPerSecond: (commands * 1000) / elapsedMs,
  trajectoryFormat,
  compression: 'gzip',
  trajectorySha256: digest.digest('hex'),
  ...result,
};
writeFileSync(values.output, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify(report, null, 2));
