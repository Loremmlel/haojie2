import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createWriteStream, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { once } from 'node:events';
import { cpus } from 'node:os';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

// 输入是冻结的 self-play ESM 构建；每次独立进程测量真实搜索、序列化和落盘。
// 固定 work 预算，计时不进入决策；轨迹摘要包含观察、命令、搜索统计和结果，排除耗时。
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
const stream = createWriteStream(`${values.output}.jsonl`, { flags: 'wx' });
await once(stream, 'open');
const digest = createHash('sha256');
let commands = 0;
const started = new Date().toISOString();
const start = performance.now();
const result = await runSelfPlay(options, async (row) => {
  const { timing, elapsedMs, ...stable } = row;
  digest.update(JSON.stringify(stable) + '\n');
  if (!stream.write(JSON.stringify(row) + '\n')) await once(stream, 'drain');
  if (row.type === 'sample' && ++commands % 100 === 0)
    console.error(`${commands}条命令，${((performance.now() - start) / 1000).toFixed(1)}秒`);
});
stream.end();
await once(stream, 'finish');
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
  trajectorySha256: digest.digest('hex'),
  ...result,
};
writeFileSync(values.output, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify(report, null, 2));
