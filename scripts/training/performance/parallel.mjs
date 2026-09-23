import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdirSync, openSync, closeSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

// 四个独立采样进程使用不同种子；与单进程脚本共用预算和落盘路径，不夹带其他基准。
const [module, directory] = process.argv.slice(2);
assert.ok(module && directory, '参数为冻结的教师模块与新报告目录');
mkdirSync(directory, { recursive: true });
const start = performance.now();
const runs = await Promise.all(
  Array.from({ length: 4 }, async (_, i) => {
    const output = join(directory, `${i}.json`);
    const log = openSync(join(directory, `${i}.log`), 'wx');
    const child = spawn(
      process.execPath,
      [
        'scripts/training/performance/sample.mjs',
        '--module',
        module,
        '--output',
        output,
        '--seed',
        String(2026092301 + i),
        '--commands',
        '600',
      ],
      { stdio: ['ignore', log, log], windowsHide: true },
    );
    try {
      const [code] = await once(child, 'exit');
      assert.equal(code, 0, `采样进程${i}失败，见对应日志`);
      console.log(`采样进程${i}完成`);
      return JSON.parse(readFileSync(output, 'utf8'));
    } finally {
      closeSync(log);
    }
  }),
);
const elapsedMs = performance.now() - start;
const commands = runs.reduce((sum, r) => sum + r.commands, 0);
const report = { elapsedMs, commands, commandsPerSecond: (commands * 1000) / elapsedMs, runs };
writeFileSync(join(directory, 'summary.json'), JSON.stringify(report, null, 2) + '\n', {
  flag: 'wx',
});
console.log(JSON.stringify({ elapsedMs, commands, commandsPerSecond: report.commandsPerSecond }));
