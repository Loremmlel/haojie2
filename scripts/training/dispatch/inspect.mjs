import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { closeSync, existsSync, mkdirSync, openSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const root = fileURLToPath(new URL('../../../', import.meta.url));
const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    output: { type: 'string' },
    workers: { type: 'string' },
    encode: { type: 'boolean' },
  },
});
const workers = Number(values.workers ?? 6);
assert.ok(Number.isSafeInteger(workers) && workers > 0, 'workers必须为正整数');
assert.ok(values.output && positionals.length, '请提供教师记录路径和新的--output文件');
const output = resolve(values.output);
const partsDir = `${output}.parts`;
assert.ok(!existsSync(output) && !existsSync(partsDir), '审计输出已存在，请保留旧实验');
mkdirSync(dirname(output), { recursive: true });
mkdirSync(partsDir);

const started = performance.now();
const jobs = Array(positionals.length);
let next = 0;
let active = 0;
let maxActive = 0;
let stopping = false;
let interrupted = false;
process.once('SIGINT', () => {
  stopping = true;
  interrupted = true;
  console.error('停止分发，等待正在重放的文件结束。');
});

// 单文件重放与编码互不依赖；各子进程写独立报告，完成后再检查跨文件约束。
async function runFile(index, worker) {
  const path = resolve(positionals[index]);
  const job = (jobs[index] = {
    path,
    worker,
    startedAt: new Date().toISOString(),
    report: join(partsDir, `${index}.json`),
    log: join(partsDir, `${index}.log`),
  });
  const log = openSync(job.log, 'wx');
  active++;
  maxActive = Math.max(maxActive, active);
  try {
    const child = spawn(
      process.execPath,
      [
        '--import',
        'tsx',
        'scripts/training/inspect-teacher.ts',
        path,
        ...(values.encode ? ['--encode'] : []),
        '--output',
        job.report,
      ],
      { cwd: root, stdio: ['ignore', log, log], windowsHide: true },
    );
    const [code] = await once(child, 'close');
    assert.equal(code, 0, `文件重放失败：${path}，见${job.log}`);
    const report = JSON.parse(readFileSync(job.report, 'utf8'));
    assert.equal(report.format, 'haojie-teacher-inspection-v1');
    assert.equal(report.encodingChecked, !!values.encode);
    assert.equal(report.files.length, 1);
    assert.equal(resolve(report.files[0].path), path);
    job.completedAt = new Date().toISOString();
    job.games = report.counts.games;
    job.commands = report.counts.commands;
    console.log(JSON.stringify({ type: 'file-complete', path, worker, commands: job.commands }));
    return report;
  } finally {
    active--;
    closeSync(log);
  }
}

const reports = Array(positionals.length);
async function dispatch(worker) {
  while (!stopping && next < positionals.length) {
    const index = next++;
    try {
      reports[index] = await runFile(index, worker);
    } catch (error) {
      jobs[index].error = error instanceof Error ? error.message : String(error);
      stopping = true;
      console.error(jobs[index].error);
    }
  }
}

await Promise.all(
  Array.from({ length: Math.min(workers, positionals.length) }, (_, i) => dispatch(i + 1)),
);
const dispatchStatus = { interrupted, maxActive, jobs: jobs.filter(Boolean) };
writeFileSync(join(partsDir, 'dispatch.json'), JSON.stringify(dispatchStatus, null, 2) + '\n', {
  flag: 'wx',
});
if (dispatchStatus.jobs.some((job) => job.error) || interrupted) {
  process.exitCode = interrupted ? 130 : 1;
} else {
  const sourceSha256 = reports[0].sourceSha256;
  const ruleset = reports[0].ruleset;
  const identities = new Set();
  const games = [];
  const files = [];
  const counts = Object.fromEntries(Object.keys(reports[0].counts).map((key) => [key, 0]));
  for (const report of reports) {
    assert.equal(report.sourceSha256, sourceSha256, '各文件审计使用的源码不同');
    assert.equal(report.ruleset, ruleset, '不能混合不同规则版本');
    files.push(...report.files);
    for (const [key, value] of Object.entries(report.counts)) counts[key] += value;
    for (const game of report.games) {
      const identity = game.gameId ?? `${game.ruleset}:${game.rules}:${game.seed}`;
      assert.ok(!identities.has(identity), `跨文件重复对局：${identity}`);
      identities.add(identity);
      games.push(game);
    }
  }
  const pairs = new Map();
  for (const game of games) {
    if (!game.teachers || ![1, 2].includes(game.primaryPlayer)) continue;
    const profile = (p) => `${p.difficulty}:${p.simulations ?? 'production'}`;
    const primary = profile(game.teachers[game.primaryPlayer]);
    const secondary = profile(game.teachers[3 - game.primaryPlayer]);
    const key = `${game.ruleset}:${game.rules}:${game.seed}:${primary}:${secondary}`;
    const pair = pairs.get(key) ?? [];
    pair.push(game);
    pairs.set(key, pair);
  }
  const pairScores = [...pairs.entries()].map(([group, pair]) => {
    const complete =
      pair.length === 2 &&
      new Set(pair.map((game) => game.primaryPlayer)).size === 2 &&
      pair.every((game) => game.terminated);
    return {
      group,
      seed: pair[0].seed,
      complete,
      primaryScore: complete
        ? pair.reduce(
            (sum, game) =>
              sum + (game.winner === 'draw' ? 0.5 : Number(game.winner === game.primaryPlayer)),
            0,
          ) / 2
        : null,
    };
  });
  const summary = {
    format: 'haojie-teacher-parallel-inspection-v1',
    encodingChecked: !!values.encode,
    ruleset,
    sourceSha256,
    files,
    games,
    counts,
    pairScores,
    profilesByFile: reports.map((report) => ({
      path: report.files[0].path,
      profiles: report.profiles,
    })),
    maxActive,
    elapsedMs: performance.now() - started,
    partsDir,
  };
  writeFileSync(output, JSON.stringify(summary, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ counts, maxActive, elapsedMs: summary.elapsedMs }));
}
