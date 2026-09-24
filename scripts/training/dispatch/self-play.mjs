import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { closeSync, mkdirSync, openSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const root = fileURLToPath(new URL('../../../', import.meta.url));
const { values } = parseArgs({
  options: Object.fromEntries(
    [
      'games',
      'seed',
      'workers',
      'rules',
      'difficulty',
      'nodes',
      'opponent',
      'opponent-nodes',
      'plies',
      'commands',
      'output-dir',
    ].map((name) => [name, { type: 'string' }]),
  ),
});
const integer = (name, fallback) => {
  const number = Number(values[name] ?? fallback);
  assert.ok(Number.isSafeInteger(number) && number > 0, `${name}必须为正整数`);
  return number;
};
const games = integer('games');
const pairs = games / 2;
const seed = integer('seed', 20260922);
const workers = integer('workers', 6);
const plies = integer('plies', 100);
const commands = integer('commands', 3000);
const rules = values.rules ?? 'classic';
const difficulty = values.difficulty ?? 'hard';
const opponent = values.opponent;
const nodes = values.nodes === undefined ? undefined : integer('nodes');
const opponentNodes =
  values['opponent-nodes'] === undefined ? undefined : integer('opponent-nodes');
assert.ok(games % 2 === 0, 'games必须为偶数，每个种子交换席位');
assert.ok(seed + pairs - 1 <= 0xffffffff, '种子范围超过引擎上限');
assert.ok(['classic', 'shrine'].includes(rules), 'rules无效');
assert.ok(['easy', 'medium', 'hard'].includes(difficulty), 'difficulty无效');
assert.ok(['easy', 'medium', 'hard'].includes(opponent), '必须指定有效的opponent');
assert.ok(difficulty !== opponent || nodes !== opponentNodes, '相同教师配置会重复轨迹');
assert.ok(values['output-dir'], '必须指定新的output-dir');

const directory = resolve(values['output-dir']);
mkdirSync(dirname(directory), { recursive: true });
mkdirSync(directory);
const startedAt = new Date().toISOString();
const started = performance.now();
const jobs = Array(pairs);
let next = 0;
let active = 0;
let maxActive = 0;
let stopping = false;
let interrupted = false;
let sourceSha256;
let commit;
process.once('SIGINT', () => {
  stopping = true;
  interrupted = true;
  console.error('停止分发，等待当前种子对结束，以保留完整压缩轨迹。');
});

// 每个子进程只运行一对换边对局；空闲槽立即领取下一种子对。
async function runPair(index, worker) {
  const pairSeed = seed + index;
  const prefix = join(directory, `seed-${pairSeed}`);
  const job = (jobs[index] = {
    seed: pairSeed,
    worker,
    startedAt: new Date().toISOString(),
    record: `${prefix}.jsonl.gz`,
    report: `${prefix}.json`,
    log: `${prefix}.log`,
  });
  const log = openSync(job.log, 'wx');
  const args = [
    '--import',
    'tsx',
    'scripts/training/self-play.ts',
    '--games',
    '2',
    '--seed',
    String(pairSeed),
    '--rules',
    rules,
    '--difficulty',
    difficulty,
    '--opponent',
    opponent,
    '--plies',
    String(plies),
    '--commands',
    String(commands),
    '--output',
    job.record,
    '--report',
    job.report,
  ];
  if (nodes !== undefined) args.push('--nodes', String(nodes));
  if (opponentNodes !== undefined) args.push('--opponent-nodes', String(opponentNodes));
  active++;
  maxActive = Math.max(maxActive, active);
  try {
    const child = spawn(process.execPath, args, {
      cwd: root,
      stdio: ['ignore', log, log],
      windowsHide: true,
    });
    const [code] = await once(child, 'close');
    assert.equal(code, 0, `种子${pairSeed}采样失败，见${job.log}`);
    const report = JSON.parse(readFileSync(job.report, 'utf8'));
    assert.equal(report.format, 'haojie-teacher-run-v1');
    assert.equal(report.options.seed, pairSeed);
    assert.equal(report.results.length, 2, `种子${pairSeed}缺少换边对局`);
    assert.deepEqual(
      report.results.map((result) => [result.seed, result.primaryPlayer]),
      [
        [pairSeed, 1],
        [pairSeed, 2],
      ],
    );
    sourceSha256 ??= report.sourceSha256;
    commit ??= report.commit;
    assert.equal(report.sourceSha256, sourceSha256, '采样期间源码发生变化');
    assert.equal(report.commit, commit, '采样期间提交发生变化');
    job.commands = report.results.reduce((sum, result) => sum + result.commands, 0);
    job.completedAt = new Date().toISOString();
    console.log(
      JSON.stringify({ type: 'pair-complete', seed: pairSeed, worker, commands: job.commands }),
    );
  } finally {
    active--;
    closeSync(log);
  }
}

async function dispatch(worker) {
  while (!stopping && next < pairs) {
    const index = next++;
    try {
      await runPair(index, worker);
    } catch (error) {
      jobs[index].error = error instanceof Error ? error.message : String(error);
      stopping = true;
      console.error(jobs[index].error);
    }
  }
}

await Promise.all(Array.from({ length: Math.min(workers, pairs) }, (_, i) => dispatch(i + 1)));
const summary = {
  format: 'haojie-teacher-dispatch-v1',
  startedAt,
  completedAt: new Date().toISOString(),
  elapsedMs: performance.now() - started,
  options: {
    games,
    seed,
    workers,
    rules,
    difficulty,
    nodes,
    opponent,
    opponentNodes,
    plies,
    commands,
  },
  maxActive,
  sourceSha256,
  commit,
  interrupted,
  jobs: jobs.filter(Boolean),
};
writeFileSync(join(directory, 'summary.json'), JSON.stringify(summary, null, 2) + '\n', {
  flag: 'wx',
});
console.log(
  JSON.stringify({
    completePairs: summary.jobs.filter((job) => job.completedAt).length,
    pairs,
    maxActive,
  }),
);
if (summary.jobs.some((job) => job.error)) process.exitCode = 1;
else if (interrupted) process.exitCode = 130;
