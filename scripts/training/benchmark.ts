import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync, readdirSync } from 'node:fs';
import { cpus, totalmem, availableParallelism } from 'node:os';
import { createHash } from 'node:crypto';
import { execFileSync, fork } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  applyCommand,
  createGame,
  createDemoGame,
  createSession,
  dispatch,
  parseSession,
  commandError,
} from '../../src/engine';
import type { Command, GameState, Player } from '../../src/engine';
import { RULESET_ID } from '../../src/engine/catalog';
import { observe, fingerprint, decisionOwner, imagined } from '../../src/ai/observation';
import { candidateGroups } from '../../src/ai/planning/candidates';
import { TrainingEnvironment } from '../../src/match/training';
import { trainingActionSpace, inspectTrainingCommand } from '../../src/ai/training/queries';
import { sampleTrainingTransition } from '../../src/ai/training/simulation';
import { TrainingService } from './protocol';
import { runSelfPlay } from './self-play';

const corpusPath = 'docs/playtests/current/cli-feedback5-20260923.jsonl';
const corpusText = readFileSync(corpusPath, 'utf8');
const [header, ...rows] = corpusText
  .trim()
  .split('\n')
  .map((line) => JSON.parse(line));
assert.equal(header.ruleset, RULESET_ID, '基准回放必须与当前规则一致。');
const initial = parseSession(JSON.stringify(header.initial)).present;
const positions: { name: string; state: GameState; command: Command; actor: Player }[] = [];
let final = initial;
for (const row of rows) {
  assert.equal(fingerprint(final), row.before);
  positions.push({
    name: `record-${positions.length + 1}`,
    state: final,
    command: row.command,
    actor: row.owner,
  });
  final = applyCommand(final, row.command);
  assert.equal(fingerprint(final), row.after);
}

/** 同一实录比较包装开销；不是自由对弈，更不包含搜索或神经网络推理。 */
function replay(mode: 'engine' | 'training' | 'session-json' | 'protocol-json') {
  if (mode === 'training') {
    const env = TrainingEnvironment.fromState(initial);
    for (const row of rows) env.step(row.owner, row.command);
    assert.equal(fingerprint(env.observation()), fingerprint(final));
    return;
  }
  if (mode === 'protocol-json') {
    const service = new TrainingService();
    const reset = { op: 'reset', env: 'bench', options: { seed: initial.seed, rules: 'shrine' } };
    service.handle(JSON.parse(JSON.stringify(reset)));
    for (const row of rows) {
      const request = JSON.stringify({
        op: 'step',
        env: 'bench',
        actor: row.owner,
        command: row.command,
      });
      const reply = service.handle(JSON.parse(request)) as { ok: boolean; error?: string };
      assert.ok(reply.ok, reply.error);
      JSON.stringify(reply);
    }
    return;
  }
  if (mode === 'session-json') {
    let session = createSession(initial);
    for (const row of rows) {
      session = dispatch(session, row.command);
      JSON.stringify(session);
      fingerprint(session.present);
    }
    assert.equal(fingerprint(session.present), fingerprint(final));
    return;
  }
  let state = initial;
  for (const row of rows) state = applyCommand(state, row.command);
  assert.equal(fingerprint(state), fingerprint(final));
}

const worker = process.argv.includes('--worker');
if (worker) {
  for (let n = 0; n < 3; n++) replay('training');
  process.on('message', (count: number) => {
    const start = performance.now();
    for (let n = 0; n < count; n++) replay('training');
    process.send!({ elapsedMs: performance.now() - start, commands: rows.length * count });
    process.disconnect();
  });
  process.send!({ ready: true });
} else {
  const args = process.argv.slice(2);
  const value = (name: string, fallback: string) => {
    const i = args.indexOf(`--${name}`);
    return i < 0 ? fallback : args[i + 1];
  };
  const integer = (name: string, fallback: number, min = 1) => {
    const n = Number(value(name, String(fallback)));
    assert.ok(Number.isSafeInteger(n) && n >= min, `${name} 必须是 >=${min} 的整数`);
    return n;
  };
  const iterations = integer('iterations', 500);
  const repeats = integer('repeats', 30);
  const rounds = integer('rounds', 3);
  const workers = integer('workers', 4);
  const selfPlayOptions = args.includes('--skip-selfplay')
    ? null
    : {
        games: integer('games', 1),
        seed: 20260922,
        difficulty: 'easy' as const,
        simulations: integer('nodes', 40, 40),
        maxCommands: integer('commands', 1200),
        maxPlies: integer('plies', 60),
      };
  const reportPath = value('output', 'artifacts/training-benchmark.json');
  const measurements: {
    name: string;
    operations: number;
    samplesMs: number[];
    medianMs: number;
    operationsPerSecond: number;
  }[] = [];
  const measure = (name: string, run: () => void, count: number, units = 1) => {
    for (let n = 0; n < Math.min(20, count); n++) run();
    const samplesMs = [];
    for (let round = 0; round < rounds; round++) {
      const start = performance.now();
      for (let n = 0; n < count; n++) run();
      samplesMs.push(performance.now() - start);
    }
    const sorted = [...samplesMs].sort((a, b) => a - b);
    const medianMs = sorted[Math.floor(sorted.length / 2)];
    const measurement = {
      name,
      operations: count * units,
      samplesMs,
      medianMs,
      operationsPerSecond: (count * units * 1000) / medianMs,
    };
    measurements.push(measurement);
    console.log(`${name}: ${measurement.operationsPerSecond.toFixed(0)} 次/秒`);
  };

  const demo = createDemoGame();
  const demoCommand = candidateGroups(imagined(observe(demo)), 'easy')
    .flatMap((g) => g.commands)
    .find((c) => commandError(demo, c) === null)!;
  assert.ok(demoCommand);
  const dense = [...positions]
    .filter((p) => p.state.phase === 'play')
    .sort((a, b) => b.state.units.length - a.state.units.length)[0];
  const cases = [
    {
      name: 'classic-opening',
      state: createGame(7),
      command: { type: 'summon' } as Command,
      actor: 1 as Player,
    },
    { ...dense, name: 'recorded-midgame' },
    { name: 'demo-stress', state: demo, command: demoCommand, actor: decisionOwner(demo) },
  ];
  const caseInfo = [];
  for (const { name, state, command, actor } of cases) {
    const observation = observe(state, actor);
    const compact = { ...state, log: [], events: [] };
    caseInfo.push({
      name,
      units: state.units.length,
      landmarks: state.landmarks?.length ?? 0,
      pending: state.pending.length,
      handCards: state.hands[1].length + state.hands[2].length,
      observationBytes: Buffer.byteLength(JSON.stringify(observation)),
      command,
    });
    measure(
      `${name}/clone`,
      () => {
        structuredClone(compact);
      },
      iterations,
    );
    measure(
      `${name}/observe`,
      () => {
        observe(compact, actor);
      },
      iterations,
    );
    measure(
      `${name}/encode-json`,
      () => {
        JSON.stringify(observation);
      },
      iterations,
    );
    measure(
      `${name}/apply`,
      () => {
        applyCommand(compact, command);
      },
      iterations,
    );
    measure(
      `${name}/sample-observation`,
      () => {
        sampleTrainingTransition(observation, actor, command, 42);
      },
      iterations,
    );
    measure(
      `${name}/public-inspect`,
      () => {
        inspectTrainingCommand(observation, actor, command);
      },
      iterations,
    );
    measure(
      `${name}/action-space`,
      () => {
        trainingActionSpace(observation, actor);
      },
      Math.max(10, Math.floor(iterations / 10)),
    );
  }
  for (const mode of ['engine', 'training', 'session-json', 'protocol-json'] as const)
    measure(`replay/${mode}`, () => replay(mode), repeats, rows.length);

  // 子进程先预热再计时；测量隔离环境的扩展性，不把多个线程当成共享同一搜索树。
  const children = Array.from({ length: workers }, () =>
    fork(fileURLToPath(import.meta.url), ['--worker'], {
      execArgv: ['--import', 'tsx'],
      stdio: ['ignore', 'ignore', 'inherit', 'ipc'],
    }),
  );
  let parallel;
  try {
    await Promise.all(
      children.map(
        (child) =>
          new Promise<void>((resolve, reject) => {
            child.once('message', () => resolve());
            child.once('error', reject);
            child.once('exit', (code) => {
              if (code) reject(new Error(`基准子进程退出 ${code}`));
            });
          }),
      ),
    );
    const results = children.map(
      (child) =>
        new Promise<{ elapsedMs: number; commands: number }>((resolve, reject) => {
          child.once('message', resolve);
          child.once('error', reject);
          child.once('exit', (code) => {
            if (code) reject(new Error(`基准子进程退出 ${code}`));
          });
        }),
    );
    const start = performance.now();
    for (const child of children) child.send(repeats);
    const perProcess = await Promise.all(results);
    const elapsedMs = performance.now() - start;
    const commands = perProcess.reduce((sum, r) => sum + r.commands, 0);
    parallel = {
      workers,
      elapsedMs,
      commands,
      commandsPerSecond: (commands * 1000) / elapsedMs,
      perProcess,
    };
    console.log(`replay/${workers}-processes: ${parallel.commandsPerSecond.toFixed(0)} 次/秒`);
  } finally {
    for (const child of children) if (!child.killed) child.kill();
  }

  const selfPlay = [];
  if (selfPlayOptions) {
    for (const rules of ['classic', 'shrine'] as const) {
      console.log(`开始 ${rules} 教师对弈基准（固定工作量，不含网络推理）`);
      const start = performance.now();
      const run = await runSelfPlay({
        ...selfPlayOptions,
        rules,
      });
      const elapsedMs = performance.now() - start;
      const commands = run.results.reduce((n, r) => n + r.commands, 0);
      const simulations = run.results.reduce((n, r) => n + r.simulations, 0);
      selfPlay.push({
        rules,
        ...run,
        elapsedMs,
        commandsPerSecond: (commands * 1000) / elapsedMs,
        simulationsPerSecond: (simulations * 1000) / elapsedMs,
      });
      console.log(
        `${rules}: ${commands} 条命令，${(elapsedMs / 1000).toFixed(1)}秒，完成 ${run.results.filter((r) => r.terminated).length} 局，截断 ${run.results.filter((r) => r.truncated).length} 局`,
      );
    }
  }
  const sourceFiles: string[] = [];
  const visit = (path: string) => {
    for (const entry of readdirSync(path, { withFileTypes: true })) {
      const file = join(path, entry.name);
      if (entry.isDirectory()) visit(file);
      else if (file.endsWith('.ts')) sourceFiles.push(file);
    }
  };
  for (const path of ['src/engine', 'src/ai', 'src/match', 'scripts/training']) visit(path);
  const digest = createHash('sha256');
  for (const file of sourceFiles.sort())
    digest.update(file.replaceAll('\\', '/')).update(readFileSync(file));
  const report = {
    format: 'haojie-training-benchmark-v1',
    timestamp: new Date().toISOString(),
    ruleset: RULESET_ID,
    commit: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
    sourceSha256: digest.digest('hex'),
    node: process.version,
    platform: process.platform,
    cpu: cpus()[0]?.model,
    logicalCpus: cpus().length,
    availableParallelism: availableParallelism(),
    memoryGiB: totalmem() / 1024 ** 3,
    memory: process.memoryUsage(),
    maxRssKiB: process.resourceUsage().maxRSS,
    config: { iterations, repeats, rounds, workers, selfPlay: selfPlayOptions },
    corpus: {
      path: corpusPath,
      commands: rows.length,
      sha256: createHash('sha256').update(corpusText).digest('hex'),
      finalFingerprint: fingerprint(final),
    },
    cases: caseInfo,
    measurements,
    parallel,
    selfPlay,
    limitations: [
      '纯转移与回放不包含动作搜索或神经网络推理；吞吐不是MCTS节点数。',
      'session-json含历史序列化，不含磁盘写入与教师trace；protocol-json含编码但不含OS管道传输。',
      'demo-stress是人工演示局；实录片段不是完整比赛。',
      '教师仅使用现有有界候选；截断不判胜，无胜率或浏览器模型延迟结论。',
      '多进程结果仅测预热后的规则环境，不代表端到端训练可线性加速。',
    ],
  };
  mkdirSync(dirname(reportPath), { recursive: true });
  writeFileSync(reportPath, JSON.stringify(report, null, 2) + '\n');
  console.log(`基准报告：${reportPath}`);
}
