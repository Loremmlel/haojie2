import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { cpus } from 'node:os';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { build } from 'esbuild';
import { applyCommand, createGame, inspectCommand } from '../../../src/engine/commands/game';
import { applyPlayerCommand } from '../../../src/engine/online/authority';
import { RuleError } from '../../../src/engine/core/state';
import { readTrainingRecords } from '../records/replay';
import { nativeClient } from './client';
import { fixtures, moveProbes } from './fixtures';
import { combatFixtures, reactionProbes } from './combat-fixtures';
import { preparationFixtures } from './preparation-fixtures';
import { completeFixtures } from './complete-fixtures';
import {
  benchmarkWindows,
  validateResidentProtocol,
  validateWindow,
  windowIdentity,
} from './resident';
import type { CommandWindow } from './resident';
import type { Job } from './fixtures';
import type { Command, GameState } from '../../../src/engine/types';

const { values } = parseArgs({
  options: {
    output: { type: 'string' },
    executable: {
      type: 'string',
      default: 'artifacts/native-target/release/haojie-engine-prototype.exe',
    },
    'fixtures-only': { type: 'boolean', default: false },
    match: { type: 'string' },
    'all-workers': { type: 'boolean', default: false },
  },
});
assert.ok(values.output, '需要全新的 --output 产物目录');
const output = resolve(values.output);
mkdirSync(output, { recursive: false });
const save = (name: string, data: unknown) =>
  writeFileSync(join(output, name), JSON.stringify(data, null, 2), {
    flag: 'wx',
  });
const digest = (data: string | Buffer) => createHash('sha256').update(data).digest('hex');
const canonical = <T>(value: T): T => JSON.parse(JSON.stringify(value));
const frozen = await build({
  entryPoints: ['scripts/training/native/validate.ts'],
  bundle: true,
  platform: 'node',
  format: 'esm',
  target: 'node22',
  write: false,
  metafile: true,
});
writeFileSync(join(output, 'runner.mjs'), frozen.outputFiles[0].contents, {
  flag: 'wx',
});
copyFileSync(values.executable, join(output, 'engine.exe'));
const nativePaths = [
  'Cargo.toml',
  'Cargo.lock',
  'src/main.rs',
  'src/model.rs',
  'src/core/geometry.rs',
  'src/core/shared.rs',
  'src/commands/movement.rs',
  'src/core/stats.rs',
  'src/core/resolution.rs',
  'src/commands/damage.rs',
  'src/commands/combat.rs',
  'src/commands/reactions.rs',
  'src/commands/preparation.rs',
  'src/commands/lifecycle.rs',
  'src/commands/abilities.rs',
  'src/commands/spells.rs',
  'src/setup/shrines.rs',
  'src/setup/synthesis.rs',
  'src/setup/runtime.rs',
  'src/training/actions.rs',
  'src/training/tree.rs',
  'src/training/encoding.rs',
  'src/training/policy.rs',
  'src/training/sampler.rs',
].map((p) => `native/engine-prototype/${p}`);
save('manifest.json', {
  head: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
  node: process.version,
  cpu: cpus()[0].model,
  rustc: execFileSync('rustc', ['--version'], { encoding: 'utf8' }).trim(),
  sources: Object.fromEntries(
    [...Object.keys(frozen.metafile.inputs), ...nativePaths].map((p) => [
      p,
      digest(readFileSync(p)),
    ]),
  ),
  nativeSource: Object.fromEntries(nativePaths.map((p) => [p, readFileSync(p, 'utf8')])),
  executableSha256: digest(readFileSync(values.executable)),
});

const client = await nativeClient(join(output, 'engine.exe'));
const counts = {
  successfulCommands: 0,
  invalidCommands: 0,
  inspections: 0,
  availableInspections: 0,
  unsupported: {} as Record<string, number>,
};
const unsupported = (result: any) => {
  if (result.status !== 'unsupported') return false;
  assert.fail(`完整引擎不能跳过规则：${JSON.stringify(result)}`);
};
function applyResult(s: GameState, c: Command) {
  try {
    return { status: 'available', state: applyCommand(s, c) };
  } catch (error) {
    if (error instanceof RuleError) return { status: 'invalid', message: error.message };
    throw error;
  }
}
function reference(job: Job) {
  return {
    inspections: job.probes.map((c) => inspectCommand(job.state, c)),
    ...(job.command ? { result: applyResult(job.state, job.command) } : {}),
  };
}
async function compare(jobs: Job[], label: string) {
  const before = JSON.stringify(jobs);
  const actual = await client.request({ op: 'run', jobs });
  assert.equal(actual.length, jobs.length);
  for (const [i, job] of jobs.entries()) {
    const got = actual[i];
    assert.equal(got.inspections.length, job.probes.length);
    for (const [j, command] of job.probes.entries()) {
      if (unsupported(got.inspections[j])) continue;
      counts.inspections++;
      if (got.inspections[j].status === 'available') counts.availableInspections++;
      assert.deepEqual(
        got.inspections[j],
        inspectCommand(job.state, command),
        `${label}:${i}:probe-${j}:${JSON.stringify(command)}`,
      );
    }
    if (job.command && !unsupported(got.result)) {
      const expected = canonical(applyResult(job.state, job.command));
      if (expected.status === 'available') counts.successfulCommands++;
      else counts.invalidCommands++;
      try {
        assert.deepEqual(
          got.result,
          expected,
          `${label}:${i}:command ${JSON.stringify(job.command)}`,
        );
      } catch (error) {
        save('mismatch.json', { label, job, got, expected });
        throw new Error(`${label}:${i}: 完整状态不同，详见 ${join(output, 'mismatch.json')}`);
      }
    }
  }
  assert.equal(JSON.stringify(jobs), before, `${label}: 修改输入`);
  return actual;
}

try {
  save('resident-contract.json', await validateResidentProtocol(client));
  const special = [
    ...fixtures(),
    ...combatFixtures(),
    ...preparationFixtures(),
    ...completeFixtures(),
  ].filter(({ name }) => !values.match || name.includes(values.match));
  for (const { name, job } of special) {
    const [result] = await compare([job], name);
    // 所有可用落点都执行并比较完整状态；非法命令也经正式入口检查原子性。
    const commands = job.probes.filter((_, i) => result.inspections[i].status !== 'unsupported');
    const outcomes = await compare(
      commands.map((command) => ({ state: job.state, probes: [], command })),
      `${name}-states`,
    );
    const reacted = new Set<string>();
    for (const outcome of outcomes) {
      if (outcome.result.status !== 'available') continue;
      const next = outcome.result.state as GameState;
      const probes = reactionProbes(next);
      const key = next.pending[0]?.kind;
      if (!probes.length || reacted.has(key)) continue;
      reacted.add(key);
      await compare([{ state: next, probes }], `${name}-reaction`);
      await compare(
        probes.map((command) => ({ state: next, probes: [], command })),
        `${name}-reaction-states`,
      );
    }
    const chosen = job.probes.find((_, i) => result.inspections[i].status === 'available');
    if (chosen) {
      const next = applyCommand(job.state, chosen);
      // 连续移动/操作耗尽在新状态上重新校验，不能沿用上一局面的查询缓存。
      await compare(
        [
          {
            state: next,
            probes: chosen.type === 'move' ? moveProbes(chosen.unitId!) : job.probes,
            command: chosen,
          },
        ],
        `${name}-next`,
      );
    }
  }
  console.log(JSON.stringify({ fixtures: special.length, ...counts }));
  save('fixtures.json', { cases: special.map((c) => c.name), ...counts });
  if (values['fixtures-only']) process.exitCode = 0;
  else {
    const records: any[] = [];
    const selected: { source: string; index: number; job: Job }[] = [];
    const attacks: { source: string; index: number; job: Job }[] = [];
    const windows: CommandWindow[] = [];
    const terminalGames: CommandWindow[] = [];
    const sources = ['classic-tiny', 'shrine-tiny', 'classic-uniform', 'shrine-uniform'];
    const files = sources.flatMap((group) =>
      Array.from({ length: values['all-workers'] ? 8 : 1 }, (_, worker) => ({
        group,
        worker,
      })),
    );
    for (const { group, worker } of files) {
      const source = `${group}/worker-${worker}`;
      const path = `artifacts/training/economics-20260926/${source}.jsonl.gz`;
      let state: GameState | undefined;
      let game = -1;
      let completeWindow: CommandWindow | undefined;
      let residentRevision = 0;
      let total = 0,
        eligible = 0,
        supported = 0;
      let chunk: Job[] = [];
      let indices: { index: number; game: number }[] = [];
      const outcomes: any[] = [];
      const roots: { index: number; job: Job }[] = [];
      const attackRoots: { index: number; job: Job }[] = [];
      const streaks: CommandWindow[] = [];
      const byCommand: Record<string, { total: number; supported: number }> = {};
      const unsupportedReasons: Record<string, number> = {};
      const flush = async () => {
        if (!chunk.length) return;
        const results = await compare(chunk, `${source}:${total}`);
        // 常驻局面从开局推进至记录末尾，每64条仅比较终态；独立差分已逐步比较全部事件。
        const continued = await client.request({
          op: 'step',
          revision: residentRevision,
          commands: chunk.map((job) => job.command),
          clearHistory: true,
        });
        assert.ok(
          continued.results.every((r: any) => r.status === 'available'),
          `${source}: 连续整局出现拒绝 ${JSON.stringify(continued.results)}`,
        );
        residentRevision = continued.revision;
        assert.deepEqual(
          (await client.request({ op: 'export' })).state,
          canonical(state),
          `${source}:${total}: 连续整局终态不同`,
        );
        for (const [i, result] of results.entries()) {
          if (result.result.status === 'available') {
            supported++;
            byCommand[chunk[i].command!.type].supported++;
            // 均匀取样之前只保存真实移动决策；保持未抽样的完整成功前缀差分。
            if (worker === 0) {
              if (chunk[i].command!.type === 'move')
                roots.push({ index: indices[i].index, job: chunk[i] });
              if (chunk[i].command!.type === 'attack')
                attackRoots.push({ index: indices[i].index, job: chunk[i] });
              const last = streaks.at(-1),
                { game, index } = indices[i];
              // 不越过任何未支持命令，每段最多64条；其他 worker 仅扩大正确性覆盖。
              if (
                last &&
                last.game === game &&
                last.start + last.commands.length === index &&
                last.commands.length < 64
              )
                last.commands.push(chunk[i].command!);
              else
                streaks.push({
                  source,
                  game,
                  start: index,
                  state: chunk[i].state,
                  commands: [chunk[i].command!],
                });
            }
          } else {
            assert.equal(result.result.status, 'unsupported', '真实成功命令不能被 Rust 判成非法');
            const reason = result.result.reason;
            unsupportedReasons[reason] = (unsupportedReasons[reason] ?? 0) + 1;
          }
        }
        chunk = [];
        indices = [];
      };
      for await (const row of readTrainingRecords(path)) {
        if (row.type === 'game') {
          await flush();
          assert.equal(row.source, 'neural');
          game = row.game;
          state = createGame(row.seed, row.rules);
          state.events = [];
          state.log = [];
          residentRevision = (await client.request({ op: 'reset', state })).revision;
          completeWindow = { source, game, start: 0, state, commands: [] };
        } else if (row.type === 'decision') {
          assert.ok(state);
          const tally = (byCommand[row.command.type] ??= {
            total: 0,
            supported: 0,
          });
          tally.total++;
          completeWindow!.commands.push(row.command);
          if (client.commands.includes(row.command.type)) {
            eligible++;
            chunk.push({ state, command: row.command, probes: [] });
            indices.push({ index: row.index, game });
          }
          state = applyPlayerCommand(state, row.actor, row.command);
          state.events = [];
          state.log = [];
          total++;
          if (chunk.length >= 64) await flush();
        } else if (row.type === 'outcome') {
          await flush();
          if (row.terminated) terminalGames.push(completeWindow!);
          outcomes.push({
            game: row.game,
            terminated: row.terminated,
            truncated: row.truncated,
            interrupted: row.interrupted ?? null,
            returns: row.returns,
          });
        }
      }
      await flush();
      const candidates = streaks.filter((w) => w.commands.length >= 2);
      if (worker === 0) {
        assert.ok(roots.length >= 8 && candidates.length >= 8, `真实样本不足：${source}`);
        assert.ok(attackRoots.length >= 8, `真实攻击样本不足：${source}`);
        for (let i = 0; i < 8; i++) {
          const root = roots[Math.floor(((i + 0.5) * roots.length) / 8)];
          const job = {
            ...root.job,
            probes: moveProbes(root.job.command!.unitId!),
          };
          // 性能集不混入未支持分支，计数单独保留，不能用 TS 回退掩盖。
          const [answer] = await compare([job], `${source}-queries-${i}`);
          assert.ok(answer.inspections.every((p: any) => p.status !== 'unsupported'));
          selected.push({ source, index: root.index, job });
          const attack = attackRoots[Math.floor(((i + 0.5) * attackRoots.length) / 8)];
          attacks.push({ source, ...attack });
          windows.push(candidates[Math.floor(((i + 0.5) * candidates.length) / 8)]);
        }
      }
      const report = {
        path,
        sha256: digest(readFileSync(path)),
        total,
        attemptedCommands: eligible,
        supported,
        otherCommandsNotPorted: total - eligible,
        unsupportedCommands: eligible - supported,
        byCommand,
        unsupportedReasons,
        residentCandidateLengths: candidates.map((w) => w.commands.length),
        outcomes,
      };
      records.push(report);
      console.log(JSON.stringify(report));
    }
    for (const window of windows) {
      try {
        await validateWindow(client, window);
      } catch (error) {
        save('resident-mismatch.json', window);
        throw error;
      }
    }
    assert.ok(counts.successfulCommands > 500);
    save('differential.json', {
      ...counts,
      records,
      selected: selected.map(({ source, index }) => ({ source, index })),
      attacks: attacks.map(({ source, index }) => ({ source, index })),
      residentWindows: windows.map(windowIdentity),
      continuousReplayAllCommands: true,
      terminalGames: terminalGames.map(windowIdentity),
      inputUnchanged: true,
      fullStatesEqual: true,
    });
    const jobs = selected.map((s) => s.job);
    const steps = jobs.map(({ state, command }) => ({
      state,
      command,
      probes: [],
    }));
    const timings: any[] = [];
    const clock = (run: () => unknown) => {
      const start = performance.now();
      run();
      return performance.now() - start;
    };
    for (const [name, workload] of [
      ['move-query-batch', jobs],
      ['single-step-batch', steps],
      ['attack-step-batch', attacks.map((s) => s.job)],
    ] as const) {
      const expected = canonical(workload.map(reference));
      assert.deepEqual(await client.request({ op: 'run', jobs: workload }), expected);
      await client.request({ op: 'load', jobs: workload });
      const repeats = name === 'move-query-batch' ? 5 : 100;
      const runTs = () => {
        for (let i = 0; i < repeats; i++) workload.map(reference);
      };
      runTs();
      await client.request({ op: 'bench', repeats });
      const rounds: any[] = [];
      for (let round = 0; round < 3; round++) {
        let tsMs: number, rustMs: number;
        if (round % 2 === 0) {
          tsMs = clock(runTs);
          rustMs = (await client.request({ op: 'bench', repeats })).elapsedMs;
        } else {
          rustMs = (await client.request({ op: 'bench', repeats })).elapsedMs;
          tsMs = clock(runTs);
        }
        // IPC计时包含请求对象序列化、管道、Rust解析/执行/序列化与Node解析。
        // 同时计一次 TS 的相同 JSON 输入/输出边界，另保留无通信 TS 时间。
        let tsPlainMs = 0,
          tsJsonMs = 0,
          rustIpcMs = 0;
        const timeTsBoundary = () => {
          tsPlainMs = clock(() => workload.map(reference));
          tsJsonMs = clock(() => canonical((canonical(workload) as Job[]).map(reference)));
        };
        const timeRustBoundary = async () => {
          const start = performance.now();
          const response = await client.request({ op: 'run', jobs: workload });
          rustIpcMs = performance.now() - start;
          assert.deepEqual(response, expected);
        };
        if (round % 2 === 0) {
          timeTsBoundary();
          await timeRustBoundary();
        } else {
          await timeRustBoundary();
          timeTsBoundary();
        }
        rounds.push({ tsMs, rustMs, tsPlainMs, tsJsonMs, rustIpcMs });
      }
      const median = (key: string) => rounds.map((r) => r[key]).sort((a, b) => a - b)[1];
      const result = {
        name,
        jobs: workload.length,
        probes: workload.reduce((n, j) => n + j.probes.length, 0),
        repeats,
        rounds,
        median: {
          tsMs: median('tsMs'),
          rustMs: median('rustMs'),
          coreSpeedup: median('tsMs') / median('rustMs'),
          tsPlainMs: median('tsPlainMs'),
          tsJsonMs: median('tsJsonMs'),
          rustIpcMs: median('rustIpcMs'),
          ipcVsPlain: median('tsPlainMs') / median('rustIpcMs'),
          ipcVsJson: median('tsJsonMs') / median('rustIpcMs'),
        },
      };
      timings.push(result);
      console.log(JSON.stringify(result));
    }
    const resident = await benchmarkWindows(client, windows);
    const completeGames = terminalGames.length
      ? await benchmarkWindows(client, terminalGames)
      : null;
    save('complete-games.json', {
      games: terminalGames.map(windowIdentity),
      benchmark: completeGames,
      includesSelection: false,
    });
    console.log(JSON.stringify(resident));
    save('benchmark.json', {
      timings,
      resident,
      completeGames,
      completeRuleEngine: true,
      completeSelfPlay: false,
      gpuTrainingIncluded: false,
      totalEconomicsPassed: false,
      note: '全部命令均原生执行；连续段与真实终局回放单列加载、执行、导出的通信成本。这里不含选招、编码或模型推理；完整采样另由 sample-benchmark.ts 测量。移动入口投影属性，收益不全归于语言。',
    });
  }
} finally {
  client.close();
}
