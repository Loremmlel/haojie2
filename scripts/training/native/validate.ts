import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { cpus } from 'node:os';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { build } from 'esbuild';
import { applyCommand, createGame, inspectCommand } from '../../../src/engine/commands/game';
import { RuleError } from '../../../src/engine/core/state';
import { nativeClient } from './client';
import { fixtures, moveProbes } from './fixtures';
import { combatFixtures, reactionProbes } from './combat-fixtures';
import { preparationFixtures } from './preparation-fixtures';
import { completeFixtures } from './complete-fixtures';
import { nativeSourcePaths } from './sampling/artifacts';
import { validateResidentProtocol } from './resident';
import type { Job } from './fixtures';
import type { Command, GameState } from '../../../src/engine/types';

const { values } = parseArgs({
  options: {
    output: { type: 'string' },
    executable: {
      type: 'string',
      default: `artifacts/native-target/release/haojie-engine${process.platform === 'win32' ? '.exe' : ''}`,
    },
    // 兼容现有 CI 命令；默认已只运行可复现的规则夹具。
    'fixtures-only': { type: 'boolean', default: false },
    match: { type: 'string' },
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
const nativePaths = nativeSourcePaths();
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
} finally {
  client.close();
}
