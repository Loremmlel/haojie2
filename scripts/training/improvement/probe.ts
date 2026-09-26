import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { parseArgs, isDeepStrictEqual } from 'node:util';
import { PythonPolicy } from '../python-policy';
import { readTrainingRecords } from '../records/replay';
import { hashRecordFile } from '../records/io';
import { encodingSourceHash } from '../encode';
import { canonicalTrainingCommand } from '../../../src/ai/training/action-tree';
import { decodeCommand } from '../../../src/ai/training/decoder';
import { beamDecode } from './beam';
import { policySearch } from './search';

/**
 * 开发种子上的离线推理对照；先按指纹哈希冻结局面，再读取模型输出，不按分歧挑题。
 * 只比较教师完整命令一致率与真实成本，不把教师选择当成最优解或整局棋力。
 * 原局面仅在内存中重放，落盘引用和指纹；模型只接收公开编码，测试集不进入训练。
 */
const { values } = parseArgs({
  options: {
    inputs: { type: 'string', multiple: true },
    output: { type: 'string' },
    checkpoint: { type: 'string' },
    module: { type: 'string', default: 'scripts.training.improvement.serve' },
    count: { type: 'string', default: '64' },
  },
});
assert.ok(values.inputs?.length && values.output && values.checkpoint);
const count = Number(values.count);
assert.ok(Number.isSafeInteger(count) && count > 0);
mkdirSync(values.output, { recursive: true });
const write = (name: string, data: unknown) =>
  writeFileSync(resolve(values.output!, name), JSON.stringify(data, null, 2), { flag: 'wx' });
const digestSource = (path: string) =>
  createHash('sha256').update(readFileSync(path)).digest('hex');
const sourceFiles = [
  'scripts/training/improvement/probe.ts',
  'scripts/training/improvement/beam.ts',
  'scripts/training/improvement/search.ts',
  'scripts/training/improvement/model.py',
  'scripts/training/search/puct.ts',
  'scripts/training/search/recovery/spatial.py',
  'scripts/training/python-policy.ts',
  'training/haojie_training/model.py',
  'training/haojie_training/inference.py',
  'training/haojie_training/data.py',
  'training/haojie_training/runtime.py',
  values.module!.replaceAll('.', '/') + '.py',
];
const sourceHashes = Object.fromEntries(sourceFiles.map((path) => [path, digestSource(path)]));
for (const path of sourceFiles) {
  const target = resolve(values.output, 'source', path);
  mkdirSync(dirname(target), { recursive: true });
  writeFileSync(target, readFileSync(path), { flag: 'wx' });
}
const selected: any[] = [];
const inputs = [];
for (const path of values.inputs) {
  inputs.push({ path, sha256: await hashRecordFile(path) });
  let game: any;
  for await (const row of readTrainingRecords(path)) {
    if (row.type === 'game') game = row;
    else if (
      row.type === 'sample' &&
      (game.teachers?.[row.actor]?.difficulty ?? game.difficulty) === 'hard'
    ) {
      if (row.observation.phase !== 'play' && !row.observation.pending.length) continue;
      const rank = createHash('sha256')
        .update(`${game.seed}:${row.before}:${row.actor}`)
        .digest('hex');
      selected.push({ ...row, path, seed: game.seed, rank });
      selected.sort((a, b) => a.rank.localeCompare(b.rank));
      if (selected.length > count) selected.pop();
    }
  }
}
assert.equal(selected.length, count);
write('protocol.json', {
  inputs,
  checkpoint: values.checkpoint,
  checkpointSha256: createHash('sha256').update(readFileSync(values.checkpoint)).digest('hex'),
  encodingSourceSha256: encodingSourceHash(),
  sourceHashes,
  count,
  simulations: 16,
  horizon: 2,
  positions: selected.map(({ observation, teacherStats, ...row }) => ({
    path: row.path,
    seed: row.seed,
    game: row.game,
    index: row.index,
    before: row.before,
    actor: row.actor,
    rank: row.rank,
  })),
});
const policy = await PythonPolicy.start({
  python:
    process.platform === 'win32'
      ? 'training/.venv/Scripts/python.exe'
      : 'training/.venv/bin/python',
  checkpoint: values.checkpoint,
  module: values.module,
  device: 'cpu',
  precision: 'fp32',
  threads: 1,
  timeoutMs: 60000,
});
const modes = ['greedy', 'beam', 'mcts', 'gumbel'] as const;
const results: any[] = [];
try {
  for (const [index, row] of selected.entries()) {
    const expected = canonicalTrainingCommand(row.observation, row.actor, row.command);
    for (const mode of modes) {
      const started = performance.now();
      const decoded =
        mode === 'greedy'
          ? await decodeCommand(row.observation, row.actor, policy.evaluate)
          : mode === 'beam'
            ? await beamDecode(row.observation, row.actor, policy.evaluate)
            : await policySearch(row.observation, row.actor, policy.evaluate, { mode });
      results.push({
        index,
        mode,
        before: row.before,
        milliseconds: performance.now() - started,
        expected,
        decoded,
        matchesTeacher:
          !!decoded.command &&
          isDeepStrictEqual(
            canonicalTrainingCommand(row.observation, row.actor, decoded.command),
            expected,
          ),
      });
    }
    if ((index + 1) % 8 === 0) console.log(JSON.stringify({ positions: index + 1 }));
  }
} finally {
  policy.close();
}
write('results.json', results);
for (const [path, sha] of Object.entries(sourceHashes))
  assert.equal(digestSource(path), sha, '推理期间源码改变');
for (const input of inputs)
  assert.equal(await hashRecordFile(input.path), input.sha256, '推理期间原轨迹改变');
const summary = modes.map((mode) => {
  const rows = results.filter((r) => r.mode === mode);
  return {
    mode,
    positions: rows.length,
    matchesTeacher: rows.filter((r) => r.matchesTeacher).length,
    paused: rows.filter((r) => r.decoded.status !== 'command').length,
    ends: rows.filter((r) => r.decoded.command?.type === 'end').length,
    teacherEnds: rows.filter((r) => r.expected.type === 'end').length,
    inferenceCalls: rows.reduce((sum, r) => sum + r.decoded.stats.evaluations, 0),
    milliseconds: rows.reduce((sum, r) => sum + r.milliseconds, 0),
    searchDecisions: rows.filter(
      (r) => r.decoded.search && !r.decoded.search.fallback && r.decoded.status === 'command',
    ).length,
    fallback: Object.fromEntries(
      [
        ...new Set(
          rows.flatMap((r) => (r.decoded.search?.fallback ? [r.decoded.search.fallback] : [])),
        ),
      ].map((reason) => [reason, rows.filter((r) => r.decoded.search?.fallback === reason).length]),
    ),
    simulations: rows.reduce((sum, r) => sum + (r.decoded.search?.stats.simulations ?? 0), 0),
    transitions: rows.reduce((sum, r) => sum + (r.decoded.search?.stats.transitions ?? 0), 0),
    terminalLeaves: rows.reduce((sum, r) => sum + (r.decoded.search?.stats.terminalLeaves ?? 0), 0),
  };
});
write('summary.json', { summary, inference: policy.totals, model: policy.ready });
console.log(JSON.stringify(summary, null, 2));
