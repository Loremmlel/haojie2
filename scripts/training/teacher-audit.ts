import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { readTrainingRecords } from './records/replay';
import { hashRecordFile } from './records/io';
import { createHash } from 'node:crypto';
import { cpus } from 'node:os';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { setImmediate } from 'node:timers/promises';
import { decide } from '../../src/ai/planning/search';
import { decisionOwner, fingerprint } from '../../src/ai/observation';
import type { Observation } from '../../src/ai/types';
import { inspectTrainingCommand } from '../../src/ai/training/queries';
import { HAOJIE_RULESET } from '../../src/engine/online/player-view';
import { encodingSourceHash } from './encode';
import type { TeacherProfile } from './self-play';

interface Position {
  id: string;
  observation: Observation;
}
export const AUDIT_PROFILES: TeacherProfile[] = [
  { difficulty: 'medium', simulations: 320 },
  { difficulty: 'hard', simulations: 800 },
  { difficulty: 'hard', simulations: 1600 },
];

/**
 * 历史教师公开局面上的独立冷计划审计；不继承预算历史、缓存或正式随机状态。
 * 对同一输入重复决策核对确定性，公开预检的uncertain仍不等于实局必定成功。
 * 不用教师自己的分数判断另一教师是否更强，也不把命令一致率当作正确率。
 */
export async function auditPositions(
  positions: Position[],
  profiles = AUDIT_PROFILES,
  signal?: AbortSignal,
) {
  const rows = [];
  for (const position of positions) {
    const actor = decisionOwner(position.observation),
      before = fingerprint(position.observation);
    for (const profile of profiles) {
      await setImmediate();
      if (signal?.aborted) throw new Error('教师审计已取消，未完成的报告不落盘');
      const limits = { simulations: profile.simulations, mode: 'work' as const, trace: false };
      const start = performance.now();
      const decision = decide(position.observation, actor, profile.difficulty, limits);
      const elapsedMs = performance.now() - start;
      assert.ok(decision.command, '教师没有返回命令');
      const checked = inspectTrainingCommand(position.observation, actor, decision.command);
      assert.notEqual(checked.status, 'invalid');
      const repeated = decide(position.observation, actor, profile.difficulty, limits);
      assert.deepEqual(repeated, decision, '相同公开输入与固定预算的教师结果漂移');
      assert.equal(fingerprint(position.observation), before, '教师改变了公开输入');
      rows.push({
        id: position.id,
        before,
        actor,
        units: position.observation.units.length,
        phase: position.observation.phase,
        profile,
        command: decision.command,
        plan: decision.plan,
        stats: decision.stats,
        publicStatus: checked.status,
        elapsedMs,
      });
    }
  }
  return rows;
}

/** 每局/回合/操作者仅取第一个非缓存play样本，再按指纹哈希选固定数量，内存有界。 */
async function selectPositions(path: string, count: number) {
  const selected: (Position & { rank: string })[] = [],
    seen = new Set<string>();
  let game: any,
    eligible = 0;
  for await (const row of readTrainingRecords(path)) {
    if (row.type === 'game') {
      assert.equal(row.ruleset, HAOJIE_RULESET, '旧规则样本不能用于当前教师审计');
      game = row;
      seen.clear();
    }
    if (
      row.type !== 'sample' ||
      row.observation.phase !== 'play' ||
      row.observation.units.length < 4 ||
      row.teacherStats?.cached ||
      !(row.teacherStats?.candidates > 1)
    )
      continue;
    assert.equal(row.game, game.game);
    const key = `${game.game}:${row.observation.ply}:${row.actor}`;
    if (seen.has(key)) continue;
    seen.add(key);
    eligible++;
    const id = `${game.gameId ?? game.seed}:${row.index}`;
    selected.push({
      id,
      observation: row.observation,
      rank: createHash('sha256').update(id).update(fingerprint(row.observation)).digest('hex'),
    });
    selected.sort((a, b) => a.rank.localeCompare(b.rank));
    if (selected.length > count) selected.pop();
  }
  assert.ok(selected.length, '语料中没有适合审计的非缓存局面');
  return {
    positions: selected.map(({ rank: _, ...p }) => p),
    eligible,
    sha256: await hashRecordFile(path),
  };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { values } = parseArgs({
    options: {
      input: { type: 'string' },
      output: { type: 'string' },
      positions: { type: 'string', default: '32' },
    },
  });
  assert.ok(values.input && values.output, '请提供--input和新的--output文件');
  assert.ok(!existsSync(values.output), '报告已存在，请保留旧实验。');
  mkdirSync(dirname(values.output), { recursive: true });
  const count = Number(values.positions);
  assert.ok(Number.isSafeInteger(count) && count >= 1 && count <= 512);
  const cancellation = new AbortController();
  process.once('SIGINT', () => cancellation.abort());
  const sourceSha256 = createHash('sha256')
    .update(encodingSourceHash())
    .update(readFileSync(fileURLToPath(import.meta.url), 'utf8').replaceAll('\r\n', '\n'))
    .digest('hex');
  const selection = await selectPositions(values.input, count);
  const rows = await auditPositions(selection.positions, AUDIT_PROFILES, cancellation.signal);
  const changed = selection.positions.filter((p) => {
    const choices = rows.filter((r) => r.id === p.id && r.profile.difficulty === 'hard');
    return JSON.stringify(choices[0].command) !== JSON.stringify(choices[1].command);
  }).length;
  const report = {
    format: 'haojie-teacher-audit-v1',
    created: new Date().toISOString(),
    input: values.input,
    inputSha256: selection.sha256,
    eligible: selection.eligible,
    sourceSha256,
    runtime: { node: process.version, cpu: cpus()[0].model },
    positions: selection.positions.length,
    changedHardBudgetCommands: changed,
    note: '相同公开局面、独立新计划；首次决策计时，确定性复查另算。选招不同不表示1600更好。',
    rows,
  };
  writeFileSync(values.output, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
  console.log(
    JSON.stringify({ output: values.output, positions: report.positions, changed }, null, 2),
  );
}
