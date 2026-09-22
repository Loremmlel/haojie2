import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { runSelfPlay } from '../../scripts/training/self-play';
import { encodeTeacherFile } from '../../scripts/training/encode';
import { TrainingEnvironment } from '../../src/match/training';
import { TrainingTeacher } from '../../src/ai/training/teacher';
import { decisionOwner } from '../../src/ai/observation';
import { observe } from '../../src/ai/observation';
import { auditPositions } from '../../scripts/training/teacher-audit';
import { inspectTeacherFiles } from '../../scripts/training/inspect-teacher';
import { add, fixture } from '../helpers';
import { applyCommand } from '../../src/engine';

test('混合教师交换席位，共用种子但轨迹ID不同；同源编码分组保持相同', async () => {
  await assert.rejects(
    runSelfPlay({ games: 2, difficulty: 'hard', opponent: { difficulty: 'hard' } }),
    /相同教师配置/,
  );
  const rows: any[] = [];
  await runSelfPlay(
    {
      games: 2,
      seed: 19,
      difficulty: 'hard',
      simulations: 40,
      opponent: { difficulty: 'medium', simulations: 40 },
      maxCommands: 4,
    },
    async (row) => {
      rows.push(row);
    },
  );
  const games = rows.filter((r) => r.type === 'game');
  assert.deepEqual(
    games.map((r) => [r.seed, r.primaryPlayer, r.teachers[1].difficulty]),
    [
      [19, 1, 'hard'],
      [19, 2, 'medium'],
    ],
  );
  assert.notEqual(games[0].gameId, games[1].gameId);
  assert.ok(
    rows.filter((r) => r.type === 'outcome').every((r) => r.truncated && r.returns === null),
  );
  const folder = await mkdtemp(join(tmpdir(), 'haojie-teachers-'));
  try {
    const path = join(folder, 'teacher.jsonl');
    await writeFile(path, rows.map((r) => JSON.stringify(r)).join('\n'));
    const encoded: any[] = [];
    for await (const row of encodeTeacherFile(path)) if (row.type === 'game') encoded.push(row);
    assert.equal(encoded[0].group, encoded[1].group);
    assert.notEqual(encoded[0].game_id, encoded[1].game_id);
    const report = await inspectTeacherFiles([path], true);
    assert.equal(report.counts.commands, 8);
    assert.equal(report.counts.truncated, 2);
    assert.equal(report.counts.primaryWins + report.counts.secondaryWins, 0);
    assert.deepEqual(Object.keys(report.profiles).sort(), ['hard:40', 'medium:40']);
    assert.equal(report.encodingChecked, true);
    for (const profile of Object.values(report.profiles)) {
      assert.equal(profile.encoding.commands, profile.commands);
      assert.ok(profile.encoding.examples >= profile.commands);
    }
    assert.equal(
      Object.values(report.profiles).reduce((sum, p) => sum + p.outcomeCommands.unknown, 0),
      8,
    );
  } finally {
    await rm(folder, { recursive: true });
  }
});

test('教师审计使用相同公开局面并复查确定性；明确的一击胜利经引擎验证', async () => {
  const state = fixture();
  state.bases[2] = 10;
  add(state, 26, 1, 5, 12);
  const rows = await auditPositions(
    [{ id: 'one-hit-win', observation: observe(state) }],
    [
      { difficulty: 'medium', simulations: 40 },
      { difficulty: 'hard', simulations: 80 },
    ],
  );
  assert.equal(rows.length, 2);
  assert.equal(rows[0].before, rows[1].before);
  for (const row of rows) assert.equal(applyCommand(state, row.command).winner, 1);
});

test('默认教师自对弈保留原有决策', async () => {
  const env = new TrainingEnvironment({ seed: 31 }),
    teacher = new TrainingTeacher('easy', 40);
  const expected = [];
  for (let n = 0; n < 12; n++) {
    const o = env.observation(),
      command = teacher.next(o).command;
    expected.push(command);
    env.step(decisionOwner(o), command);
  }
  const actual: unknown[] = [];
  await runSelfPlay({ games: 1, seed: 31, simulations: 40, maxCommands: 12 }, async (value) => {
    const row = value as any;
    if (row.type === 'sample') actual.push(row.command);
  });
  assert.deepEqual(actual, expected);
});
