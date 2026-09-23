import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { savedGameSamples } from '../../scripts/training/import/save';
import { encodeTeacherFile } from '../../scripts/training/encode';
import { inspectTeacherFiles } from '../../scripts/training/inspect-teacher';
import { readTrainingRecords } from '../../scripts/training/records/replay';
import {
  createGame,
  createSession,
  dispatch,
  parseSession,
  serializeSession,
  type Session,
} from '../../src/engine';
import { add, fixture } from '../helpers';

async function verifyPipeline(session: Session) {
  const rows: any[] = [...savedGameSamples(serializeSession(session))];
  const folder = await mkdtemp(join(tmpdir(), 'haojie-save-training-'));
  try {
    const path = join(folder, 'samples.jsonl');
    await writeFile(path, rows.map((row) => JSON.stringify(row)).join('\n'));
    const report = await inspectTeacherFiles([path], true);
    assert.equal(report.counts.commands, session.record!.cursor);
    assert.equal(report.profiles['saved-game:unrated'].unknownSearchStats, session.record!.cursor);
    const encoded: any[] = [];
    for await (const row of encodeTeacherFile(path)) encoded.push(row);
    assert.ok(encoded.some((r) => r.type === 'example'));
    assert.ok(rows.every((r) => !('observation' in r)));
    const restored: any[] = [];
    for await (const row of readTrainingRecords(path)) restored.push(row);
    for (const row of restored.filter((r) => r.type === 'sample')) {
      assert.equal('rng' in row.observation, false);
      assert.equal('seed' in row.observation, false);
      assert.equal(row.teacherStats, undefined);
    }
    return restored;
  } finally {
    await rm(folder, { recursive: true, force: true });
  }
}

test('网页增量存档的79步记录可审计并进入现有训练编码器', async () => {
  const [header, ...entries] = (
    await readFile('docs/playtests/current/cli-feedback5-20260923.jsonl', 'utf8')
  )
    .trim()
    .split('\n')
    .map((line) => JSON.parse(line));
  let session = parseSession(JSON.stringify(header.initial));
  for (const entry of entries) session = dispatch(session, entry.command);
  const rows = await verifyPipeline(session);
  const outcome = rows.at(-1);
  assert.equal(outcome.interrupted, 'saved-before-terminal');
  assert.equal(outcome.returns, null);
  assert.equal(outcome.winner, null);
});

test('神龛存档从第二方先锁定也按实际视角生成样本，不泄露未揭示选择', async () => {
  const state = createGame(90, 'shrine');
  let session = createSession(state);
  for (const player of [2, 1] as const)
    session = dispatch(session, {
      type: 'choose-shrine',
      player,
      shrineKind: state.shrineDraft!.offers[player][0],
      parity: 'even',
    });
  const rows = await verifyPipeline(session);
  const samples = rows.filter((r) => r.type === 'sample');
  assert.deepEqual(
    samples.map((r) => r.actor),
    [2, 1],
  );
  assert.equal(samples[1].observation.shrineDraft.choices[2], undefined);
  assert.equal(samples[1].observation.shrineDraft.committed[2], true);
});

test('中途起录保留回合外巨大化与敌方反应的操作者，终局标签只来自真实结果', async () => {
  const giant = fixture();
  const own = add(giant, 'u7', 1, 2, 3);
  const target = add(giant, 1, 2, 5, 7);
  giant.active = 2;
  const enlarged = dispatch(createSession(giant), {
    type: 'skill',
    unitId: own.id,
    targetId: target.id,
    x: 4,
    y: 6,
  });
  const giantRows = await verifyPipeline(enlarged);
  assert.equal(giantRows[0].origin, 'position');
  assert.equal(giantRows[1].actor, 1);

  const reaction = fixture();
  const attacker = add(reaction, 26, 1, 3, 4);
  const victim = add(reaction, 2, 2, 3, 5);
  victim.hp = 1;
  let session = dispatch(createSession(reaction), {
    type: 'attack',
    unitId: attacker.id,
    targetId: victim.id,
  });
  session = dispatch(session, { type: 'react' });
  const reactionRows = await verifyPipeline(session);
  assert.deepEqual(
    reactionRows.filter((r) => r.type === 'sample').map((r) => r.actor),
    [1, 2],
  );

  const terminal = fixture();
  terminal.bases[2] = 10;
  const finisher = add(terminal, 26, 1, 5, 12);
  const won = dispatch(createSession(terminal), {
    type: 'attack',
    unitId: finisher.id,
    targetId: 'base-2',
  });
  const wonRows = await verifyPipeline(won);
  assert.deepEqual(wonRows.at(-1).returns, { 1: 1, 2: -1 });
  assert.equal(wonRows.at(-1).interrupted, null);
  assert.throws(() => [...savedGameSamples(JSON.stringify(createSession(terminal)))], /没有可转换/);
});
