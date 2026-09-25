import assert from 'node:assert/strict';
import { dirname, resolve } from 'node:path';
import { readRecordLines, hashRecordFile } from '../records/io';
import { readTrainingRecords } from '../records/replay';
import { HAOJIE_RULESET } from '../../../src/engine/online/player-view';
import { RULESET_ID } from '../../../src/engine/catalog';
import { inspectTrainingCommand } from '../../../src/ai/training/queries';
import { TrainingActionTree } from '../../../src/ai/training/action-tree';

export const CORRECTION_FORMAT = 'haojie-teacher-corrections-v1';

/**
 * 纠错文件只保存轨迹引用与教师命令；先验完整文件，再按共享引擎重放恢复公开输入。
 * 反事实标签没有实际续局，始终禁止继承学生胜负；缺失尾行/重复/错位标签直接失败。
 */
export async function* readCorrections(path: string): AsyncGenerator<any> {
  let header: any, footer: any;
  const labels = new Map<string, any>();
  for await (const row of readRecordLines(path)) {
    assert.ok(!footer, '纠错结束行之后存在数据');
    assert.ok(!('observation' in row) && !('returns' in row), '纠错不能保存快照或胜负');
    if (!header) {
      assert.equal(row.format, CORRECTION_FORMAT);
      assert.equal(row.type, 'corrections');
      assert.equal(row.ruleset, HAOJIE_RULESET);
      assert.equal(row.recordRuleset, RULESET_ID);
      assert.equal(row.teacher.difficulty, 'hard');
      assert.ok(Number.isSafeInteger(row.teacher.simulations) && row.teacher.simulations >= 40);
      header = row;
    } else if (row.type === 'label') {
      assert.ok(Number.isSafeInteger(row.game) && row.game >= 0);
      assert.ok(Number.isSafeInteger(row.index) && row.index >= 0);
      const key = `${row.game}:${row.index}`;
      assert.ok(!labels.has(key), '重复纠错位置');
      labels.set(key, row);
    } else {
      assert.equal(row.type, 'complete');
      footer = row;
    }
  }
  assert.ok(header && footer && labels.size > 0, '纠错文件未完成或为空');
  assert.equal(footer.labels, labels.size);
  const source = resolve(dirname(path), header.source.path);
  assert.equal(await hashRecordFile(source), header.source.sha256, '纠错原轨迹指纹改变');
  const identity = await hashRecordFile(path);
  let game: any,
    selected = 0,
    emitted = false;
  for await (const row of readTrainingRecords(source)) {
    if (row.type === 'game') {
      assert.equal(row.source, 'neural', '纠错来源必须是学生实际对局');
      game = row;
      selected = 0;
      emitted = false;
    } else if (row.type === 'decision') {
      const key = `${row.game}:${row.index}`,
        label = labels.get(key);
      if (!label) continue;
      assert.equal(row.policy, 'network', '不能把对手决策标成学生纠错');
      assert.equal(label.actor, row.actor);
      assert.equal(label.before, row.before, '纠错位置指纹不匹配');
      assert.notEqual(
        inspectTrainingCommand(row.observation, row.actor, label.command).status,
        'invalid',
      );
      new TrainingActionTree(row.observation, row.actor).trace(label.command);
      if (!emitted) {
        yield {
          ...game,
          source: 'correction',
          gameId: `correction:${identity}:${game.game}`,
          difficulty: header.teacher.difficulty,
          budget: header.teacher.simulations,
          origin: { source: header.source, game: game.game, policyOnly: true },
        };
        emitted = true;
      }
      yield { ...row, type: 'sample', command: label.command, teacherStats: label.stats };
      selected++;
      labels.delete(key);
    } else if (row.type === 'outcome' && emitted) {
      yield {
        type: 'outcome',
        game: game.game,
        commands: selected,
        policyOnly: true,
        terminated: false,
        truncated: false,
        interrupted: false,
        returns: null,
        winner: null,
      };
    }
  }
  assert.equal(labels.size, 0, '纠错位置不在完整源轨迹中');
}
