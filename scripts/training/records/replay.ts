import assert from 'node:assert/strict';
import { TrainingEnvironment } from '../../../src/match/training';
import { parseSession } from '../../../src/engine';
import { RULESET_ID } from '../../../src/engine/catalog';
import { HAOJIE_RULESET } from '../../../src/engine/online/player-view';
import { decisionOwner, fingerprint } from '../../../src/ai/observation';
import { readRecordLines } from './io';

export const TRAINING_RECORD_FORMAT = 'haojie-training-record-v1';

export function recordHeader(env: TrainingEnvironment) {
  return {
    format: TRAINING_RECORD_FORMAT,
    ruleset: HAOJIE_RULESET,
    recordRuleset: RULESET_ID,
    limits: env.limits(),
  };
}

/**
 * 唯一训练轨迹读取边界：版本、权限、顺序、前后指纹和结果必须通过正式引擎重放。
 * 磁盘不含逐步观察；只在内存补出操作者白名单，种子/初始存档绝不进入编码器。
 * 缺少结束行的完整命令前缀只允许策略监督，不从最后局面猜测价值标签。
 */
export async function* readTrainingRecords(path: string): AsyncGenerator<any> {
  let env: TrainingEnvironment | undefined;
  let header: any;
  const games = new Set<number>();
  const interrupted = () => ({
    type: 'outcome',
    game: header.game,
    ...env!.status(),
    terminated: false,
    truncated: false,
    truncation: null,
    winner: null,
    returns: null,
    interrupted: 'missing-outcome',
    after: fingerprint(env!.observation()),
  });
  for await (const row of readRecordLines(path)) {
    assert.ok(row && typeof row === 'object', '训练记录必须是对象。');
    assert.ok(!('observation' in row), '旧快照训练记录不再支持，请重新生成。');
    if (row.type === 'game') {
      if (env) yield interrupted();
      assert.equal(row.format, TRAINING_RECORD_FORMAT, '训练记录格式不支持，请重新生成。');
      assert.equal(row.ruleset, HAOJIE_RULESET, '训练规则版本不匹配，请重新生成。');
      assert.equal(row.recordRuleset, RULESET_ID, '重放规则版本不匹配，请重新生成。');
      assert.ok(['teacher', 'saved-game', 'neural'].includes(row.source), '未知训练记录来源。');
      assert.ok(
        Number.isSafeInteger(row.game) && row.game >= 0 && !games.has(row.game),
        '对局编号重复或无效。',
      );
      assert.ok(
        row.gameId === undefined ||
          (typeof row.gameId === 'string' && row.gameId.length > 0 && row.gameId.length <= 512),
        '对局标识无效。',
      );
      for (const key of ['maxCommands', 'maxPlies'])
        assert.ok(
          Number.isSafeInteger(row.limits?.[key]) && row.limits[key] > 0,
          '缺少实际训练上限。',
        );
      const limits = { maxCommands: row.limits.maxCommands, maxPlies: row.limits.maxPlies };
      env = new TrainingEnvironment({ seed: row.seed, rules: row.rules, ...limits });
      assert.ok(row.rules === 'classic' || row.rules === 'shrine', '未知规则模式。');
      assert.ok(Number.isSafeInteger(row.seed), '缺少复现种子。');
      if (row.source === 'saved-game') {
        const initial = parseSession(JSON.stringify(row.initial)).present;
        assert.equal(initial.seed, row.seed);
        assert.equal(initial.mode ?? 'classic', row.rules);
        env = TrainingEnvironment.fromState(initial, limits);
      } else assert.equal(row.initial, undefined, '种子开局不能同时指定局面。');
      header = row;
      games.add(row.game);
      yield row;
      continue;
    }
    assert.ok(env, '记录缺少对局头。');
    assert.equal(row.game, header.game, '记录不属于当前对局。');
    if (row.type === 'outcome') {
      const status = env.status();
      for (const key of Object.keys(status) as (keyof typeof status)[])
        assert.deepEqual(row[key], status[key], `结束状态不匹配：${key}`);
      assert.equal(row.after, fingerprint(env.observation()), '结束指纹不匹配。');
      const wasInterrupted =
        row.interrupted === true ||
        (typeof row.interrupted === 'string' && row.interrupted.length > 0);
      assert.ok(
        row.interrupted == null || row.interrupted === false || wasInterrupted,
        '中断标记无效。',
      );
      assert.equal(
        Number(status.terminated) + Number(status.truncated) + Number(wasInterrupted),
        1,
        '结束原因必须唯一。',
      );
      yield row;
      env = undefined;
      continue;
    }
    const commandRow = row.type === 'sample' || row.type === 'decision';
    assert.ok(
      commandRow || ['pause', 'error', 'rejected'].includes(row.type),
      '未知训练记录类型。',
    );
    assert.equal(
      row.type === 'sample' ? header.source !== 'neural' : header.source === 'neural',
      true,
      '记录类型与来源不匹配。',
    );
    assert.equal(row.index, env.status().commands, '命令序号不连续。');
    assert.ok(row.actor === 1 || row.actor === 2, '操作者无效。');
    const observation = env.observation(row.actor);
    // 教师旧协议按主决策方出手；新神经策略还可合法暗选或回合外巨大化。
    // 实际权限仍由下面的 env.step / actorCommandError 逐条裁定，不信任记录声称的席位。
    if (header.source === 'teacher')
      assert.equal(row.actor, decisionOwner(observation), '教师操作者不匹配。');
    assert.equal(row.before, fingerprint(observation), '命令前指纹不匹配。');
    if (commandRow) {
      env.step(row.actor, row.command);
      assert.equal(row.after, fingerprint(env.observation()), '命令后指纹不匹配。');
    }
    yield { ...row, observation };
  }
  if (env) yield interrupted();
}
