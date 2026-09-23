import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createSession, parseSession, sessionSave } from '../../../src/engine';
import { recordedActor } from '../../../src/engine/session/recording';
import { TrainingEnvironment } from '../../../src/match/training';
import { fingerprint } from '../../../src/ai/observation';
import { ensure } from '../../../src/engine/core/state';
import { recordHeader } from '../records/replay';
import { withRecordOutput } from '../records/io';

/**
 * 已验证的存档转换为增量训练轨迹；操作者观察由统一读取入口重放恢复。
 * 起点可能是旧档/演示局，明确保存起点；未结束局只给策略标签，不推断胜负。
 * 生成器不执行IO，不伪造教师评分、搜索统计或耗时。
 */
export function* savedGameSamples(text: string) {
  const session = parseSession(text);
  const record = session.record;
  ensure(record && record.cursor > 0, '存档没有可转换的命令记录；旧快照不能补出此前动作。');
  const initial = sessionSave(createSession(record.initial));
  const env = TrainingEnvironment.fromState(record.initial, {
    maxCommands: 100_001,
    maxPlies: 1e9,
  });
  const gameId = `save:${createHash('sha256')
    .update(JSON.stringify(sessionSave(session)))
    .digest('hex')}`;
  yield {
    type: 'game',
    game: 0,
    gameId,
    source: 'saved-game',
    origin: record.origin,
    ...recordHeader(env),
    rules: record.initial.mode ?? 'classic',
    seed: record.initial.seed,
    initial,
  };
  // 使用权威局面确定操作者，但只经 observation(viewer) 导出白名单。
  let state: Parameters<typeof recordedActor>[0] = record.initial;
  for (const [index, command] of record.commands.slice(0, record.cursor).entries()) {
    const actor = recordedActor(state, command);
    const observation = env.observation(actor);
    env.step(actor, command);
    // 操作者查询只需要棋子身份、当前阶段和反应；环境观察不含正式随机字段。
    const next = env.observation();
    state = next;
    yield {
      type: 'sample',
      game: 0,
      index,
      actor,
      before: fingerprint(observation),
      command,
      after: fingerprint(next),
    };
  }
  const status = env.status();
  yield {
    type: 'outcome',
    game: 0,
    ...status,
    after: fingerprint(env.observation()),
    interrupted: status.terminated ? null : 'saved-before-terminal',
  };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  ensure(process.argv.length === 4, '用法：npm run train:import-save -- 存档.json 输出.jsonl.gz');
  const rows = savedGameSamples(readFileSync(process.argv[2], 'utf8'));
  // 先验证存档，再独占创建输出；任何转换失败都明确退出，不覆盖已有数据集。
  const first = rows.next();
  await withRecordOutput(process.argv[3], async (emit) => {
    if (!first.done) await emit(first.value);
    for (const row of rows) await emit(row);
  });
}
