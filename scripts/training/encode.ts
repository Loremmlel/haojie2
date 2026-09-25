import { createHash } from 'node:crypto';
import { once } from 'node:events';
import { readFileSync, readdirSync } from 'node:fs';
import { relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { TrainingActionTree } from '../../src/ai/training/action-tree';
import { encodeDecision } from '../../src/ai/training/encoding/decision';
import { ENCODING_SCHEMA } from '../../src/ai/training/encoding/schema';
import { HAOJIE_RULESET } from '../../src/engine/online/player-view';
import { ensure } from '../../src/engine/core/state';
import { readTrainingRecords } from './records/replay';
import { readRecordLines } from './records/io';
import { CORRECTION_FORMAT, readCorrections } from './corrections/records';

/** 记录实际工作树内容，未提交的编码/规则修改同样改变指纹。 */
export function encodingSourceHash() {
  const root = fileURLToPath(new URL('../../', import.meta.url));
  const hash = createHash('sha256');
  for (const folder of [
    'src/engine',
    'src/ai',
    'src/match',
    'scripts/training/records',
    'scripts/training/corrections',
  ]) {
    const path = resolve(root, folder);
    for (const name of readdirSync(path, { recursive: true })
      .map(String)
      .filter((p) => p.endsWith('.ts'))
      .sort()) {
      const file = resolve(path, name);
      hash.update(relative(root, file).replaceAll('\\', '/'));
      hash.update(readFileSync(file, 'utf8').replaceAll('\r\n', '\n'));
    }
  }
  hash.update(readFileSync(fileURLToPath(import.meta.url), 'utf8').replaceAll('\r\n', '\n'));
  return hash.digest('hex');
}

/**
 * 将可信教师记录转换为版本化网络输入。游戏种子仅留在整局元数据，绝不传入编码器。
 * 逐行处理并等待消费者背压；格式/动作覆盖失败即报错，不跳过样本。
 * 缺少outcome的尾局显式标记interrupted，后续只可使用策略标签。
 */
export async function* encodeTeacherFile(path: string) {
  yield {
    type: 'encoding',
    format: 'haojie-encoded-jsonl-v1',
    schema: ENCODING_SCHEMA,
    source_sha256: encodingSourceHash(),
    ruleset: HAOJIE_RULESET,
  };
  let games = 0;
  let current: number | undefined,
    count = 0;
  let corrections = false;
  for await (const header of readRecordLines(path)) {
    corrections = header.format === CORRECTION_FORMAT;
    break;
  }
  for await (const row of corrections ? readCorrections(path) : readTrainingRecords(path)) {
    if (row.type === 'game') {
      ensure(row.source !== 'neural', '教师编码器不接受网络对战标签。');
      games++;
      current = row.game;
      count = 0;
      const group = `${row.ruleset}:${row.rules}:${row.seed}`;
      yield {
        type: 'game',
        game: current,
        group,
        game_id: row.gameId ?? group,
        teachers: row.teachers,
        primary_player: row.primaryPlayer,
        rules: row.rules,
        seed: row.seed,
        difficulty: row.difficulty,
        budget: row.budget,
        source: row.source,
        origin: row.origin,
      };
    } else if (row.type === 'sample') {
      const tree = new TrainingActionTree(row.observation, row.actor);
      let trace;
      try {
        trace = tree.trace(row.command);
      } catch (error) {
        throw new Error(`game=${current}, command=${count}: ${JSON.stringify(row.command)}`, {
          cause: error,
        });
      }
      for (const [step, { node, selected }] of trace.entries())
        yield {
          type: 'example',
          game: current,
          index: count,
          step,
          actor: row.actor,
          command: row.command.type,
          stage: node.stage,
          selected,
          input: encodeDecision(row.observation, row.actor, node),
        };
      count++;
    } else if (row.type === 'outcome') {
      const wasInterrupted =
        row.interrupted === true ||
        (typeof row.interrupted === 'string' && row.interrupted.length > 0);
      yield {
        type: 'outcome',
        game: current,
        commands: count,
        terminated: row.terminated,
        truncated: row.truncated,
        interrupted: wasInterrupted,
        interruptionReason: typeof row.interrupted === 'string' ? row.interrupted : undefined,
        returns: row.returns,
        winner: row.winner,
      };
      current = undefined;
    } else throw new Error(`未知教师记录类型：${String(row.type)}`);
  }
  ensure(games > 0, '教师文件没有对局。');
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  ensure(process.argv.length === 3, '用法：tsx scripts/training/encode.ts <教师JSONL>');
  for await (const row of encodeTeacherFile(process.argv[2]))
    if (!process.stdout.write(JSON.stringify(row) + '\n')) await once(process.stdout, 'drain');
}
