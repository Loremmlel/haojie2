import assert from 'node:assert/strict';
import { once } from 'node:events';
import { parseArgs } from 'node:util';
import { readTrainingRecords } from '../../records/replay';
import { hashRecordFile } from '../../records/io';
import { encodingSourceHash } from '../../encode';
import { TrainingActionTree } from '../../../../src/ai/training/action-tree';
import { encodeDecision } from '../../../../src/ai/training/encoding/decision';
import { ENCODING_SCHEMA } from '../../../../src/ai/training/encoding/schema';
import { HAOJIE_RULESET } from '../../../../src/engine/online/player-view';

/**
 * 学生实战的独立价值流：双方实际操作者各取决策根，收益只由重放后的真实终局决定。
 * 记录的动作仅用于验证根选择与张量契约，不能作为教师策略目标；消费者必须禁用策略更新。
 * 不给纠错后的反事实命令沿用胜负，不输出局面快照，编码器不接收真实随机数。
 */
async function* encode(path: string) {
  const sha256 = await hashRecordFile(path);
  yield {
    type: 'encoding',
    format: 'haojie-encoded-jsonl-v1',
    schema: ENCODING_SCHEMA,
    source_sha256: encodingSourceHash(),
    ruleset: HAOJIE_RULESET,
  };
  let count = 0;
  for await (const row of readTrainingRecords(path)) {
    if (row.type === 'game') {
      assert.equal(row.source, 'neural');
      count = 0;
      yield {
        type: 'game',
        game: row.game,
        game_id: `value:${sha256}:${row.game}`,
        group: `${row.ruleset}:${row.rules}:${row.seed}`,
        source: 'value-only-behavior',
        rules: row.rules,
        seed: row.seed,
      };
    } else if (row.type === 'decision') {
      const root = new TrainingActionTree(row.observation, row.actor).trace(row.command)[0];
      yield {
        type: 'example',
        game: row.game,
        index: count++,
        step: 0,
        actor: row.actor,
        command: row.command.type,
        stage: root.node.stage,
        phase: row.observation.phase,
        selected: root.selected,
        input: encodeDecision(row.observation, row.actor, root.node),
      };
    } else if (row.type === 'outcome') {
      yield { ...row, commands: count };
    } else throw new Error('价值来源包含暂停或失败，不能默默过滤');
  }
  assert.equal(await hashRecordFile(path), sha256);
}

const { values } = parseArgs({ options: { encode: { type: 'string' } } });
assert.ok(values.encode);
for await (const row of encode(values.encode))
  if (!process.stdout.write(JSON.stringify(row) + '\n')) await once(process.stdout, 'drain');
