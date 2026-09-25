import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { isDeepStrictEqual, parseArgs } from 'node:util';
import { setImmediate } from 'node:timers/promises';
import { readTrainingRecords } from '../records/replay';
import { hashRecordFile, withRecordOutput } from '../records/io';
import { encodingSourceHash } from '../encode';
import { CORRECTION_FORMAT } from './records';
import { decide } from '../../../src/ai/planning/search';
import { fingerprint } from '../../../src/ai/observation';
import { inspectTrainingCommand } from '../../../src/ai/training/queries';
import { sampleTrainingTransition } from '../../../src/ai/training/simulation';
import { TrainingActionTree } from '../../../src/ai/training/action-tree';
import { HAOJIE_RULESET } from '../../../src/engine/online/player-view';
import { RULESET_ID } from '../../../src/engine/catalog';

/**
 * 每局按公开指纹哈希取固定上限；同回合/阶段只取首个需要推理的学生决策。
 * 选择在查询教师前完成，不因教师分歧筛选；只存引用，不落盘Observation或原胜负。
 * 查询是独立冷计划，不继承学生/对手缓存；取消后不发标签，也不写完成尾行。
 */
export async function relabel(
  input: string,
  output: string,
  perGame = 32,
  nodes = 800,
  signal?: AbortSignal,
) {
  assert.ok(Number.isSafeInteger(perGame) && perGame > 0);
  assert.ok(Number.isSafeInteger(nodes) && nodes >= 40);
  const sourceHash = await hashRecordFile(input);
  const byGame = new Map<number, any[]>();
  let seen = new Set<string>();
  for await (const row of readTrainingRecords(input)) {
    if (signal?.aborted) throw new Error('纠错已取消');
    if (row.type === 'game') {
      assert.equal(row.source, 'neural');
      assert.equal(row.rules, 'classic', '首轮纠错只处理公开信息经典模式');
      seen = new Set();
      byGame.set(row.game, []);
    }
    if (row.type !== 'decision' || row.policy !== 'network' || !row.decoded?.stats.evaluations)
      continue;
    const window = `${row.ply}:${row.phase}:${row.actor}:${row.reaction}`;
    if (seen.has(window)) continue;
    seen.add(window);
    const selected = byGame.get(row.game)!;
    selected.push({
      ...row,
      rank: createHash('sha256').update(`${row.game}:${row.index}:${row.before}`).digest('hex'),
    });
    selected.sort((a, b) => a.rank.localeCompare(b.rank));
    if (selected.length > perGame) selected.pop();
  }
  assert.equal(await hashRecordFile(input), sourceHash, '选择期间原轨迹改变');
  let count = 0,
    disagreements = 0;
  await withRecordOutput(output, async (emit) => {
    await emit({
      type: 'corrections',
      format: CORRECTION_FORMAT,
      ruleset: HAOJIE_RULESET,
      recordRuleset: RULESET_ID,
      source: { path: relative(dirname(resolve(output)), resolve(input)), sha256: sourceHash },
      sourceSha256: encodingSourceHash(),
      teacher: { difficulty: 'hard', simulations: nodes },
      perGame,
    });
    for (const rows of byGame.values())
      for (const row of rows.sort((a, b) => a.index - b.index)) {
        await setImmediate();
        if (signal?.aborted) throw new Error('纠错已取消，文件未完成');
        const before = fingerprint(row.observation);
        const decision = decide(row.observation, row.actor, 'hard', {
          simulations: nodes,
          mode: 'work',
          trace: false,
        });
        assert.ok(decision.command);
        const checked = inspectTrainingCommand(row.observation, row.actor, decision.command);
        assert.notEqual(checked.status, 'invalid');
        new TrainingActionTree(row.observation, row.actor).trace(decision.command);
        // 固定独立模拟源只检查一次合法转移，不据它生成价值标签或声称穷尽随机结果。
        sampleTrainingTransition(row.observation, row.actor, decision.command, 0);
        assert.equal(fingerprint(row.observation), before);
        await setImmediate();
        if (signal?.aborted) throw new Error('纠错已取消，文件未完成');
        const changed = !isDeepStrictEqual(row.command, decision.command);
        disagreements += Number(changed);
        await emit({
          type: 'label',
          game: row.game,
          index: row.index,
          actor: row.actor,
          before,
          command: decision.command,
          stats: decision.stats,
          publicStatus: checked.status,
          changed,
        });
        count++;
        if (count % 16 === 0) console.error(JSON.stringify({ labels: count, disagreements }));
      }
    assert.ok(count > 0, '没有可纠错位置');
    assert.equal(await hashRecordFile(input), sourceHash, '查询期间原轨迹改变');
    await emit({ type: 'complete', labels: count, disagreements });
  });
  return { labels: count, disagreements };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { values } = parseArgs({
    options: Object.fromEntries(
      ['input', 'output', 'per-game', 'nodes'].map((k) => [k, { type: 'string' }]),
    ),
  });
  assert.ok(values.input && values.output);
  const controller = new AbortController();
  process.once('SIGINT', () => controller.abort());
  console.log(
    await relabel(
      values.input,
      values.output,
      Number(values['per-game'] ?? 32),
      Number(values.nodes ?? 800),
      controller.signal,
    ),
  );
}
