import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { isDeepStrictEqual } from 'node:util';
import { readTrainingRecords } from '../../records/replay';
import { hashRecordFile } from '../../records/io';

/**
 * 重新由引擎验证实战，再比较搜索选中Q与该局真实后续收益。
 * Q含搜索选择与机会抽样，不等同于裸价值头预测；仅作部署分布诊断，不用于训练或选检查点。
 * 截断/暂停没有目标，按局另列；只保存引用与标量，不保存Observation或反事实胜负。
 */
const [output, ...reports] = process.argv.slice(2);
assert.ok(output && reports.length);
const rows: any[] = [],
  inputs = [];
const mean = (values: number[]) =>
  values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
for (const path of reports) {
  const bytes = readFileSync(path),
    report = JSON.parse(bytes.toString('utf8'));
  assert.equal(report.format, 'haojie-neural-match-v1');
  const trace = join(dirname(path), 'games.jsonl.gz');
  inputs.push({
    report: path,
    reportSha256: createHash('sha256').update(bytes).digest('hex'),
    traceSha256: await hashRecordFile(trace),
  });
  let header: any,
    chosen: number[] = [],
    sameActorTransitions = 0,
    changedActorTransitions = 0,
    terminalLeaves = 0,
    valueLeaves = 0,
    games = 0;
  for await (const row of readTrainingRecords(trace)) {
    if (row.type === 'game') {
      header = row;
      chosen = [];
      sameActorTransitions = 0;
      changedActorTransitions = 0;
      terminalLeaves = 0;
      valueLeaves = 0;
    } else if (row.type === 'decision' && row.policy === 'network') {
      const search = row.decoded?.search;
      if (!search?.edges || search.fallback) continue;
      assert.equal(row.actor, header.networkPlayer);
      const selected = search.edges.find((e: any) => isDeepStrictEqual(e.command, row.command));
      assert.ok(selected && Number.isFinite(selected.value) && Math.abs(selected.value) <= 1);
      chosen.push(selected.value);
      // 只统计实际采用搜索结果的决策，半途退回束的工作量仍由实战报告另计。
      sameActorTransitions += search.stats.sameActor;
      changedActorTransitions += search.stats.changedActor;
      terminalLeaves += search.stats.terminalLeaves;
      valueLeaves += search.stats.valueCalls;
    } else if (row.type === 'outcome') {
      games++;
      const recorded = report.outcomes.find((r: any) => r.game === row.game);
      assert.ok(recorded);
      for (const key of ['winner', 'terminated', 'truncated', 'interrupted', 'after'])
        assert.deepEqual(recorded[key], row[key]);
      const target = !row.terminated
        ? null
        : row.winner === 'draw'
          ? 0
          : row.winner === header.networkPlayer
            ? 1
            : -1;
      rows.push({
        report: path,
        checkpointSha256: report.model.checkpoint_sha256,
        decoder: report.decoder,
        seed: header.seed,
        networkPlayer: header.networkPlayer,
        terminated: row.terminated,
        interrupted: row.interrupted,
        truncated: row.truncated,
        target,
        searchedDecisions: chosen.length,
        sameActorTransitions,
        changedActorTransitions,
        terminalLeaves,
        valueLeaves,
        chosenQMean: mean(chosen),
        positiveFraction: mean(chosen.map((q) => Number(q > 0))),
        chosenQMse: target === null ? null : mean(chosen.map((q) => (q - target) ** 2)),
      });
    }
  }
  assert.equal(games, report.games);
}
writeFileSync(
  output,
  JSON.stringify(
    { interpretation: 'chosen search Q versus actual continuation; not raw V', inputs, rows },
    null,
    2,
  ) + '\n',
  { flag: 'wx' },
);
console.log(JSON.stringify({ output, games: rows.length, replayValidated: true }));
