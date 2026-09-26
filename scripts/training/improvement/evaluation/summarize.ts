import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';

/**
 * 汇总冻结种子、换边实战的报告，拒绝缺局、重复座位、混用模型及预算。
 * 胜负从逐局终态重新计数，暂停/截断始终保留为未知；不据汇总选择模型或修改门槛。
 * 耗时按实际命令数加权，不平均各文件的分位数；并发实战耗时不是独占性能基准。
 * 原始轨迹由比赛报告和value/audit.ts另行重放，此处只读报告并以排他方式写新摘要。
 */
const [output, firstSeedText, seedCountText, ...arguments_] = process.argv.slice(2);
const firstSeed = Number(firstSeedText),
  seedCount = Number(seedCountText);
assert.ok(output && arguments_.length);
assert.ok(Number.isSafeInteger(firstSeed) && Number.isSafeInteger(seedCount) && seedCount > 0);
const expected = Array.from({ length: seedCount }, (_, i) =>
  [1, 2].map((player) => `${firstSeed + i}:${player}`),
).flat();
const groups = new Map<string, any[]>();
const inputs = [];
let common: unknown;
for (const argument of arguments_) {
  const separator = argument.indexOf('=');
  assert.ok(separator > 0);
  const label = argument.slice(0, separator),
    path = argument.slice(separator + 1),
    bytes = readFileSync(path),
    report = JSON.parse(bytes.toString('utf8'));
  assert.equal(report.format, 'haojie-neural-match-v1');
  const { games: _, seed: __, ...options } = report.options;
  const identity = { options, sourceSha256: report.sourceSha256, ruleset: report.model.ruleset };
  if (common === undefined) common = identity;
  else assert.deepEqual(identity, common, `规则或比赛预算不同：${path}`);
  assert.equal(report.games, report.outcomes.length, `缺少终态记录：${path}`);
  assert.equal(report.games, report.options.games, `实际局数不匹配：${path}`);
  const count = (predicate: (row: any) => boolean) => report.outcomes.filter(predicate).length;
  const counts = {
    networkWins: count((r) => r.terminated && r.winner === r.networkPlayer),
    teacherWins: count((r) => r.terminated && r.winner !== 'draw' && r.winner !== r.networkPlayer),
    draws: count((r) => r.terminated && r.winner === 'draw'),
    terminatedGames: count((r) => r.terminated),
    truncatedGames: count((r) => r.truncated),
    interruptedGames: count((r) => !!r.interrupted),
  };
  for (const [key, value] of Object.entries(counts))
    assert.equal(report[key], value, `${path}:${key}`);
  inputs.push({ label, path, sha256: createHash('sha256').update(bytes).digest('hex') });
  const rows = groups.get(label) ?? [];
  rows.push(report);
  groups.set(label, rows);
}
const summaries = [];
for (const [label, reports] of groups) {
  const first = reports[0];
  for (const report of reports) {
    assert.equal(report.model.checkpoint_sha256, first.model.checkpoint_sha256);
    assert.equal(report.decoder, first.decoder);
    assert.equal(report.searchSimulations, first.searchSimulations);
    assert.equal(report.experimentSourceSha256, first.experimentSourceSha256);
    assert.deepEqual(report.decoderSources, first.decoderSources);
  }
  const outcomes = reports.flatMap((r) => r.outcomes);
  assert.deepEqual(
    outcomes.map((r) => `${r.seed}:${r.networkPlayer}`).sort(),
    [...expected].sort(),
    `${label}的种子换边集合不完整或重复`,
  );
  const sum = (key: string) => reports.reduce((total, r) => total + r[key], 0);
  const decisions = reports.reduce((total, r) => total + r.network.decisions, 0);
  const decisionMs = reports.reduce(
    (total, r) => total + r.network.decisions * (r.network.decisionMs.mean ?? 0),
    0,
  );
  const search: Record<string, number> = {},
    fallbacks: Record<string, number> = {};
  for (const report of reports) {
    for (const [key, value] of Object.entries(report.searchSummary ?? {})) {
      if (typeof value === 'number') search[key] = (search[key] ?? 0) + value;
    }
    for (const [key, value] of Object.entries(report.searchSummary?.fallbacks ?? {}))
      fallbacks[key] = (fallbacks[key] ?? 0) + Number(value);
  }
  summaries.push({
    label,
    checkpointSha256: first.model.checkpoint_sha256,
    decoder: first.decoder,
    games: outcomes.length,
    independentSeedFamilies: seedCount,
    networkWins: sum('networkWins'),
    teacherWins: sum('teacherWins'),
    draws: sum('draws'),
    truncated: sum('truncatedGames'),
    interrupted: sum('interruptedGames'),
    rejectedCommands: sum('rejectedCommands'),
    successfulNetworkDecisions: decisions,
    meanDecisionMs: decisions ? decisionMs / decisions : null,
    inferenceCallsIncludingUnsuccessfulDecisions: reports.reduce(
      (total, r) => total + r.inference.calls,
      0,
    ),
    search,
    fallbacks,
    outcomes: outcomes.map((r) => ({
      seed: r.seed,
      networkPlayer: r.networkPlayer,
      winner: r.terminated ? r.winner : null,
      truncated: r.truncated,
      interrupted: r.interrupted,
      ply: r.ply,
      commands: r.commands,
    })),
  });
}
writeFileSync(output, JSON.stringify({ common, inputs, summaries }, null, 2) + '\n', {
  flag: 'wx',
});
console.log(
  JSON.stringify({ output, arms: summaries.length, games: summaries.length * expected.length }),
);
