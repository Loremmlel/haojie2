import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';

function distribution(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  return {
    count: sorted.length,
    mean: sorted.length ? sorted.reduce((a, b) => a + b, 0) / sorted.length : null,
    p50: sorted.length ? sorted[Math.ceil(sorted.length * 0.5) - 1] : null,
    p95: sorted.length ? sorted[Math.ceil(sorted.length * 0.95) - 1] : null,
    max: sorted.at(-1) ?? null,
  };
}

/** 从落盘trace汇总，暂停/截断不计胜负；同时列出强制步骤及需要推理的完整命令。 */
export async function summarizeMatches(path: string) {
  const outcomes: any[] = [],
    anomalies: any[] = [],
    decisions: any[] = [];
  let games = 0,
    rejectedCommands = 0;
  const lines = createInterface({ input: createReadStream(path), crlfDelay: Infinity });
  for await (const line of lines) {
    if (!line.trim()) continue;
    const row = JSON.parse(line);
    if (row.type === 'game') games++;
    if (row.type === 'decision') decisions.push(row);
    if (row.type === 'outcome') outcomes.push(row);
    if (['pause', 'error', 'rejected'].includes(row.type)) {
      // 完整观察留在trace，摘要只记可定位的原因，避免报告随异常局面体积膨胀。
      const { observation: _, ...small } = row;
      anomalies.push(small);
      if (row.type === 'rejected') rejectedCommands++;
    }
  }
  const network = decisions.filter((row) => row.policy === 'network');
  const evaluated = network.filter((row) => row.decoded.stats.evaluations > 0);
  const summarize = (rows: any[]) => ({
    decisions: rows.length,
    totalMs: distribution(rows.map((r) => r.timing.totalMs)),
    decisionMs: distribution(rows.map((r) => r.timing.decisionMs)),
    inferenceCalls: distribution(rows.map((r) => r.decoded?.stats.evaluations ?? 0)),
    treeMs: distribution(rows.map((r) => r.decoded?.stats.treeMs ?? 0)),
    encodingMs: distribution(rows.map((r) => r.decoded?.stats.encodingMs ?? 0)),
    inferenceMs: distribution(rows.map((r) => r.decoded?.stats.inferenceMs ?? 0)),
  });
  const commands: Record<string, number> = {};
  for (const row of network) commands[row.command.type] = (commands[row.command.type] ?? 0) + 1;
  const turns = new Map<string, number>();
  for (const row of network) {
    const key = `${row.game}:${row.ply}:${row.actor}`;
    turns.set(key, (turns.get(key) ?? 0) + row.timing.totalMs);
  }
  return {
    games,
    terminatedGames: outcomes.filter((r) => r.terminated).length,
    truncatedGames: outcomes.filter((r) => r.truncated).length,
    interruptedGames: outcomes.filter((r) => r.interrupted).length + games - outcomes.length,
    rejectedCommands,
    networkWins: outcomes.filter((r) => r.terminated && r.winner === r.networkPlayer).length,
    teacherWins: outcomes.filter(
      (r) => r.terminated && r.winner !== 'draw' && r.winner !== r.networkPlayer,
    ).length,
    draws: outcomes.filter((r) => r.terminated && r.winner === 'draw').length,
    network: summarize(network),
    evaluatedNetwork: summarize(evaluated),
    networkTimePerGlobalPlyMs: distribution([...turns.values()]),
    teacher: summarize(decisions.filter((r) => r.policy === 'teacher')),
    firstEvaluatedDecision: evaluated[0] ?? null,
    networkCommands: commands,
    networkReactions: network.filter((r) => r.reaction).length,
    maxEntities: Math.max(0, ...network.map((r) => r.decoded.stats.maxEntities)),
    maxCandidates: Math.max(0, ...network.map((r) => r.decoded.stats.maxCandidates)),
    backtracks: network.reduce((sum, r) => sum + r.decoded.stats.backtracks, 0),
    anomalies,
    outcomes,
  };
}
