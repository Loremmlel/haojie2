import assert from 'node:assert/strict';
import { createReadStream, readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { createHash } from 'node:crypto';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { TrainingEnvironment } from '../../src/match/training';
import { fingerprint, decisionOwner } from '../../src/ai/observation';
import { HAOJIE_RULESET } from '../../src/engine/online/player-view';
import { encodingSourceHash } from './encode';
import { TrainingActionTree } from '../../src/ai/training/action-tree';
import { encodeDecision } from '../../src/ai/training/encoding/decision';

const increment = (counts: Record<string, number>, key: string) => {
  counts[key] = (counts[key] ?? 0) + 1;
};
const profileName = (profile: { difficulty: string; simulations?: number }) =>
  `${profile.difficulty}:${profile.simulations ?? 'production'}`;
const latency = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  return {
    count: sorted.length,
    totalMs: sorted.reduce((a, b) => a + b, 0),
    p50: sorted[Math.ceil(sorted.length * 0.5) - 1] ?? null,
    p95: sorted[Math.ceil(sorted.length * 0.95) - 1] ?? null,
    max: sorted.at(-1) ?? null,
  };
};

/**
 * 流式重放教师记录，只在宿主恢复正式种子；验证每条公开观察、命令后指纹和实际胜负。
 * 统计被观察到与实际作为动作来源的种类，区分缓存/零模拟/真实搜索；不把覆盖率当棋力。
 * 相邻同名JSON报告存在时也验证训练上限，否则仅验证引擎状态、不猜测旧采样上限。
 */
export async function inspectTeacherFiles(paths: string[], encodeInputs = false) {
  const files = [],
    games: any[] = [],
    profiles: Record<string, any> = {},
    identities = new Set();
  for (const path of paths) {
    const reportPath = path.replace(/\.jsonl$/, '.json');
    const producer =
      reportPath !== path && existsSync(reportPath)
        ? JSON.parse(readFileSync(reportPath, 'utf8'))
        : null;
    if (producer) assert.equal(producer.format, 'haojie-teacher-run-v1');
    const source = createReadStream(path),
      digest = createHash('sha256');
    source.on('data', (chunk) => digest.update(chunk));
    const lines = createInterface({ input: source, crlfDelay: Infinity });
    let env: TrainingEnvironment | undefined,
      header: any,
      actorCommands = { 1: 0, 2: 0 },
      actorProfiles: Partial<Record<1 | 2, string>> = {},
      fileCommands = 0;
    for await (const line of lines) {
      if (!line.trim()) continue;
      const row = JSON.parse(line);
      if (row.type === 'game') {
        assert.equal(env, undefined, '上一局缺少结束记录');
        assert.equal(row.ruleset, HAOJIE_RULESET);
        const identity = row.gameId ?? `${row.ruleset}:${row.rules}:${row.seed}`;
        assert.ok(!identities.has(identity), '重复教师轨迹');
        identities.add(identity);
        header = row;
        actorCommands = { 1: 0, 2: 0 };
        actorProfiles = {};
        env = new TrainingEnvironment({
          maxCommands: 1e9,
          maxPlies: 1e9,
          ...producer?.options,
          seed: row.seed,
          rules: row.rules,
        });
      } else if (row.type === 'sample') {
        assert.ok(env, '样本缺少对局头');
        assert.equal(row.game, header.game);
        assert.equal(row.index, env.status().commands);
        const observation = env.observation();
        assert.deepEqual(
          row.observation,
          observation,
          `公开局面漂移：${path}/${row.game}/${row.index}`,
        );
        const owner = decisionOwner(observation);
        assert.equal(row.actor, owner);
        const name = profileName(
          header.teachers?.[owner] ?? {
            difficulty: header.difficulty,
            simulations: typeof header.budget === 'number' ? header.budget : undefined,
          },
        );
        const entry = (profiles[name] ??= {
          commands: 0,
          cached: 0,
          zeroSimulations: 0,
          simulations: 0,
          replyDecisions: 0,
          exhausted: 0,
          reactions: 0,
          paths: 0,
          commandKinds: {},
          phases: {},
          observedKinds: {},
          sourceKinds: {},
          skillKinds: {},
          maxUnits: 0,
          durations: [],
          searchDurations: [],
          outcomeCommands: { win: 0, loss: 0, draw: 0, unknown: 0 },
        });
        entry.commands++;
        actorCommands[owner]++;
        actorProfiles[owner] = name;
        increment(entry.commandKinds, row.command.type);
        increment(entry.phases, observation.phase);
        entry.cached += Number(!!row.teacherStats.cached);
        entry.zeroSimulations += Number(row.teacherStats.simulations === 0);
        entry.simulations += row.teacherStats.simulations;
        entry.replyDecisions += Number((row.teacherStats.replyCandidates ?? 0) > 0);
        entry.exhausted += Number(row.teacherStats.exhausted);
        entry.reactions += Number(observation.pending.length > 0);
        entry.paths += Number(!!row.command.path);
        entry.maxUnits = Math.max(entry.maxUnits, observation.units.length);
        for (const kind of new Set(
          [...observation.units, ...observation.hands[1], ...observation.hands[2]].map((v) =>
            String(v.kind),
          ),
        ))
          increment(entry.observedKinds, kind);
        const actor = [...observation.units, ...observation.hands[owner]].find(
          (v) => v.id === (row.command.unitId ?? row.command.cardId),
        );
        if (actor) {
          increment(entry.sourceKinds, String(actor.kind));
          if (row.command.type === 'skill') increment(entry.skillKinds, String(actor.kind));
        }
        if (row.timing) {
          entry.durations.push(row.timing.teacherMs);
          if (!row.teacherStats.cached && row.teacherStats.simulations > 0)
            entry.searchDurations.push(row.timing.teacherMs);
        }
        if (encodeInputs) {
          const summary = (entry.encoding ??= {
            commands: 0,
            examples: 0,
            forcedExamples: 0,
            multiChoiceCommands: 0,
            multiChoiceRoots: 0,
            maxEntities: 0,
            maxCandidates: 0,
            totalEntities: 0,
            totalCandidates: 0,
          });
          const trace = new TrainingActionTree(observation, owner).trace(row.command);
          summary.commands++;
          summary.multiChoiceCommands += Number(trace.some((s) => s.node.choices.length > 1));
          summary.multiChoiceRoots += Number(trace[0].node.choices.length > 1);
          for (const { node } of trace) {
            const input = encodeDecision(observation, owner, node);
            summary.examples++;
            summary.forcedExamples += Number(node.choices.length === 1);
            summary.maxEntities = Math.max(summary.maxEntities, input.entities.length);
            summary.maxCandidates = Math.max(summary.maxCandidates, input.candidates.length);
            summary.totalEntities += input.entities.length;
            summary.totalCandidates += input.candidates.length;
          }
        }
        env.step(owner, row.command);
        fileCommands++;
        if (row.after) assert.equal(row.after, fingerprint(env.observation()));
      } else if (row.type === 'outcome') {
        assert.ok(env, '结束记录缺少对局');
        assert.equal(row.game, header.game);
        const status = env.status();
        for (const key of ['commands', 'ply', 'phase', 'terminated', 'winner', 'returns'] as const)
          assert.deepEqual(row[key], status[key], `终局状态漂移：${path}/${row.game}/${key}`);
        if (producer)
          for (const key of ['truncated', 'truncation', 'toPlay'] as const)
            assert.deepEqual(row[key], status[key]);
        assert.equal(Number(row.terminated) + Number(row.truncated) + Number(!!row.interrupted), 1);
        for (const owner of [1, 2] as const) {
          const name = actorProfiles[owner];
          if (!name) continue;
          const outcome = !row.terminated
            ? 'unknown'
            : row.winner === 'draw'
              ? 'draw'
              : row.winner === owner
                ? 'win'
                : 'loss';
          profiles[name].outcomeCommands[outcome] += actorCommands[owner];
        }
        const { observation: _, ...outcome } = row;
        games.push({
          file: path,
          ...outcome,
          rules: header.rules,
          seed: header.seed,
          primaryPlayer: header.primaryPlayer,
          teachers: header.teachers,
          finalFingerprint: fingerprint(env.observation()),
        });
        env = undefined;
      } else throw new Error(`未知教师记录：${row.type}`);
    }
    assert.equal(env, undefined, '文件尾部缺少显式结束记录');
    files.push({ path, sha256: digest.digest('hex'), commands: fileCommands, producer });
  }
  const pairs = new Map<string, any[]>();
  for (const game of games) {
    if (!game.teachers || ![1, 2].includes(game.primaryPlayer)) continue;
    const primary = profileName(game.teachers[game.primaryPlayer]);
    const secondary = profileName(game.teachers[3 - game.primaryPlayer]);
    const key = `${game.ruleset}:${game.rules}:${game.seed}:${primary}:${secondary}`;
    const group = pairs.get(key) ?? [];
    group.push(game);
    pairs.set(key, group);
  }
  const pairScores = [...pairs.entries()].map(([group, pair]) => {
    const complete =
      pair.length === 2 &&
      new Set(pair.map((g) => g.primaryPlayer)).size === 2 &&
      pair.every((g) => g.terminated);
    return {
      group,
      seed: pair[0].seed,
      complete,
      primaryScore: complete
        ? pair.reduce(
            (sum, g) => sum + (g.winner === 'draw' ? 0.5 : Number(g.winner === g.primaryPlayer)),
            0,
          ) / 2
        : null,
    };
  });
  return {
    format: 'haojie-teacher-inspection-v1',
    encodingChecked: encodeInputs,
    ruleset: HAOJIE_RULESET,
    sourceSha256: createHash('sha256')
      .update(encodingSourceHash())
      .update(readFileSync(fileURLToPath(import.meta.url), 'utf8').replace(/\r\n/g, '\n'))
      .digest('hex'),
    files,
    games,
    counts: {
      games: games.length,
      commands: files.reduce((sum, f) => sum + f.commands, 0),
      terminated: games.filter((g) => g.terminated).length,
      truncated: games.filter((g) => g.truncated).length,
      interrupted: games.filter((g) => g.interrupted).length,
      primaryWins: games.filter((g) => g.terminated && g.winner === g.primaryPlayer).length,
      secondaryWins: games.filter(
        (g) =>
          g.terminated &&
          [1, 2].includes(g.primaryPlayer) &&
          g.winner !== 'draw' &&
          g.winner !== g.primaryPlayer,
      ).length,
      draws: games.filter((g) => g.terminated && g.winner === 'draw').length,
    },
    profiles: Object.fromEntries(
      Object.entries(profiles).map(([name, { durations, searchDurations, ...entry }]) => [
        name,
        {
          ...entry,
          teacherLatency: latency(durations),
          searchLatency: latency(searchDurations),
        },
      ]),
    ),
    pairScores,
  };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: { output: { type: 'string' }, encode: { type: 'boolean' } },
  });
  assert.ok(values.output && positionals.length, '请提供教师JSONL路径和新的--output文件');
  assert.ok(!existsSync(values.output), '报告已存在，请保留旧实验。');
  mkdirSync(dirname(values.output), { recursive: true });
  const report = await inspectTeacherFiles(positionals, values.encode);
  writeFileSync(values.output, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify(report.counts, null, 2));
}
