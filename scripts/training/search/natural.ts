import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync, appendFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { encodingSourceHash } from '../encode';
import { hashRecordFile } from '../records/io';
import { readTrainingRecords } from '../records/replay';
import { search, commands } from './puct';
import { reference } from './reference';
import { decide } from '../../../src/ai/planning/search';
import { TrainingActionTree } from '../../../src/ai/training/action-tree';
import { inspectTrainingCommand } from '../../../src/ai/training/queries';
import type { Observation } from '../../../src/ai/types';
import type { Player } from '../../../src/engine/types';

interface Selection {
  game: number;
  seed: number;
  index: number;
  actor: Player;
  before: string;
  policy: string;
  rank: string;
  observation: Observation;
}

/**
 * 从已完整重放且与训练/验证族隔离的自然开发对局中，先选样后查询搜索/教师。
 * 只保存轨迹引用和公开指纹；所有失败位置保留，不按可解性或模型表现换题。
 * 无法完成精确参照时明确未决，不把采样估计升级为真值或训练标签。
 */
export async function run(input: string, manifest: string, output: string) {
  mkdirSync(output, { recursive: true });
  assert.ok(!existsSync(resolve(output, 'protocol.json')), '拒绝重复运行或覆盖');
  const digest = (value: Buffer | string) => createHash('sha256').update(value).digest('hex');
  const write = (name: string, value: unknown) =>
    writeFileSync(resolve(output, name), JSON.stringify(value, null, 2), { flag: 'wx' });
  const source = encodingSourceHash();
  const inputHash = await hashRecordFile(input);
  const training = JSON.parse(readFileSync(manifest, 'utf8'));
  const usedGroups = new Set<string>([
    ...training.splits.train.groups,
    ...training.splits.validation.groups,
  ]);
  const scriptFiles = ['puct.ts', 'reference.ts', 'natural.ts'].map(
    (f) => `scripts/training/search/${f}`,
  );
  const scriptHashes = Object.fromEntries(scriptFiles.map((f) => [f, digest(readFileSync(f))]));
  const protocol = {
    format: 'haojie-natural-search-probe-v1',
    input,
    inputSha256: inputHash,
    manifest,
    manifestSha256: digest(readFileSync(manifest)),
    sourceSha256: source,
    scriptHashes,
    selection:
      'first decision per ply/actor/reaction window, play or pending, at least two units; two lowest salted hashes per game',
    selectionSalt: 'natural-search-20260925-v1',
    perGame: 2,
    budgets: [0, 32, 64, 128],
    searchSeeds: [2026092731, 2026092732],
    horizon: 2,
    maxActionNodes: 512,
    referenceMaxActions: 512,
    teacher: { difficulty: 'hard', simulations: 800 },
    prior: 'uniform complete-command',
    leaf: 'terminal result else unknown zero estimate',
    note: '复用未参与训练的自然开发种子，不是新盲测；没有训练标签或网络推理',
  };
  write('protocol.json', protocol);
  const selected = new Map<number, Selection[]>();
  const outcomes: any[] = [];
  const eligible = new Map<number, number>();
  const headers: any[] = [];
  let header: any;
  let windows = new Set<string>();
  for await (const row of readTrainingRecords(input)) {
    if (row.type === 'game') {
      assert.equal(row.source, 'neural');
      assert.equal(row.rules, 'classic');
      const group = `${row.ruleset}:${row.rules}:${row.seed}`;
      assert.ok(!usedGroups.has(group), '自然评估族与训练或验证重合');
      header = row;
      headers.push({ game: row.game, seed: row.seed, group, limits: row.limits });
      selected.set(row.game, []);
      eligible.set(row.game, 0);
      windows = new Set();
    } else if (row.type === 'outcome') {
      outcomes.push({
        game: row.game,
        terminated: row.terminated,
        truncated: row.truncated,
        interrupted: row.interrupted,
        winner: row.winner,
        commands: row.commands,
      });
    } else if (row.type === 'decision') {
      const o = row.observation as Observation;
      if ((o.phase !== 'play' && !o.pending.length) || o.units.length < 2) continue;
      const window = `${o.ply}:${row.actor}:${o.pending.length > 0}`;
      if (windows.has(window)) continue;
      windows.add(window);
      eligible.set(row.game, eligible.get(row.game)! + 1);
      const rank = digest(`${protocol.selectionSalt}:${row.game}:${row.index}:${row.before}`);
      const list = selected.get(row.game)!;
      list.push({
        game: row.game,
        seed: header.seed,
        index: row.index,
        actor: row.actor,
        before: row.before,
        policy: row.policy,
        rank,
        observation: o,
      });
      list.sort((a, b) => a.rank.localeCompare(b.rank));
      list.splice(protocol.perGame);
    }
  }
  assert.equal(await hashRecordFile(input), inputHash);
  assert.equal(headers.length, 10);
  assert.equal(outcomes.length, headers.length);
  assert.ok(outcomes.every((o) => !o.interrupted));
  const selections = [...selected.values()].flat();
  assert.equal(selections.length, 20);
  write('selection.json', {
    headers,
    outcomes,
    eligible: Object.fromEntries(eligible),
    positions: selections.map(({ observation, ...r }) => r),
  });
  const references: any[] = [];
  const rows: any[] = [];
  const errors = (error: unknown) => (error instanceof Error ? error.message : String(error));
  for (const p of selections) {
    const before = JSON.stringify(p.observation);
    const identity = { game: p.game, index: p.index, actor: p.actor, before: p.before };
    let oracle: ReturnType<typeof reference> | undefined;
    const start = performance.now();
    try {
      oracle = reference(p.observation, protocol.horizon, {
        maxActions: protocol.referenceMaxActions,
        maxActionNodes: protocol.maxActionNodes,
      });
      references.push({
        ...identity,
        status: 'exact',
        elapsedMs: performance.now() - start,
        ...oracle,
        informative: oracle.rootValues.some((r) => Math.abs(r.value - oracle!.best) > 1e-9),
      });
    } catch (error) {
      references.push({
        ...identity,
        status: 'unresolved',
        reason: errors(error),
        elapsedMs: performance.now() - start,
      });
    }
    let rootCommands: ReturnType<typeof commands> | undefined;
    try {
      rootCommands = commands(p.observation, protocol.maxActionNodes);
    } catch {
      /* 搜索仍独立报告失败。 */
    }
    const score = (command: any) => {
      assert.notEqual(inspectTrainingCommand(p.observation, p.actor, command).status, 'invalid');
      if (!oracle) return { reference: 'unresolved' };
      const canonical = new TrainingActionTree(p.observation, p.actor).trace(command).at(-1)!;
      const c = canonical.node.choices[canonical.selected].command;
      const value = oracle.rootValues.find(
        (r) => JSON.stringify(r.command) === JSON.stringify(c),
      )?.value;
      assert.notEqual(value, undefined, '选招必须属于完整参照域');
      return {
        reference: 'exact',
        value,
        best: oracle.best,
        regret: oracle.best - value!,
        optimal: Math.abs(oracle.best - value!) < 1e-9,
      };
    };
    const teacherStart = performance.now();
    const teacher = decide(p.observation, p.actor, 'hard', { simulations: 800, mode: 'work' });
    assert.ok(teacher.command);
    rows.push({
      ...identity,
      method: 'hard-800',
      elapsedMs: performance.now() - teacherStart,
      command: teacher.command,
      stats: teacher.stats,
      ...score(teacher.command),
    });
    for (const seed of protocol.searchSeeds)
      for (const budget of protocol.budgets) {
        const began = performance.now();
        const result = search(p.observation, {
          simulations: budget,
          horizon: protocol.horizon,
          sampleSeed: seed,
          maxActionNodes: protocol.maxActionNodes,
        });
        const row = {
          ...identity,
          method: 'uniform-puct',
          seed,
          budget,
          elapsedMs: performance.now() - began,
          rootCommands: rootCommands?.commands.length ?? null,
          ...result,
          ...(result.status === 'command'
            ? score(result.command)
            : { reference: oracle ? 'exact' : 'unresolved' }),
        };
        rows.push(row);
        appendFileSync(resolve(output, 'live.jsonl'), JSON.stringify(row) + '\n');
      }
    assert.equal(JSON.stringify(p.observation), before);
    console.log(
      JSON.stringify({
        completed: identity,
        reference: references.at(-1)!.status,
        reason: references.at(-1)!.reason,
        commands: rootCommands?.commands.length ?? null,
      }),
    );
  }
  assert.equal(await hashRecordFile(input), inputHash);
  assert.equal(encodingSourceHash(), source);
  for (const f of scriptFiles) assert.equal(digest(readFileSync(f)), scriptHashes[f]);
  const groups = protocol.budgets.map((budget) => {
    const group = rows.filter((r) => r.method === 'uniform-puct' && r.budget === budget);
    const exact = group.filter((r) => r.status === 'command' && r.reference === 'exact');
    return {
      budget,
      requested: group.length,
      completed: group.filter((r) => r.status === 'command').length,
      exactScored: exact.length,
      optimal: exact.filter((r) => r.optimal).length,
      paused: group
        .filter((r) => r.status === 'paused')
        .map((r) => ({ game: r.game, index: r.index, seed: r.seed, reason: r.reason })),
      transitions: group.reduce((n, r) => n + r.stats.transitions, 0),
      terminalLeaves: group.reduce((n, r) => n + r.stats.terminalLeaves, 0),
      elapsedMs: group.reduce((n, r) => n + r.elapsedMs, 0),
    };
  });
  write('references.json', references);
  write('decisions.json', rows);
  write('summary.json', {
    complete: true,
    positions: selections.length,
    exactPositions: references.filter((r) => r.status === 'exact').length,
    informativePositions: references.filter((r) => r.informative).length,
    groups,
    noTrainingLabels: true,
    sourceHashesMatch: true,
  });
  console.log(JSON.stringify(groups));
}

const { values } = parseArgs({
  options: { input: { type: 'string' }, manifest: { type: 'string' }, output: { type: 'string' } },
});
assert.ok(values.input && values.manifest && values.output);
await run(values.input, values.manifest, values.output);
