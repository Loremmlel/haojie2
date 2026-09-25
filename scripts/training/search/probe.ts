import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { positions } from './positions';
import { reference } from './reference';
import { search } from './puct';
import { encodingSourceHash } from '../encode';
import { RULESET_ID } from '../../../src/engine/catalog';
import { HAOJIE_RULESET } from '../../../src/engine/online/player-view';

/** 独立开发探针，不采集训练标签；协议先落盘，预算/种子固定，不按中途结果挑局面。 */
export function main(output: string) {
  mkdirSync(output, { recursive: true });
  assert.ok(!existsSync(resolve(output, 'protocol.json')), '拒绝覆盖已有探针');
  const write = (name: string, data: unknown) =>
    writeFileSync(resolve(output, name), JSON.stringify(data, null, 2), { flag: 'wx' });
  const hash = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');
  const files = ['puct.ts', 'positions.ts', 'reference.ts', 'probe.ts'].map(
    (f) => `scripts/training/search/${f}`,
  );
  const hashes = Object.fromEntries(files.map((f) => [f, hash(readFileSync(f))]));
  const source = encodingSourceHash();
  const cases = positions();
  const protocol = {
    format: 'haojie-search-probe-v1',
    ruleset: HAOJIE_RULESET,
    recordRuleset: RULESET_ID,
    sourceSha256: source,
    scriptHashes: hashes,
    budgets: [0, 32, 64, 128],
    seeds: [2026092721, 2026092722, 2026092723, 2026092724],
    horizon: 2,
    cpuct: 1.5,
    prior: 'uniform over complete commands',
    leaf: 'terminal result or zero estimate at unknown boundary',
    boundary: 'two complete commands or exit play/reaction window, whichever is first',
    node: process.version,
    cases: cases.map(({ observation, ...p }) => ({
      ...p,
      observationSha256: hash(JSON.stringify(observation)),
    })),
    limitations: [
      '人工开发夹具，不是自然对战或独立种子族验证',
      '未连接网络，没有训练或强度结论',
      '仅当前play/反应窗口，不支持暗选和回合外巨大化',
      '0是未决叶端估计，绝不输出胜负训练标签',
    ],
  };
  write('protocol.json', protocol);
  const rows = [];
  const references = [];
  for (const p of cases) {
    const before = JSON.stringify(p.observation);
    const started = performance.now();
    const oracle = reference(p.observation, protocol.horizon);
    const referenceMs = performance.now() - started;
    const expected =
      p.family === 'opponent-death-reaction' ? 0 : p.family === 'chance-win' ? 1 / 3 : 1;
    assert.ok(Math.abs(oracle.best - expected) < 1e-9, `夹具没有预期的规则结果：${p.name}`);
    references.push({ name: p.name, referenceMs, ...oracle });
    for (const seed of protocol.seeds)
      for (const budget of protocol.budgets) {
        const start = performance.now();
        const result = search(p.observation, {
          simulations: budget,
          horizon: protocol.horizon,
          sampleSeed: seed,
        });
        const elapsedMs = performance.now() - start;
        assert.equal(JSON.stringify(p.observation), before);
        if (result.status !== 'command') throw new Error(`${p.name}: ${result.reason}`);
        assert.equal(result.stats.simulations, budget);
        assert.equal(
          result.edges.reduce((n, e) => n + e.visits, 0),
          budget,
        );
        const value = oracle.rootValues.find(
          (r) => JSON.stringify(r.command) === JSON.stringify(result.command),
        )?.value;
        assert.notEqual(value, undefined);
        rows.push({
          name: p.name,
          family: p.family,
          actor: p.actor,
          seed,
          budget,
          elapsedMs,
          referenceValue: value!,
          referenceBest: oracle.best,
          regret: oracle.best - value!,
          optimal: Math.abs(oracle.best - value!) < 1e-9,
          ...result,
        });
      }
    console.log(
      JSON.stringify({
        completed: p.name,
        best: oracle.best,
        actions: oracle.rootValues.length,
        referenceMs,
      }),
    );
  }
  assert.equal(encodingSourceHash(), source);
  for (const file of files) assert.equal(hash(readFileSync(file)), hashes[file]);
  const groups = [];
  for (const family of ['all', ...new Set(cases.map((p) => p.family))])
    for (const budget of protocol.budgets) {
      const selected = rows.filter(
        (r) => r.budget === budget && (family === 'all' || r.family === family),
      );
      groups.push({
        family,
        budget,
        decisions: selected.length,
        optimal: selected.filter((r) => r.optimal).length,
        meanRegret: selected.reduce((n, r) => n + r.regret, 0) / selected.length,
        transitions: selected.reduce((n, r) => n + r.stats.transitions, 0),
        actionNodes: selected.reduce((n, r) => n + r.stats.actionNodes, 0),
        elapsedMs: selected.reduce((n, r) => n + r.elapsedMs, 0),
        networkCalls: 0,
      });
    }
  write('reference.json', references);
  write('decisions.json', rows);
  write('summary.json', {
    passed: true,
    cases: cases.length,
    groups,
    sourceUnchanged: true,
    protocolSha256: hash(readFileSync(resolve(output, 'protocol.json'))),
  });
  console.log(JSON.stringify(groups.filter((g) => g.family === 'all')));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { values } = parseArgs({ options: { output: { type: 'string' } } });
  assert.ok(values.output, '指定新的--output目录');
  main(values.output);
}
