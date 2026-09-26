import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs, isDeepStrictEqual } from 'node:util';

/** 按根/族汇总独立续弈的选择差；未知保持区间，绝不把分支数当独立对局数。 */
const { values } = parseArgs({
  options: { source: { type: 'string' }, output: { type: 'string' } },
});
assert.ok(values.source && values.output);
const read = (name: string) => JSON.parse(readFileSync(resolve(values.source!, name), 'utf8'));
const dev = read('development/rankings.json');
const candidates = read('development/candidates.json');
const development = read('development/results.json');
const confirm = read('confirm/results.json');
const confirmCandidates = read('confirm/candidates.json');
const select = (r: any) => {
  const rows = development.filter((v: any) => v.index === r.index);
  // 保守改招要求四场景全部可比且每场景都优于基线；否则仍保留基线。
  if (r.paired.length !== 4 || !r.best) return 0;
  return r.paired.every(
    (s: number) =>
      rows.find((v: any) => v.candidate === r.best && v.scene === s).heuristic >
      rows.find((v: any) => v.candidate === 0 && v.scene === s).heuristic,
  )
    ? r.best
    : 0;
};
const pairs: {
  index: number;
  position: unknown;
  meanRanked: number;
  selected: number;
  comparisons: { name: string; lower: number; upper: number; scenes: { complete: boolean }[] }[];
}[] = [];
for (const c of confirmCandidates) {
  assert.deepEqual(c.candidates, candidates[c.index].candidates);
  const ranking = dev[c.index];
  const selected = select(ranking);
  const rows = confirm.filter((r: any) => r.index === c.index);
  const comparisons = Object.entries({
    baseline: 0,
    cold: c.coldIndex,
    easy: c.easyIndex,
    production: c.productionIndex,
  }).map(([name, reference]) => {
    const scenes = [0, 1].map((scene) => {
      const a = rows.find((r: any) => r.candidate === selected && r.scene === scene);
      const b = rows.find((r: any) => r.candidate === reference && r.scene === scene);
      if (!b) return { scene, complete: false, lower: -2, upper: 2, difference: null };
      const same = selected === reference;
      const complete = a.value != null && b.value != null;
      return {
        scene,
        complete,
        same,
        selectedValue: a.value,
        referenceValue: b.value,
        difference: same ? 0 : complete ? a.value - b.value : null,
        lower: same ? 0 : (a.value ?? -1) - (b.value ?? 1),
        upper: same ? 0 : (a.value ?? 1) - (b.value ?? -1),
      };
    });
    return {
      name,
      scenes,
      lower: scenes.reduce((n, s) => n + s.lower, 0) / 2,
      upper: scenes.reduce((n, s) => n + s.upper, 0) / 2,
    };
  });
  pairs.push({
    index: c.index,
    position: c.position,
    meanRanked: ranking.best,
    selected,
    comparisons,
  });
}
const summary = {
  development: {
    roots: dev.length,
    fullyPaired: dev.filter((r: any) => r.paired.length === 4).length,
    anyPaired: dev.filter((r: any) => r.paired.length > 0).length,
    heuristicChanges: dev.filter((r: any) => !r.fallback && r.best !== 0).length,
    conservativeChanges: dev.filter((r: any) => select(r) !== 0).length,
    networkDisagreements: dev.filter(
      (r: any) => !r.fallback && !r.network.fallback && r.best !== r.network.best,
    ).length,
    coldMissingFromFullBeam: candidates.filter(
      (r: any) => !r.beam.candidates.some((c: any) => isDeepStrictEqual(c.command, r.hard.command)),
    ).length,
    byStratum: ['deploy', 'action'].map((stratum) => {
      const rows = dev.filter((r: any) => r.position.stratum === stratum);
      return {
        stratum,
        roots: rows.length,
        anyPaired: rows.filter((r: any) => r.paired.length).length,
        conservativeChanges: rows.filter((r: any) => select(r) !== 0).length,
      };
    }),
  },
  confirm: {
    roots: confirmCandidates.length,
    paths: confirm.length,
    terminal: confirm.filter((r: any) => r.stop === 'terminal').length,
    wins: confirm.filter((r: any) => r.value === 1).length,
    losses: confirm.filter((r: any) => r.value === -1).length,
    draws: confirm.filter((r: any) => r.value === 0).length,
    unknown: confirm.filter((r: any) => r.value == null).length,
    stops: Object.fromEntries(
      [...new Set(confirm.map((r: any) => r.stop))].map((stop) => [
        stop,
        confirm.filter((r: any) => r.stop === stop).length,
      ]),
    ),
    comparisons: ['baseline', 'cold', 'easy', 'production'].map((name) => ({
      name,
      lower:
        pairs.reduce((n, r) => n + r.comparisons.find((c) => c.name === name)!.lower, 0) /
        pairs.length,
      upper:
        pairs.reduce((n, r) => n + r.comparisons.find((c) => c.name === name)!.upper, 0) /
        pairs.length,
      completePairs: pairs.reduce(
        (n, r) =>
          n + r.comparisons.find((c) => c.name === name)!.scenes.filter((s) => s.complete).length,
        0,
      ),
    })),
  },
  pairs,
  note: 'bounded development diagnostic; interval is unknown-outcome sensitivity, not confidence interval or whole-game win rate',
};
writeFileSync(values.output, JSON.stringify(summary, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify({ ...summary, pairs: undefined }));
