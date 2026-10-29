import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { nativeClient } from '../engine/validation/client';
import { fixtures } from '../engine/fixtures/movement';
import { combatFixtures } from '../engine/fixtures/combat';
import { completeFixtures } from '../engine/fixtures/phases';
import { preparationFixtures } from '../engine/fixtures/preparation';
import { observe, decisionOwner } from '../../../src/ai/observation';
import { decoderFrontier } from '../../../src/ai/classic/decoder';
import { classicRelations } from '../../../src/ai/classic/relations';
import { imagined, positionKey, fingerprint } from '../../../src/ai/observation';
import { explainEvaluation } from '../../../src/ai/evaluation/evaluate';
import { definition } from '../../../src/engine/catalog';
import { iterateCandidateGroups } from '../../../src/ai/planning/candidates';
import { decide } from '../../../src/ai/planning/search';

const executable =
  process.argv.slice(2).find((argument) => !argument.startsWith('--')) ??
  resolve(
    'artifacts/native-target/release',
    process.platform === 'win32' ? 'haojie-engine.exe' : 'haojie-engine',
  );
const native = await nativeClient(executable);
let checked = 0,
  invalidDomains = 0;
try {
  for (const { name, job } of [
    ...fixtures(),
    ...combatFixtures(),
    ...completeFixtures(),
    ...preparationFixtures(),
  ]) {
    const selected = process.argv.find((a) => a.startsWith('--case='))?.slice(7);
    if (selected && !name.includes(selected)) continue;
    if (job.state.phase.startsWith('shrine-')) continue;
    if (job.state.summonOffer || job.state.shrineDraft) continue;
    if (
      [...job.state.units, ...job.state.pending.map((r) => r.source)].some(
        (u) =>
          definition(u.kind).tier === 'shrine' ||
          u.equipment.some((k) => definition(k).tier === 'shrine'),
      ) ||
      job.state.deaths.some((d) => definition(d.kind).tier === 'shrine')
    )
      continue;
    if (
      job.state.mode ||
      job.state.winner ||
      job.state.landmarks?.length ||
      [...(job.state.auras?.[1] ?? []), ...(job.state.auras?.[2] ?? [])].some(
        (a) => a.kind !== 'laoqian',
      ) ||
      job.state.units.some((u) => u.traits?.length || definition(u.kind).tier === 'shrine') ||
      [1, 2].some((p) =>
        job.state.hands[p as 1 | 2].some((v) => definition(v.kind).tier === 'shrine'),
      )
    )
      continue;
    const observation = observe(job.state, decisionOwner(job.state));
    if (job.state.summonSlots < 0) {
      // 负召唤次数是故意构造的非法状态，两端都应拒绝生成完整命令。
      const expected = { leaf: false, stage: 0, focus: 'source', choices: [] };
      assert.deepEqual(decoderFrontier(observation, []), expected, name);
      assert.deepEqual(
        await native.request({ op: 'classic-decode', observation, path: [] }),
        expected,
        name,
      );
      invalidDomains++;
      continue;
    }
    if (process.argv.includes('--decoder')) {
      for (const last of [false, true]) {
        const path: number[] = [];
        for (let depth = 0; depth < 260; depth++) {
          const left = decoderFrontier(observation, path);
          const right = await native.request({ op: 'classic-decode', observation, path });
          assert.deepEqual(right, left, `${name} decoder ${path}`);
          if (left.leaf) break;
          if (!left.choices?.length) throw new Error(`${name}: empty decoder`);
          path.push(last ? left.choices.length - 1 : 0);
        }
      }
      const left = classicRelations(observation),
        right = await native.request({ op: 'classic-relations', observation });
      assert.equal(right.entities, left.entities, `${name}: compact entities`);
      assert.equal(right.relations.length, left.relations.length);
      for (let i = 0; i < left.relations.length; i++)
        assert.ok(
          Math.abs(left.relations[i] - right.relations[i]) < 1e-6,
          `${name}: geometry ${i} TS=${left.relations[i]} Rust=${right.relations[i]}`,
        );
    }
    if (process.argv.includes('--search')) {
      const keys = await native.request({ op: 'classic-keys', observation });
      assert.equal(keys.position, positionKey(imagined(observation)), `${name}: position key`);
      assert.equal(keys.fingerprint, fingerprint(imagined(observation)), `${name}: fingerprint`);
      for (const difficulty of ['easy', 'medium', 'hard'] as const) {
        const side = decisionOwner(observation),
          simulations = 40;
        const left = decide(observation, side, difficulty, { simulations, mode: 'work' });
        const right = await native.request({
          op: 'classic-decide',
          observation,
          side,
          difficulty,
          simulations,
        });
        assert.deepEqual(right, left, `${name} ${difficulty} search`);
      }
    }
    if (process.argv.includes('--evaluation')) {
      const left = explainEvaluation(imagined(observation), 1);
      const right = await native.request({ op: 'classic-evaluate', observation, side: 1 });
      for (const key of Object.keys(left) as (keyof typeof left)[])
        assert.ok(
          Math.abs(left[key] - right[key]) < 1e-7,
          `${name} ${key}: TS=${left[key]}, Rust=${right[key]}`,
        );
    }
    if (process.argv.includes('--candidates')) {
      for (const difficulty of ['easy', 'medium', 'hard'] as const) {
        const left = [...iterateCandidateGroups(imagined(observation), difficulty)].map((g) => ({
          family: g.family,
          keep: g.keep,
          priority: g.priority ?? null,
          commands: g.commands,
        }));
        const right = await native.request({ op: 'classic-candidates', observation, difficulty });
        assert.deepEqual(right, left, `${name} ${difficulty}`);
      }
    }
    checked++;
  }
  console.log(JSON.stringify({ classicFixtures: checked, invalidDomains }));
} finally {
  native.close();
}
