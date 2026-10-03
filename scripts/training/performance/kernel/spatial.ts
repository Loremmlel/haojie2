import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import { fixtures } from '../../native/fixtures';
import { combatFixtures } from '../../native/combat-fixtures';
import { completeFixtures } from '../../native/complete-fixtures';
import { preparationFixtures } from '../../native/preparation-fixtures';
import { attackPath, attackRoutes, targets } from '../../../../src/engine/core/geometry';
import { allPieces } from '../../../../src/engine/core/traits';

// 外部传入冻结的旧几何模块，逐条比较完整路径和方向顺序；不用于计时。
const baseline = await import(pathToFileURL(resolve(process.argv[2])).href);
let comparisons = 0;
for (const { name, job } of [
  ...fixtures(),
  ...combatFixtures(),
  ...completeFixtures(),
  ...preparationFixtures(),
]) {
  for (const u of allPieces(job.state).slice(0, 3)) {
    for (const t of targets(job.state)) {
      for (const limit of [0, 1.5, 3, 8]) {
        for (const pierce of [false, true]) {
          const label = `${name}:${u.id}:${t.id}:${limit}:${pierce}`;
          assert.deepEqual(
            attackPath(job.state, u, t, limit, undefined, pierce),
            baseline.attackPath(job.state, u, t, limit, undefined, pierce),
            label,
          );
          assert.deepEqual(
            attackRoutes(job.state, u, t, limit, pierce),
            baseline.attackRoutes(job.state, u, t, limit, pierce),
            label,
          );
          comparisons += 2;
        }
      }
    }
  }
}
process.stdout.write(JSON.stringify({ comparisons, passed: true }) + '\n');
