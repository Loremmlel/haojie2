import test from 'node:test';
import assert from 'node:assert/strict';
import { appendFile, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { encodeTeacherFile } from '../../scripts/training/encode';
import { TrainingEnvironment } from '../../src/match/training';
import { HAOJIE_RULESET } from '../../src/engine/online/player-view';

test('教师转换保留中断局的策略样本，标签与宿主种子不进入网络输入', async () => {
  const folder = await mkdtemp(join(tmpdir(), 'haojie-encoding-'));
  try {
    const path = join(folder, 'teacher.jsonl');
    const env = new TrainingEnvironment({ seed: 19 });
    await writeFile(
      path,
      [
        { type: 'game', game: 0, ruleset: HAOJIE_RULESET, rules: 'classic', seed: 19, budget: 40 },
        {
          type: 'sample',
          game: 0,
          index: 0,
          actor: 1,
          observation: env.observation(),
          command: { type: 'summon' },
        },
      ]
        .map((r) => JSON.stringify(r))
        .join('\n'),
      'utf8',
    );
    const rows = [];
    for await (const row of encodeTeacherFile(path)) rows.push(row);
    const ending = rows.at(-1)!;
    assert.equal(ending.type, 'outcome');
    assert.ok('interrupted' in ending && ending.interrupted);
    assert.ok('returns' in ending && ending.returns === null);
    const sample = rows.find((r) => r.type === 'example')!;
    assert.ok('input' in sample && sample.input);
    for (const forbidden of ['seed', 'rng', 'policy', 'value', 'selected', 'observation'])
      assert.equal(forbidden in sample.input, false);
    await appendFile(
      path,
      '\n' +
        JSON.stringify({
          type: 'outcome',
          game: 0,
          commands: 1,
          terminated: false,
          truncated: false,
          interrupted: 'cancelled',
          returns: null,
          winner: null,
        }),
    );
    const explicit = [];
    for await (const row of encodeTeacherFile(path)) explicit.push(row);
    const outcome = explicit.at(-1)!;
    assert.ok('interrupted' in outcome && outcome.interrupted);
    assert.ok('interruptionReason' in outcome && outcome.interruptionReason === 'cancelled');
    assert.ok('returns' in outcome && outcome.returns === null);
  } finally {
    await rm(folder, { recursive: true, force: true });
  }
});
