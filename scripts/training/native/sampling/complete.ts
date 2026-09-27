import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { freeze } from './artifacts';
import { nativeClient } from '../client';
import { createGame, applyCommand } from '../../../../src/engine/commands/game';
import { observe, fingerprint } from '../../../../src/ai/observation';
import { TrainingEnvironment } from '../../../../src/match/training';
import { recordHeader, readTrainingRecords } from '../../records/replay';
import { withRecordOutput } from '../../records/io';
import type { Command, Player } from '../../../../src/engine/types';

const { values } = parseArgs({
  options: {
    output: { type: 'string' },
    executable: {
      type: 'string',
      default: 'artifacts/native-target/release/haojie-engine-prototype.exe',
    },
    references: { type: 'string', default: 'artifacts/training/rust-fresh-games-20260927' },
  },
});
assert.ok(values.output);
const output = resolve(values.output);
const { executable, save } = await freeze(
  'scripts/training/native/sampling/complete.ts',
  output,
  values.executable,
);
const reports = [];
for (const rules of ['classic', 'shrine'] as const) {
  const source = readFileSync(join(values.references, `${rules}.json`));
  const reference = JSON.parse(source.toString());
  assert.equal(reference.options.rules, rules);
  assert.equal(reference.options.policy, 'tiny');
  assert.equal(reference.options.workers, 1);
  assert.equal(reference.options.worker, 0);
  assert.equal(reference.finalStateAndBothObservationsEqual, true);
  const client = await nativeClient(executable, 600_000);
  try {
    // 原生采样仅接收种子/预算；已有 TS 选招轨迹仅在采样完成后用于断言，绝不送入选择器。
    const native = await client.request({ op: 'sample-game', ...reference.options });
    assert.equal(native.error, null);
    assert.equal(native.status.terminated, true);
    assert.equal(native.status.truncated, false);
    assert.deepEqual(native.commands, reference.decisions);
    const commands: { actor: Player; command: Command }[] = native.commands;
    let state = createGame(reference.options.seed, rules);
    state.log = [];
    state.events = [];
    for (const { actor, command } of commands) {
      state = applyCommand(
        state,
        command.type === 'choose-shrine' ? { ...command, player: actor } : command,
      );
      state.log = [];
      state.events = [];
    }
    assert.deepEqual(native.state, JSON.parse(JSON.stringify(state)));
    assert.deepEqual(
      native.observations,
      [1, 2].map((p) => JSON.parse(JSON.stringify(observe(state, p as Player)))),
    );
    const env = new TrainingEnvironment(reference.options),
      path = join(output, `${rules}.jsonl.gz`);
    await withRecordOutput(path, async (emit) => {
      await emit({
        type: 'game',
        ...recordHeader(env),
        game: 0,
        gameId: `native-complete:${rules}:${reference.options.seed}`,
        seed: reference.options.seed,
        rules,
        source: 'neural',
        experiment: 'native-cold-start-economics',
      });
      for (const [index, { actor, command }] of commands.entries()) {
        const before = fingerprint(env.observation(actor));
        env.step(actor, command);
        await emit({
          type: 'decision',
          game: 0,
          index,
          actor,
          command,
          before,
          after: fingerprint(env.observation()),
          source: 'tiny',
        });
      }
      assert.deepEqual(env.status().returns, native.status.returns);
      await emit({
        type: 'outcome',
        game: 0,
        ...env.status(),
        after: fingerprint(env.observation()),
        interrupted: null,
        error: null,
      });
    });
    let checkedCommands = 0,
      checkedOutcomes = 0;
    for await (const row of readTrainingRecords(path)) {
      if (row.type === 'decision') checkedCommands++;
      if (row.type === 'outcome') {
        checkedOutcomes++;
        assert.equal(row.terminated, true);
      }
    }
    assert.equal(checkedCommands, commands.length);
    assert.equal(checkedOutcomes, 1);
    const report = {
      rules,
      seed: reference.options.seed,
      commands: commands.length,
      status: native.status,
      work: native.metrics,
      exactTsReferenceCommands: true,
      finalAuthorityAndBothObservationsEqual: true,
      verifiedTrainingRecord: path,
      referenceSha256: createHash('sha256').update(source).digest('hex'),
      note: 'Functional terminal check; reference is prior TS-selected full game, native selects from seed independently; timing is not used for speed claims',
    };
    save(`${rules}.json`, report);
    reports.push(report);
    console.log(JSON.stringify(report));
  } finally {
    client.close();
  }
}
save('complete.json', reports);
