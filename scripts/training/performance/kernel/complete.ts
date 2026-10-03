// 新双端各自从种子采样到真实终局，再与冻结参照逐条比较；不向选择器传参照命令。
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { nativeClient } from '../../native/client';
import type * as API from '../api';

const { values } = parseArgs({
  options: {
    api: { type: 'string' },
    executable: { type: 'string' },
    workset: { type: 'string' },
    output: { type: 'string' },
  },
});
assert.ok(values.api && values.executable && values.workset && values.output);
const output = resolve(values.output);
mkdirSync(output);
const api: typeof API = await import(pathToFileURL(resolve(values.api)).href);
const client = await nativeClient(resolve(values.executable), 1_800_000);
const canonical = (v: unknown) => JSON.parse(JSON.stringify(v));
const reports = [];
try {
  for (const [rules, seed] of [
    ['classic', 731270001],
    ['shrine', 741270001],
    ['classic', 731270031],
    ['shrine', 741270037],
  ] as const) {
    const reference = JSON.parse(
      readFileSync(join(values.workset, `${rules}-${seed}.json`), 'utf8'),
    );
    const native = await client.request({
      op: 'sample-game',
      rules,
      seed,
      policy: 'tiny',
      maxCommands: 12000,
      maxPlies: 500,
    });
    assert.equal(native.error, null);
    assert.equal(native.status.terminated, true);
    assert.deepEqual(native.commands, reference.decisions);
    assert.deepEqual(native.state, reference.state);
    writeFileSync(join(output, `${rules}-${seed}-native.json`), JSON.stringify(native), {
      flag: 'wx',
    });
    console.log(JSON.stringify({ stage: 'native', rules, seed, commands: native.commands.length }));
    const env = new api.TrainingEnvironment({ rules, seed, maxCommands: 12000, maxPlies: 500 });
    const decisions: any[] = [];
    const result = await api.sampleWorker(
      {
        rules,
        seed,
        policy: 'tiny',
        maxCommands: 12000,
        maxPlies: 500,
        seconds: 1800,
        games: 1,
        workers: 1,
        worker: 0,
      },
      async (row: any) => {
        if (row.type === 'decision') {
          assert.deepEqual(
            { actor: row.actor, command: row.command },
            reference.decisions[decisions.length],
          );
          decisions.push({ actor: row.actor, command: row.command });
        }
      },
      () => env,
      'commands',
    );
    assert.equal(result.results[0].error, null);
    assert.equal(result.results[0].interrupted, null);
    assert.equal(env.status().terminated, true);
    assert.deepEqual(decisions, reference.decisions);
    assert.deepEqual(canonical([env.observation(1), env.observation(2)]), native.observations);
    // 独立外部入口重放还核对正式 RNG、序号及所有权威字段，公开观察不替代完整状态。
    let state = api.createGame(seed, rules);
    state.log = [];
    state.events = [];
    for (const { actor, command } of decisions) {
      state = api.applyPlayerCommand(state, actor, command);
      state.log = [];
      state.events = [];
    }
    assert.deepEqual(canonical(state), native.state);
    const report = {
      rules,
      seed,
      commands: decisions.length,
      status: env.status(),
      exactCommands: true,
      exactAuthorityAndBothViews: true,
    };
    reports.push(report);
    writeFileSync(join(output, `${rules}-${seed}.json`), JSON.stringify({ report, result }), {
      flag: 'wx',
    });
    console.log(JSON.stringify(report));
  }
  writeFileSync(join(output, 'complete.json'), JSON.stringify(reports, null, 2), { flag: 'wx' });
} finally {
  client.close();
}
