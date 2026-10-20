import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { mkdtemp, readFile, writeFile, access, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { gzipSync } from 'node:zlib';
import test from 'node:test';
import { nativeHash } from '../../scripts/training/native/pipeline/hash';
import { createGame } from '../../src/engine/commands/game';

test('原生记录规范固定字节向量覆盖负零、缺失值和UTF-8键顺序', () => {
  const vectors = JSON.parse(
    readFileSync(new URL('../../native/engine/data/hash-vectors.json', import.meta.url), 'utf8'),
  );
  for (const vector of vectors) {
    assert.equal(
      nativeHash(vector.value),
      createHash('sha256').update(Buffer.from(vector.bytes, 'hex')).digest('hex'),
    );
  }
  assert.equal(
    nativeHash(null),
    '6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d',
  );
  assert.notEqual(nativeHash({}), nativeHash({ a: null }));
  assert.equal(nativeHash({}), nativeHash({ a: undefined }));
});

test('原生摘要审核流式读取普通及压缩记录，损坏输入不发布报告', async () => {
  const folder = await mkdtemp(join(tmpdir(), 'haojie-native-audit-'));
  const state = createGame(173, 'classic');
  state.log = [];
  state.events = [];
  let previous = '';
  const rows = [
    {
      type: 'game',
      start: { seed: 173, rules: 'classic', prelude: [] },
      initialHash: nativeHash(state),
    },
    { type: 'outcome', commands: 0, terminated: false, winner: null },
  ].map((body) => {
    const content = { previous, body };
    previous = nativeHash(content);
    return { ...content, sha256: previous };
  });
  const text = rows.map((row) => JSON.stringify(row)).join('\n') + '\n';
  const run = promisify(execFile);
  const audit = (input: string, output: string) =>
    run(
      process.execPath,
      [
        '--import',
        'tsx',
        'scripts/training/native/pipeline/audit.ts',
        input,
        output,
        '--summary-only',
      ],
      { timeout: 10000, windowsHide: true },
    );
  try {
    const summaries = [];
    for (const suffix of ['.jsonl', '.jsonl.gz']) {
      const input = join(folder, 'valid' + suffix),
        output = input + '.summary.json';
      await writeFile(input, suffix.endsWith('.gz') ? gzipSync(text) : text);
      const { stdout } = await audit(input, output);
      const summary = JSON.parse(await readFile(output, 'utf8'));
      assert.equal(summary.commands, 0);
      assert.deepEqual(summary.examples, []);
      assert.equal(JSON.parse(stdout).finalHash, nativeHash(state));
      summaries.push(summary);
    }
    assert.deepEqual(summaries[0], summaries[1]);
    for (const [name, data] of [
      ['empty.jsonl', ''],
      ['bad-json.jsonl', text + '{'],
      ['bad-hash.jsonl', text.replace(rows[1].sha256, 'bad-hash')],
      ['truncated.jsonl.gz', gzipSync(text).subarray(0, -6)],
    ] as const) {
      const input = join(folder, name),
        output = input + '.summary.json';
      await writeFile(input, data);
      await assert.rejects(audit(input, output));
      await assert.rejects(access(output));
    }
  } finally {
    await rm(folder, { recursive: true, force: true });
  }
});
