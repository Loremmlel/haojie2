import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { Arena } from '../../src/match/arena';
import { fingerprint } from '../../src/ai/observation';
import { prepareTranscript, appendEntry, replayTranscript } from '../../scripts/ai/cli/transcript';

test('headless game records real commands, rejects corrupt replays and will not mix a different save into an old transcript', () => {
  const folder = mkdtempSync(join(tmpdir(), 'haojie-arena-'));
  try {
    const path = join(folder, 'match.jsonl'),
      arena = Arena.create(7, { mode: 'ai', human: 1, difficulty: 'hard' });
    prepareTranscript(path, arena.session, false);
    for (const command of [{ type: 'summon' }, { type: 'summon' }, { type: 'begin' }] as const)
      appendEntry(path, arena.play(command));
    const text = readFileSync(path, 'utf8'),
      replay = replayTranscript(text);
    assert.equal(replay.commands, 3);
    assert.equal(fingerprint(replay.state), fingerprint(arena.session.present));
    assert.doesNotThrow(() => prepareTranscript(path, arena.session, false));
    assert.throws(() =>
      prepareTranscript(
        path,
        Arena.create(99, { mode: 'local', human: 1, difficulty: 'medium' }).session,
        false,
      ),
    );
    assert.equal(readFileSync(path, 'utf8'), text);
    const rows = text
      .trim()
      .split('\n')
      .map((l) => JSON.parse(l));
    rows[1].after = 'tampered';
    assert.throws(() => replayTranscript(rows.map((r) => JSON.stringify(r)).join('\n')));
  } finally {
    rmSync(folder, { recursive: true, force: true });
  }
});
test('command-line entry accepts piped human play and replay without loading or replacing an unrelated save', () => {
  const folder = mkdtempSync(join(tmpdir(), 'haojie-cli-'));
  try {
    const save = join(folder, 'save.json'),
      record = join(folder, 'match.jsonl');
    const play = spawnSync(
      process.execPath,
      [
        '--import',
        'tsx',
        'scripts/ai/play.ts',
        '--new',
        '--seed',
        '7',
        '--save',
        save,
        '--record',
        record,
      ],
      { input: 'summon\nsummon\nbegin\nquit\n', encoding: 'utf8', timeout: 15000 },
    );
    assert.equal(play.status, 0, play.stderr);
    assert.match(play.stdout, /P1 play/);
    const corrupt = join(folder, 'unrelated.json');
    writeFileSync(corrupt, 'not a save');
    const replay = spawnSync(
      process.execPath,
      ['--import', 'tsx', 'scripts/ai/play.ts', '--replay', record, '--save', corrupt],
      { encoding: 'utf8', timeout: 15000 },
    );
    assert.equal(replay.status, 0, replay.stderr);
    assert.match(replay.stdout, /回放 3 条命令全部匹配/);
    assert.equal(readFileSync(corrupt, 'utf8'), 'not a save');
  } finally {
    rmSync(folder, { recursive: true, force: true });
  }
});
