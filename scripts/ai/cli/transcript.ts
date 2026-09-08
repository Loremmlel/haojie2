import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { applyCommand, parseSession } from '../../../src/engine';
import type { Session } from '../../../src/engine';
import type { ArenaEntry } from '../../../src/match/arena';
import { fingerprint } from '../../../src/ai/observation';

/** Full state is stored for deterministic local replay, never sent to the planner. */
export function replayTranscript(text: string) {
  const rows = text
    .trim()
    .split('\n')
    .map((line) => JSON.parse(line));
  if (rows[0]?.format !== 'haojie-cli-v1') throw new Error('不兼容的回放格式');
  let state = parseSession(JSON.stringify(rows[0].initial)).present;
  for (const [index, row] of rows.slice(1).entries()) {
    if (fingerprint(state) !== row.before) throw new Error(`第${index + 1}步之前不一致`);
    state = applyCommand(state, row.command);
    if (fingerprint(state) !== row.after) throw new Error(`第${index + 1}步之后不一致`);
  }
  return { state, commands: rows.length - 1 };
}
export function prepareTranscript(path: string, session: Session, replace: boolean) {
  mkdirSync(dirname(path), { recursive: true });
  if (!replace && existsSync(path)) {
    const last = replayTranscript(readFileSync(path, 'utf8'));
    if (fingerprint(last.state) !== fingerprint(session.present))
      throw new Error('存档与回放末尾不一致；请指定新的 --record 路径，未覆盖旧记录。');
    return;
  }
  writeFileSync(path, JSON.stringify({ format: 'haojie-cli-v1', initial: session }) + '\n');
}
export function appendEntry(path: string, entry: ArenaEntry) {
  appendFileSync(path, JSON.stringify(entry) + '\n');
}
