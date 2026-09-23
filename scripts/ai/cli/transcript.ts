import { RULESET_ID } from '../../../src/engine/catalog';
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { applyCommand, parseSession, createSession, sessionSave } from '../../../src/engine';
import type { Session } from '../../../src/engine';
import type { ArenaEntry } from '../../../src/match/arena';
import { fingerprint } from '../../../src/ai/observation';

/** 完整局面仅用于确定性本地回放，绝不发送给规划器。 */
export function replayTranscript(text: string) {
  // 网页存档与CLI命令行共用回放入口；旧JSONL仍逐步验证原指纹。
  if (text.trimStart().startsWith('{')) {
    let save;
    try {
      save = JSON.parse(text);
    } catch {
      /* 多行JSONL继续走原解析路径。 */
    }
    if (save?.format === 'haojie-record-v1') {
      const session = parseSession(text);
      return { state: session.present, commands: session.record!.cursor };
    }
  }
  const rows = text
    .trim()
    .split('\n')
    .map((line) => JSON.parse(line));
  if (rows[0]?.format !== 'haojie-cli-v1') throw new Error('不兼容的回放格式');
  if (rows[0].ruleset && rows[0].ruleset !== RULESET_ID)
    throw new Error(`回放规则版本${rows[0].ruleset}不兼容当前${RULESET_ID}`);
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
  writeFileSync(
    path,
    JSON.stringify({
      format: 'haojie-cli-v1',
      ruleset: RULESET_ID,
      initial: sessionSave(createSession(session.present, session.match)),
    }) + '\n',
  );
}
export function appendEntry(path: string, entry: ArenaEntry) {
  appendFileSync(path, JSON.stringify(entry) + '\n');
}
