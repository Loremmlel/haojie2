/** Interactive or single-command headless play. JSON commands cover every engine capability. */
import { gunzipSync } from 'node:zlib';
import { createInterface } from 'node:readline';
import { stdin, stdout } from 'node:process';
import { existsSync, readFileSync, writeFileSync, renameSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { Arena } from '../../src/match/arena';
import type { ArenaEntry } from '../../src/match/arena';
import { parseSession, applyCommand } from '../../src/engine';
import { fingerprint } from '../../src/ai/observation';
import { candidateGroups } from '../../src/ai/candidates';
import { commandError } from '../../src/engine/game';
import type { Command } from '../../src/engine/types';
import { validMatch } from '../../src/match/settings';
import { render, describe, actions } from './cli/render';
import { replayTranscript, prepareTranscript, appendEntry } from './cli/transcript';

const args = process.argv.slice(2);
function flag(name: string, fallback?: string) {
  const at = args.indexOf('--' + name);
  return at < 0 ? fallback : args[at + 1];
}
function integer(name: string, fallback: number, min: number, max: number) {
  const n = Number(flag(name, String(fallback)));
  if (!Number.isSafeInteger(n) || n < min || n > max)
    throw new Error(`--${name} 必须是 ${min}..${max} 的整数`);
  return n;
}
const replay = flag('replay');
if (replay) {
  const data = readFileSync(replay);
  const result = replayTranscript(
    replay.endsWith('.gz') ? gunzipSync(data).toString('utf8') : data.toString('utf8'),
  );
  console.log(`回放 ${result.commands} 条命令全部匹配\n${render(result.state)}`);
  process.exit(0);
}
const save = flag('save', 'artifacts/cli-session.json')!,
  record = flag('record', save.replace(/\.json$/, '') + '.jsonl')!;
const match = {
  mode: 'ai' as const,
  human: integer('human', 1, 1, 2),
  difficulty: flag('difficulty', 'hard'),
};
if (!validMatch(match)) throw new Error('difficulty 必须为 easy / medium / hard');
const source = flag(
  'load',
  args.includes('--new') ? undefined : existsSync(save) ? save : undefined,
);
let arena = source
  ? new Arena(parseSession(readFileSync(source, 'utf8')))
  : Arena.create(integer('seed', 20260907, 1, 4294967295), match);
const limits = {
  ...(flag('nodes') ? { simulations: integer('nodes', 5500, 40, 100000) } : {}),
  ...(flag('ms') ? { milliseconds: integer('ms', 1000, 10, 100000) } : {}),
  trace: true,
};
function persist() {
  mkdirSync(dirname(save), { recursive: true });
  writeFileSync(save + '.tmp', JSON.stringify(arena.session));
  renameSync(save + '.tmp', save);
}
function log(entry: ArenaEntry) {
  appendEntry(record, entry);
}
function startRecord() {
  prepareTranscript(record, arena.session, args.includes('--new'));
}
const help = `show | actions ID | legal [ID] | go (AI直到轮到人类) | step (AI一步) | save | quit
summon [ultimate] | begin | deploy CARD X Y [charge] | move UNIT X Y | attack UNIT TARGET
charge UNIT attack/move/skill | cast CARD TARGET | equip CARD TARGET | finish UNIT | react TARGET | end
复杂技能直接输入JSON，例如 {"type":"skill","unitId":"u8","targetId":"u9","x":5,"y":6}
--new --seed N --human 1|2 --difficulty hard --save FILE；--command '指令' 可逐条无交互操作。
--replay FILE.jsonl 重放并逐步核对指纹。legal只是候选，不限制原始JSON的合法操作。`;
function parse(line: string): Command {
  if (line.startsWith('{')) return JSON.parse(line);
  const [type, a, b, c, d] = line.split(/\s+/);
  switch (type) {
    case 'summon':
      return { type, ultimate: a === 'ultimate' };
    case 'begin':
    case 'end':
      return { type };
    case 'deploy':
      return { type, cardId: a, x: Number(b), y: Number(c), charge: d === 'charge' };
    case 'move':
      return { type, unitId: a, x: Number(b), y: Number(c) };
    case 'attack':
      return { type, unitId: a, targetId: b };
    case 'charge':
      return { type, unitId: a, mode: b };
    case 'cast':
    case 'equip':
      return { type, cardId: a, targetId: b };
    case 'finish':
      return { type: 'finish-mode', unitId: a };
    case 'react':
      return { type, ...(b ? { x: Number(a), y: Number(b) } : { targetId: a }) };
    default:
      throw new Error('未知命令，输入 help。');
  }
}
function run(line: string): boolean {
  if (line === 'quit') return false;
  if (line === 'help') {
    console.log(help);
    return true;
  }
  if (line === 'show') {
    console.log(render(arena.session.present));
    return true;
  }
  if (line === 'save') {
    persist();
    console.log(save);
    return true;
  }
  if (line.startsWith('actions')) {
    console.log(actions(arena.session.present, line.split(/\s+/)[1]));
    return true;
  }
  if (line.startsWith('legal')) {
    const id = line.split(/\s+/)[1],
      s = arena.session.present;
    let n = 0;
    for (const g of candidateGroups(s, 'hard'))
      for (const c of g.commands)
        if ((!id || id === c.unitId || id === c.cardId) && !commandError(s, c)) {
          console.log(describe(s, c), JSON.stringify(c));
          if (++n >= 150) return true;
        }
    return true;
  }
  if (line === 'go' || line === 'step') {
    let count = 0;
    while (arena.computerTurn && count++ < (line === 'step' ? 1 : 200)) {
      const before = arena.session.present,
        t = performance.now(),
        entry = arena.step(limits)!;
      log(entry);
      persist();
      console.log(
        `AI P${entry.owner} ${describe(before, entry.command)} (${Math.round(performance.now() - t)}ms; 深${entry.decision?.stats.depth}; ${entry.decision?.stats.simulations}模拟)`,
      );
    }
    if (count >= 200) console.log('达到200条安全上限，局面已保存，未静默跳过。');
  } else {
    const entry = arena.play(parse(line));
    log(entry);
    persist();
    console.log(entry.events.map((e) => e.text ?? `${e.type}${e.amount ?? ''}`).join(' · '));
  }
  console.log(render(arena.session.present));
  return true;
}
{
  startRecord();
  persist();
  const single = flag('command');
  if (single) run(single);
  else {
    console.log(help + '\n' + render(arena.session.present));
    const rl = createInterface({ input: stdin, output: stdout, terminal: stdin.isTTY });
    for await (const line of rl) {
      try {
        if (!run(line.trim())) break;
      } catch (e) {
        console.error(e instanceof Error ? e.message : e);
      }
      if (stdin.isTTY) stdout.write('浩劫> ');
    }
    rl.close();
  }
}
