import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
const read = (p) => readFileSync(p, 'utf8');
const write = (p, s) => writeFileSync(p, s);
function replace(p, from, to) {
  const s = read(p);
  if (!s.includes(from)) throw new Error(`Missing refactor anchor in ${p}: ${from.slice(0, 100)}`);
  if (s.indexOf(from) !== s.lastIndexOf(from)) throw new Error(`Ambiguous refactor anchor: ${p}`);
  write(p, s.replace(from, to));
}
const types = 'src/engine/types.ts';
replace(types, 'export interface GameState {\n  version: 2;\n  seed: number;\n  rng: number;', 'export interface GamePosition {\n  version: 2;');
replace(types, '/** Typed command payload shared by UI, saved replays and future server adapters. */', '/** Full authority. Public positions never manufacture these private random fields. */\nexport interface GameState extends GamePosition {\n  seed: number;\n  rng: number;\n}\n/** Typed command payload shared by UI, saved replays and future server adapters. */');
const privateFiles = new Set(['types.ts', 'game.ts', 'history.ts', 'migrations.ts', 'authority.ts', 'player-view.ts', 'index.ts']);
for (const entry of readdirSync('src/engine')) {
  if (!entry.endsWith('.ts') || privateFiles.has(entry)) continue;
  const p = `src/engine/${entry}`;
  write(p, read(p).replace(/\bGameState\b/g, 'GamePosition').replace(/\bcommandError\b/g, 'queryCommandError').replace(/\bisLegal\b/g, 'canAttemptCommand'));
}
replace('src/engine/state.ts', '  let x = s.rng;', "  ensure('rng' in s && typeof s.rng === 'number', '随机结算需要完整权威状态。');\n  let x = s.rng;");
const shrinePath = 'src/engine/shrines.ts';
{
  const s = read(shrinePath);
  const start = s.indexOf('export function chooseShrine(s: GamePosition, c: Command) {');
  const assignment = s.indexOf('  d.choices[p] =', start);
  if (start < 0 || assignment < 0) throw new Error('Shrine validation split anchor missing');
  const validation = s.slice(start, assignment).replace('function chooseShrine', 'function validateShrineChoice');
  write(shrinePath, s.slice(0, start) + validation + '  return { draft: d, player: p, kind: c.shrineKind };\n}\nexport function chooseShrine(s: GamePosition, c: Command) {\n  const { draft: d, player: p, kind } = validateShrineChoice(s, c);\n' + s.slice(assignment).replace('d.choices[p] = { kind: c.shrineKind,', 'd.choices[p] = { kind,'));
}
const game = 'src/engine/game.ts';
replace(game, '  chooseShrine,', '  chooseShrine,\n  validateShrineChoice,');
replace(game, "import type { Command, GameState, Kind, Player } from './types';", "import type { Command, GamePosition, GameState, Kind, Player } from './types';");
replace(game, 'export function applyCommand(\n  previous: GameState,\n  c: Command,\n  randomSource?: RandomSource,\n): GameState {', 'function transition<S extends GamePosition>(\n  previous: S,\n  c: Command,\n  randomSource?: RandomSource,\n  preview = false,\n): S {');
replace(game, "      case 'choose-shrine':\n        chooseShrine(s, c);", "      case 'choose-shrine':\n        if (preview) { validateShrineChoice(s, c); break; }\n        chooseShrine(s, c);");
replace(game, 'export function commandError(s: GameState, c: Command): string | null {', `function hasRandomState(s: GamePosition): s is GameState {
  return 'seed' in s && typeof s.seed === 'number' && 'rng' in s && typeof s.rng === 'number';
}
export function applyCommand(previous: GameState, c: Command, randomSource?: RandomSource): GameState {
  ensure(hasRandomState(previous), '权威结算需要完整随机状态；玩家视图不能代替GameState。');
  return transition(previous, c, randomSource);
}
const unresolvedRandom = Symbol('preview requires private randomness');
export type CommandInspection = { status: 'available' | 'uncertain' } | { status: 'invalid'; message: string };
/** Shared deterministic preflight. Stop BEFORE a random value is requested, never invent one.
 * No resulting state is returned, and a successful preflight is not an authoritative acceptance. */
export function inspectCommand(s: GamePosition, c: Command): CommandInspection {
  try {
    transition(s, c, () => { throw unresolvedRandom; }, true);
    return { status: 'available' };
  } catch (error) {
    if (error === unresolvedRandom) return { status: 'uncertain' };
    if (error instanceof RuleError) return { status: 'invalid', message: error.message };
    throw error;
  }
}
/** Local hints keep the exact existing full-state validation. Public hints never roll randoms. */
export function queryCommandError(s: GamePosition, c: Command): string | null {
  if (hasRandomState(s)) return commandError(s, c);
  const result = inspectCommand(s, c);
  return result.status === 'invalid' ? result.message : null;
}
export const canAttemptCommand = (s: GamePosition, c: Command) => queryCommandError(s, c) === null;
export function commandError(s: GameState, c: Command): string | null {`);
function walk(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((e) => e.isDirectory() ? walk(`${dir}/${e.name}`) : [`${dir}/${e.name}`]);
}
for (const p of walk('src/ui')) {
  if (!/\.tsx?$/.test(p) || p.startsWith('src/ui/session/') || p.startsWith('src/ui/opponent/') || p === 'src/ui/game/types.ts') continue;
  write(p, read(p).replace(/\bGameState\b/g, 'GamePosition').replace(/\bcommandError\b/g, 'queryCommandError').replace(/\bisLegal\b/g, 'canAttemptCommand'));
}
const header = 'src/ui/game/GameHeader.tsx';
replace(header, '  onOpen,\n}: {', '  onOpen,\n  allowNew = true,\n}: {');
replace(header, '  sound: boolean;', '  allowNew?: boolean;\n  sound: boolean;');
replace(header, `        <span className="nav-divider" />
        <button className="new-game-button" onClick={() => onOpen('new')}>
          <Icon name="plus" />
          新对局
        </button>`, `        {allowNew && <><span className="nav-divider" />
        <button className="new-game-button" onClick={() => onOpen('new')}>
          <Icon name="plus" />新对局
        </button></>}`);
const victory = 'src/ui/game/VictoryBanner.tsx';
replace(victory, '  onNewGame: () => void;', '  onNewGame?: () => void;');
replace(victory, '<p>浩劫落幕。仍可悔棋，重新推演最后一步。</p>', "<p>{onNewGame ? '浩劫落幕。仍可悔棋，重新推演最后一步。' : '浩劫落幕，对局已结束。'}</p>");
replace(victory, `      <button className="primary" onClick={() => onNewGame()}>
        再来一局
      </button>`, `      {onNewGame && <button className="primary" onClick={() => onNewGame()}>
        再来一局
      </button>}`);
const dialogs = 'src/ui/feedback/GameDialogs.tsx';
replace(dialogs, '  startGame: (seedText:', '  startGame?: (seedText:');
replace(dialogs, "{modal === 'new' && (", "{modal === 'new' && startGame && (");
replace(dialogs, 'subtitle="最近180条事件；悔棋时一并恢复。"', "subtitle={startGame ? '最近180条事件；悔棋时一并恢复。' : '最近180条公开事件。'}");
const battlefield = 'src/ui/board/Battlefield.tsx';
replace(battlefield, '  readOnly = false,\n}: {', '  readOnly = false,\n  showHistory = true,\n}: {');
replace(battlefield, '  readOnly?: boolean;', '  readOnly?: boolean;\n  showHistory?: boolean;');
{
  const s = read(battlefield), start = s.indexOf('        <div className="history-actions">'), end = s.indexOf('</div>', start) + 6;
  if (start < 0 || end < 6) throw new Error('History JSX anchor missing');
  write(battlefield, s.slice(0, start) + '        {showHistory && (' + s.slice(start, end).trim() + ')}' + s.slice(end));
}
replace(battlefield, "? '合成与部署一次确认；可取消选点或悔棋。'", "? showHistory ? '合成与部署一次确认；可取消选点或悔棋。' : '合成与部署一次确认；可取消选点。'");
const hand = 'src/ui/hand/HandPanel.tsx';
replace(hand, 'ActionSpec, Command, GamePosition', 'ActionSpec, Command, GamePosition, Player');
replace(hand, '  readOnly = false,\n}: {', '  readOnly = false,\n  viewer,\n}: {');
replace(hand, '  readOnly?: boolean;', '  readOnly?: boolean;\n  viewer?: Player;');
replace(hand, '      <ShrinePanel\n        state={s}', '      <ShrinePanel\n        viewer={viewer}\n        state={s}');
const shrinePanel = 'src/ui/shrine/ShrinePanel.tsx';
replace(shrinePanel, '  readOnly,\n}: {', '  readOnly,\n  viewer: seat,\n}: {');
replace(shrinePanel, '  readOnly: boolean;', '  readOnly: boolean;\n  viewer?: Player;');
replace(shrinePanel, "  const viewer: Player = match.mode === 'ai' ? match.human : s.active;", "  const viewer: Player = seat ?? (match.mode === 'ai' ? match.human : s.active);");
write(shrinePanel, read(shrinePanel).replace(/s\.active !== viewer/g, '(seat === undefined && s.active !== viewer)'));
replace(shrinePanel, "{match.mode === 'ai'\n            ?", "{seat !== undefined ? '对手的最终选择会在双方锁定后同时揭示。' : match.mode === 'ai'\n            ?");
const obs = 'src/ai/observation.ts';
{
  let s = read(obs);
  const start = s.indexOf('          shrineDraft: {'), end = s.indexOf('\n          },', start);
  if (start < 0 || end < 0) throw new Error('Observation projection anchor missing');
  s = "import { visibleShrineDraft } from '../engine/player-view';\n" + s.slice(0, start) + '          shrineDraft: visibleShrineDraft(s.shrineDraft, viewer),' + s.slice(end + '\n          },'.length);
  write(obs, s);
}
write('src/engine/index.ts', read('src/engine/index.ts') + "\nexport { inspectCommand, queryCommandError, canAttemptCommand } from './game';\nexport type { CommandInspection } from './game';\nexport { parseCommand, actorCommandError, applyPlayerCommand } from './authority';\nexport { getPlayerView, HAOJIE_RULESET, PLAYER_VIEW_VERSION } from './player-view';\nexport type { PlayerView } from './player-view';\n");
write('src/index.ts', read('src/index.ts') + "\nexport { HaojieOnlineGame } from './ui/online/HaojieOnlineGame';\nexport type { HaojieOnlineGameProps, OnlineUpdate, CommandReceipt, CommandContext } from './ui/online/types';\n");
write('src/ui/online/online.css', '.hj-game .online-status { margin: 0; padding: 0.55rem 0.7rem; font-size: 0.78rem; line-height: 1.4; }\n');
write('src/ui/online/HaojieOnlineGame.tsx', "import './online.css';\n" + read('src/ui/online/HaojieOnlineGame.tsx'));
// A directive must remain the first statement for Next's client boundary.
replace('src/ui/online/HaojieOnlineGame.tsx', "import './online.css';\n'use client';", "'use client';\nimport './online.css';");
write('AGENTS.md', read('AGENTS.md') + '\n## 联机适配边界\n\nHaojieGame保持本地协议；HaojieOnlineGame只消费PlayerView和宿主回执。共享GameSurface/交互/表现，不能给公开视图补seed/rng或启动本地AI/存档。公开预检在需要随机值前停止，只能作为可尝试提示，权威结果仍由applyPlayerCommand裁定。维护嵌套视图、暗选日志与回合外巨大化权限测试。网站负责身份、房间、修订号、请求去重和重连，不能在网站复制技能规则；更改规则同步HAOJIE_RULESET并固定双端代码版本。见docs/ONLINE-ADAPTATION.md。\n');
console.log('Applied scoped online adaptation; local rules and PRNG transitions are unchanged.');
