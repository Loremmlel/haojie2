import { useEffect, useMemo, useRef, useState } from 'react';
import {
  actionError,
  applyCommand,
  cardActions,
  createDemoGame,
  createGame,
  createSession,
  definition,
  dispatch,
  faction,
  getStats,
  isStored,
  now,
  occupants,
  parseSession,
  reactionAction,
  redo,
  targetAt,
  undo,
  unitActions,
} from '../engine';
import type { ActionSpec, Command, GameEvent, GameState, Player, Point, Session } from '../engine';
import { Board } from './Board';
import { advanceIntent, canChoose, commandFor, instruction, startIntent } from './intents';
import type { Intent } from './intents';
import { Codex, DefinitionStats, Modal, Rules } from './Library';
import { Icon, numberLabel, playerStyle, Rune } from './visuals';
import { Soundscape } from './sound';
import './styles.css';
import './v2.css';
export interface HaojieGameProps {
  initialState?: GameState;
  storageKey?: string | null;
  onStateChange?: (s: GameState) => void;
}
const seed = () =>
  typeof crypto !== 'undefined' && crypto.getRandomValues
    ? crypto.getRandomValues(new Uint32Array(1))[0] || 1
    : Date.now() >>> 0 || 1;
function initial(initialState?: GameState, key?: string | null) {
  if (initialState) return { session: createSession(initialState), notice: '' };
  try {
    if (key) {
      const saved = window.localStorage.getItem(key);
      if (saved) return { session: parseSession(saved), notice: '已恢复浩劫2.0对局。' };
      if (window.localStorage.getItem('haojie2.session.v1'))
        return {
          session: createSession(createGame(seed())),
          notice: '检测到旧版存档。新版规则已变化，旧存档保留不动，当前创建浩劫2.0新局。',
        };
    }
  } catch {
    return {
      session: createSession(createGame(seed())),
      notice: '未能读取旧存档，原文件未被修改。可手动导入浩劫2.0存档。',
    };
  }
  return { session: createSession(createGame(seed())), notice: '' };
}
const effectNames: Record<string, string> = {
  attack: '攻击强化',
  immune: '金身',
  execute: '死吧！',
  convert: '策反',
  mark: '投石标记',
  freeze: '冰冻 · 中立',
  burn: '灼烧',
  stun: '眩晕',
  'inner-fire': '心灵之火',
};
const modeNames: Record<string, string> = {
  none: '尚未选择',
  move: '移动模式',
  attack: '攻击模式',
  skill: '技能模式',
  charge: '蓄力模式',
};
export function HaojieGame({
  initialState,
  storageKey = 'haojie.session.v2',
  onStateChange,
}: HaojieGameProps) {
  const [boot] = useState(() => initial(initialState, storageKey)),
    [session, setSession] = useState(boot.session),
    live = useRef(session);
  const [intent, setIntent] = useState<Intent>({ kind: 'none' }),
    [selectedId, setSelectedId] = useState<string | null>(null),
    [cardId, setCardId] = useState<string | null>(null);
  const [modal, setModal] = useState<'rules' | 'codex' | 'new' | 'log' | null>(null),
    [notice, setNotice] = useState(boot.notice),
    [events, setEvents] = useState<GameEvent[]>([]),
    [sound, setSound] = useState(false),
    [seedText, setSeedText] = useState('');
  const audio = useRef(new Soundscape()),
    file = useRef<HTMLInputElement>(null),
    callback = useRef(onStateChange);
  callback.current = onStateChange;
  const s = session.present,
    controller = s.pending[0]?.owner ?? s.active,
    hand = s.hands[s.active],
    unit = s.units.find((u) => u.id === selectedId),
    card = hand.find((c) => c.id === cardId),
    reaction = s.pending[0];
  const activeIntent = reaction ? startIntent(reactionAction(s)!) : intent;
  const inspected = reaction?.source ?? unit,
    d = card ? definition(card.kind) : inspected ? definition(inspected.kind) : undefined,
    stats = inspected ? getStats(s, inspected) : null;
  const actions = useMemo(
    () => (card ? cardActions(s, card) : unit ? unitActions(s, unit) : []),
    [s, card, unit],
  );
  const minions = hand.filter((c) => !isStored(definition(c.kind))).length;
  const endError = useMemo(() => {
    try {
      applyCommand(s, { type: 'end' });
      return null;
    } catch (e) {
      return e instanceof Error ? e.message : '不能结束回合';
    }
  }, [s]);
  const selectedBase = selectedId === 'base-1' ? 1 : selectedId === 'base-2' ? 2 : null;
  useEffect(() => {
    if (storageKey)
      try {
        window.localStorage.setItem(storageKey, JSON.stringify(session));
      } catch {
        try {
          window.localStorage.setItem(storageKey, JSON.stringify(createSession(session.present)));
          setNotice('存储空间有限，本机只保存了当前局面；完整历史请导出。');
        } catch {
          setNotice('浏览器禁止本机保存，请用“导出存档”保存对局。');
        }
      }
    callback.current?.(session.present);
  }, [session, storageKey]);
  useEffect(() => {
    if (!events.length) return;
    const t = window.setTimeout(() => setEvents([]), 1700);
    return () => clearTimeout(t);
  }, [events]);
  useEffect(() => {
    if (!notice) return;
    const t = window.setTimeout(() => setNotice(''), 6500);
    return () => clearTimeout(t);
  }, [notice]);
  useEffect(() => () => audio.current.dispose(), []);
  function replace(next: Session) {
    live.current = next;
    setSession(next);
    setIntent({ kind: 'none' });
    setCardId(null);
    setEvents([]);
  }
  function rewind(forward = false) {
    const next = forward ? redo(live.current) : undo(live.current);
    replace(next);
    setNotice(
      forward ? '已重做，随机结果保持不变。' : '已悔棋，人头、武器、效果计时与随机数全部恢复。',
    );
  }
  useEffect(() => {
    const handle = (e: KeyboardEvent) => {
      if (
        modal ||
        (e.target instanceof HTMLElement &&
          e.target.closest('input,textarea,select,[contenteditable="true"]'))
      )
        return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        rewind(e.shiftKey);
      }
      if (e.key === 'Escape') {
        setIntent({ kind: 'none' });
        setCardId(null);
      }
    };
    window.addEventListener('keydown', handle);
    return () => window.removeEventListener('keydown', handle);
  }, [modal]);
  function run(c: Command) {
    try {
      const before = live.current,
        next = dispatch(before, c);
      live.current = next;
      setSession(next);
      setNotice('');
      setCardId(null);
      setIntent({ kind: 'none' });
      setEvents(next.present.events.slice(-90));
      audio.current.play(next.present.events, sound);
      if (c.type === 'end' || c.type === 'begin') setSelectedId(null);
      if (c.type === 'deploy')
        setSelectedId(next.present.events.find((e) => e.type === 'spawn')?.unitId ?? null);
      const id =
          c.unitId ?? (c.type === 'react' ? before.present.pending[0]?.source.id : undefined),
        u = next.present.units.find((u) => u.id === id);
      if (u && !next.present.pending.length && (u.mode === 'attack' || u.mode === 'move')) {
        const a = unitActions(next.present, u).find((a) => a.id === u.mode);
        if (a && !actionError(next.present, a)) setIntent(startIntent(a));
      }
    } catch (e) {
      setNotice(e instanceof Error ? e.message : '操作失败，当前局面没有改变。');
    }
  }
  function chooseAction(a: ActionSpec) {
    const error = actionError(s, a);
    if (error) {
      setNotice(error);
      return;
    }
    if (!a.steps.length) run(a.command);
    else setIntent(startIntent(a));
  }
  function chooseCard(id: string) {
    if (reaction) return;
    const c = hand.find((c) => c.id === id);
    if (!c) return;
    setCardId(id);
    setSelectedId(null);
    setIntent({ kind: 'none' });
    const candidates = cardActions(s, c);
    if (candidates.length === 1 && candidates[0].steps.length && !actionError(s, candidates[0]))
      setIntent(startIntent(candidates[0]));
  }
  function onCell(p: Point) {
    if (activeIntent.kind === 'none') {
      const stack = occupants(s, p);
      if (stack.length > 1) {
        const idx = stack.findIndex((u) => u.id === selectedId);
        setSelectedId(stack[(idx + 1) % stack.length].id);
      } else setSelectedId(targetAt(s, p)?.id ?? null);
      setCardId(null);
      return;
    }
    if (!canChoose(s, activeIntent, p)) {
      const c = commandFor(s, activeIntent, p);
      if (c)
        try {
          applyCommand(s, c);
        } catch (e) {
          setNotice(e instanceof Error ? e.message : '请选择高亮目标');
          return;
        }
      setNotice('请选择高亮目标；技能的选点顺序显示在棋盘上方。');
      return;
    }
    const c = commandFor(s, activeIntent, p);
    if (c) run(c);
    else setIntent(advanceIntent(activeIntent, p, s));
  }
  function exportSave() {
    const blob = new Blob([JSON.stringify(live.current, null, 2)], { type: 'application/json' }),
      url = URL.createObjectURL(blob),
      a = document.createElement('a');
    a.href = url;
    a.download = `haojie-2.0-turn-${s.ply}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async function importSave(f?: File) {
    if (!f) return;
    try {
      if (f.size > 24_000_000) throw new Error('存档超过24MB。');
      replace(parseSession(await f.text()));
      setSelectedId(null);
      setNotice('浩劫2.0存档已载入，包含完整悔棋历史。');
    } catch (e) {
      setNotice(e instanceof Error ? e.message : '载入失败。');
    }
    if (file.current) file.current.value = '';
  }
  function newGame(demo = false) {
    try {
      const state = demo
        ? createDemoGame()
        : createGame(seedText.trim() ? Number(seedText) : seed());
      replace(createSession(state));
      setSelectedId(null);
      setModal(null);
      setSeedText('');
      setNotice(
        demo
          ? '已载入演示局，包含法师、装备、叠放军团和6人头。'
          : '浩劫2.0新局开始。先召唤，再部署与行动。',
      );
    } catch (e) {
      setNotice(e instanceof Error ? e.message : '无法开始对局。');
    }
  }
  const reveals = events.filter((e) => e.type === 'summon');
  const craftCards = hand.filter((c) => c.kind === 'u28');
  const deaths = inspected
    ? s.deaths.filter(
        (r) => r.owner === inspected.owner && !r.revived && r.ply < s.ply && r.ply >= s.ply - 4,
      )
    : [];
  return (
    <div className="hj-game v2-game" style={playerStyle(controller)}>
      <header className="site-header">
        <div className="brand">
          <span className="brand-mark">
            <Icon name="sword" size={23} />
          </span>
          <div>
            <h1>
              浩劫<span>2.0</span>
            </h1>
            <p>HAOJIE · THE RECKONING</p>
          </div>
        </div>
        <nav aria-label="游戏工具">
          <button aria-label="棋子图鉴" onClick={() => setModal('codex')}>
            <Icon name="book" />
            <span>棋子图鉴</span>
          </button>
          <button aria-label="规则" onClick={() => setModal('rules')}>
            <Icon name="help" />
            <span>规则</span>
          </button>
          <button
            className="icon-button sound-toggle"
            aria-label={sound ? '关闭音效' : '开启音效'}
            aria-pressed={sound}
            onClick={() => setSound((v) => !v)}
          >
            <Icon name={sound ? 'volume' : 'mute'} />
          </button>
          <span className="nav-divider" />
          <button className="new-game-button" onClick={() => setModal('new')}>
            <Icon name="plus" />
            新对局
          </button>
        </nav>
      </header>
      <main className="game-container">
        <div className="match-heading">
          <p className="eyebrow">
            <span /> LOCAL TWO-PLAYER DUEL
          </p>
          <p>
            普通召唤 × 26 <i /> 终极召唤 × 28
          </p>
        </div>
        <section className="scoreboard" aria-label="双方基地、人头与回合状态">
          {([1, 2] as Player[]).map((p) => (
            <div
              key={p}
              className={`player-score p${p} ${controller === p ? 'current-player' : ''}`}
              style={playerStyle(p)}
            >
              <div className="player-seal">
                <Icon name="crown" size={28} />
              </div>
              <div className="player-score-main">
                <div className="player-name">
                  <h2>{faction(p)}</h2>
                  <span>PLAYER 0{p}</span>
                  {controller === p && <b>{reaction ? '结算中' : '你的回合'}</b>}
                </div>
                <div className="base-readout">
                  <strong>{s.bases[p]}</strong>
                  <span>/ 300</span>
                  <small className="heads-readout" aria-label={`${faction(p)}人头${s.heads[p]}`}>
                    <Icon name="sword" size={12} />
                    <b>{s.heads[p]}</b> 人头
                  </small>
                </div>
                <div className="base-health">
                  <i style={{ width: `${(s.bases[p] / 300) * 100}%` }} />
                </div>
              </div>
            </div>
          ))}
          <div className="round-medallion">
            <span>ROUND</span>
            <strong>{String(Math.ceil(s.ply / 2)).padStart(2, '0')}</strong>
            <i>
              {s.winner
                ? '终局'
                : reaction
                  ? '效果结算'
                  : s.phase === 'summon'
                    ? '召唤阶段'
                    : '行动阶段'}
            </i>
          </div>
        </section>
        {s.winner && (
          <section className="victory-banner" role="alert">
            <Icon name="crown" size={35} />
            <div>
              <h2>{s.winner === 'draw' ? '同归于尽，平局。' : `${faction(s.winner)}获胜`}</h2>
              <p>浩劫落幕。仍可悔棋，重新推演最后一步。</p>
            </div>
            <button className="primary" onClick={() => setModal('new')}>
              再来一局
            </button>
          </section>
        )}
        <div className="play-layout">
          <aside className="inspection-rail">
            <section className="panel inspector">
              <div className="panel-heading">
                <h2>{reaction ? '待结算效果' : card ? '召唤详情' : '棋子情报'}</h2>
                <span>{d?.tier === 'ultimate' ? 'ULTIMATE' : 'INSPECT'}</span>
              </div>
              {d ? (
                <>
                  <div className="inspector-profile">
                    <Rune kind={d.id} owner={card ? s.active : inspected!.owner} large />
                    <span className="role-chip">
                      {d.role}
                      {card
                        ? ' · 手牌'
                        : ` · ${inspected && stats?.frozen ? '冰冻中立' : faction(inspected!.owner)}`}
                    </span>
                    <h3>{d.name}</h3>
                  </div>
                  {inspected && !card && stats ? (
                    <>
                      <div className="live-stats">
                        {[
                          ['sword', '攻击', Math.round(stats.attack * 10) / 10],
                          [
                            'heart',
                            '生命',
                            `${Math.round(inspected.hp * 10) / 10}/${inspected.maxHp}`,
                          ],
                          ['target', '射程', stats.range],
                          ['clock', '攻击次数', stats.actions],
                          ['move', '移动', numberLabel(stats.move)],
                        ].map(([icon, label, value]) => (
                          <div key={label}>
                            <span>
                              <Icon name={String(icon)} size={13} />
                              {label}
                            </span>
                            <b>{value}</b>
                          </div>
                        ))}
                      </div>
                      <div className="operation-state">
                        <span>
                          {modeNames[inspected.mode]}
                          {stats.sleeping ? ' · 休整中' : ''}
                        </span>
                        <b>
                          操作 {stats.operationsLeft}/{stats.operationLimit}
                        </b>
                      </div>
                      {inspected.mode === 'attack' && (
                        <p className="sequence-hint">
                          本次攻击还剩 {stats.remaining} 次，不能切换移动或技能。
                        </p>
                      )}
                      {inspected.mode === 'move' && inspected.moves > 0 && (
                        <p className="sequence-hint">
                          冲撞还剩 {inspected.moves} 步；必须最终回到空地。
                        </p>
                      )}
                    </>
                  ) : (
                    <DefinitionStats d={d} />
                  )}
                  <details className="ability-details">
                    <summary>
                      能力与规则 <Icon name="book" size={12} />
                    </summary>
                    <p className="ability-copy">{d.description}</p>
                  </details>
                  {inspected && !card && (
                    <>
                      <div className="status-tags">
                        {inspected.silenced && <span>原技能已沉默</span>}
                        {inspected.guardUsed && <span>名刀已用</span>}
                        {inspected.effects.map((e, i) => (
                          <span key={i}>
                            {e.from > now(s, inspected) ? '下回合 · ' : ''}
                            {effectNames[e.type]}
                          </span>
                        ))}
                        {inspected.equipment.map((k) => (
                          <span className="equipment-tag" key={k}>
                            {definition(k).name}
                          </span>
                        ))}
                        {inspected.kills > 0 && <span>击杀 {inspected.kills}</span>}
                      </div>
                      {inspected.charge > 0 && (
                        <div className="charge-meter">
                          <span>
                            {inspected.chargeType === 'move'
                              ? '移动'
                              : inspected.chargeType === 'skill'
                                ? '技能'
                                : '攻击'}
                            蓄力{' '}
                            <b>
                              {inspected.charge} 层 · 就绪 {inspected.readyCharge}
                            </b>
                          </span>
                          <div>
                            {Array.from(
                              {
                                length:
                                  inspected.kind === 4
                                    ? 5
                                    : inspected.kind === 'u2'
                                      ? 4
                                      : inspected.kind === 21
                                        ? 2
                                        : 1,
                              },
                              (_, i) => (
                                <i key={i} className={i < inspected.charge ? 'filled' : ''} />
                              ),
                            )}
                          </div>
                        </div>
                      )}
                      {unit && occupants(s, unit).length > 1 && (
                        <div className="stack-selector">
                          <span>同格棋子 · 点击切换</span>
                          {occupants(s, unit).map((v, i) => (
                            <button
                              className={v.id === selectedId ? 'active' : ''}
                              onClick={() => {
                                setSelectedId(v.id);
                                setIntent({ kind: 'none' });
                              }}
                              key={v.id}
                            >
                              {i + 1}
                              {v.id === selectedId ? ' ✓' : ''}
                            </button>
                          ))}
                        </div>
                      )}
                    </>
                  )}
                  {card?.kind === 1 &&
                    intent.kind === 'select' &&
                    intent.draft.type === 'deploy' && (
                      <label className="charge-choice">
                        <input
                          type="checkbox"
                          checked={!!intent.draft.charge}
                          onChange={(e) =>
                            setIntent({
                              ...intent,
                              draft: { ...intent.draft, charge: e.target.checked },
                            })
                          }
                        />
                        <span>
                          <b>支付10生命，获得冲锋</b>
                          <small>当前与最大生命变为40</small>
                        </span>
                      </label>
                    )}
                </>
              ) : selectedBase ? (
                <div className="base-inspector">
                  <Icon name="crown" size={45} />
                  <h3>{faction(selectedBase)}基地</h3>
                  <strong>{s.bases[selectedBase]} / 300</strong>
                  <p>普通治疗不能作用基地；灵魂法师虹吸是文档明确允许的例外。</p>
                </div>
              ) : (
                <div className="inspector-empty">
                  <div className="empty-orbit">
                    <Icon name="target" size={42} />
                  </div>
                  <h3>在浩劫中，落子。</h3>
                  <p>每一枚棋子选择一种操作。击杀积累人头，下一回合开启终极召唤。</p>
                  <button className="text-button" onClick={() => setModal('rules')}>
                    先了解新版规则 <Icon name="arrow" size={14} />
                  </button>
                </div>
              )}
              {!reaction && actions.length > 0 && (
                <div className="unit-actions">
                  <h4>选择操作 · 剩余攻击不能换模式</h4>
                  <div className="v2-actions">
                    {actions.map((a) => {
                      const error = actionError(s, a);
                      return (
                        <button
                          key={a.id}
                          disabled={!!error}
                          title={error ?? a.label}
                          aria-pressed={intent.kind === 'select' && intent.action.id === a.id}
                          onClick={() => chooseAction(a)}
                          className={a.free ? 'free-action' : ''}
                        >
                          <Icon name={a.icon} size={14} />
                          {a.label}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
              {activeIntent.kind === 'select' &&
                activeIntent.action.steps[activeIntent.index]?.kind === 'death' && (
                  <div className="revival-picker">
                    <h4>可复活的近期阵亡友方</h4>
                    {deaths.length ? (
                      deaths.map((r) => (
                        <button
                          key={r.id}
                          onClick={() =>
                            setIntent({
                              ...activeIntent,
                              draft: { ...activeIntent.draft, deathId: r.id },
                              index: activeIntent.index + 1,
                            })
                          }
                        >
                          {definition(r.kind).name}
                          <small>第{Math.ceil(r.ply / 2)}轮阵亡</small>
                        </button>
                      ))
                    ) : (
                      <p>最近两个己方回合窗口内没有可复活目标。</p>
                    )}
                  </div>
                )}
            </section>
            <section className="field-note">
              <span>THE RECKONING</span>
              <h3>一次选择，一场浩劫。</h3>
              <p>第四属性是攻击次数，不是自由行动点。主动蓄力不会与移动自动叠加。</p>
              <div>
                <kbd>Esc</kbd> 取消选点 <kbd>Ctrl / ⌘ Z</kbd> 悔棋
              </div>
            </section>
          </aside>
          <section className="battle-column">
            <div className={`instruction-bar ${reaction ? 'reaction-bar' : ''}`} role="status">
              <span className="instruction-icon">
                <Icon name={reaction ? 'spark' : 'target'} />
              </span>
              <p>
                {reaction ? `${faction(reaction.owner)} · ` : ''}
                {instruction(s, activeIntent)}
              </p>
              {intent.kind !== 'none' && !reaction && (
                <button
                  className="icon-button"
                  aria-label="取消当前操作"
                  onClick={() => setIntent({ kind: 'none' })}
                >
                  <Icon name="x" size={16} />
                </button>
              )}
            </div>
            <Board
              state={s}
              intent={activeIntent}
              selectedId={selectedId}
              onCell={onCell}
              events={events}
            />
            <div className="command-bar">
              <div className="history-actions">
                <button aria-label="悔棋" disabled={!session.past.length} onClick={() => rewind()}>
                  <Icon name="undo" />
                  悔棋
                </button>
                <button
                  aria-label="重做"
                  disabled={!session.future.length}
                  onClick={() => rewind(true)}
                >
                  <Icon name="redo" />
                </button>
              </div>
              {reaction ? (
                <button
                  className="secondary finish-button"
                  disabled={reaction.kind === 'bounce'}
                  onClick={() => run({ type: 'react' })}
                >
                  {reaction.kind === 'bounce' ? '必须弹出空地' : '放弃此效果'}
                  <Icon name="arrow" />
                </button>
              ) : (
                <button
                  className="primary finish-button"
                  disabled={!!endError}
                  title={endError ?? '结束回合'}
                  onClick={() => run({ type: 'end' })}
                >
                  结束回合
                  <Icon name="arrow" />
                </button>
              )}
            </div>
            <p className="turn-hint">
              {reaction
                ? '效果由所属玩家处理，之后回到原有流程。'
                : s.phase === 'summon'
                  ? '在手牌区完成召唤选择，再开始行动。'
                  : minions
                    ? `还有 ${minions} 枚随从待部署。`
                    : '可以继续操作，或交给对手。'}
            </p>
          </section>
          <aside className="hand-rail">
            <section className={`panel hand-panel ${s.phase === 'summon' ? 'summoning' : ''}`}>
              <div className="panel-heading">
                <h2>
                  {s.phase === 'summon' ? '召唤仪式' : '本回合手牌'}
                  <b>{hand.length}</b>
                </h2>
                <Icon name="spark" size={17} />
              </div>
              {s.phase === 'summon' && !s.winner && (
                <div className="summon-controls">
                  <p>
                    剩余 <strong>{s.summonSlots}</strong> 次召唤 · 持有{' '}
                    <strong>{s.heads[s.active]}</strong> 人头
                  </p>
                  <button
                    className="secondary"
                    disabled={s.summonSlots <= 0 || !!reaction}
                    onClick={() => run({ type: 'summon' })}
                  >
                    <Icon name="plus" />
                    普通召唤
                  </button>
                  <button
                    className="ultimate-summon"
                    disabled={s.summonSlots <= 0 || s.heads[s.active] < 2 || !!reaction}
                    onClick={() => run({ type: 'summon', ultimate: true })}
                  >
                    <Icon name="spark" />
                    终极召唤 <span>−2 人头</span>
                  </button>
                  <button
                    className="primary"
                    disabled={s.summonSlots !== 0 || !!reaction}
                    onClick={() => run({ type: 'begin' })}
                  >
                    完成召唤，开始行动
                    <Icon name="arrow" />
                  </button>
                  <small>费用在揭示结果之前支付，可在开始行动前使用改判。</small>
                </div>
              )}
              <p className="hand-intro">
                <span className={`tiny-side p${s.active}`} />
                {faction(s.active)}
                <span>随从必须部署 · 法术与武器可储存</span>
              </p>
              <div className="hand-cards">
                {hand.map((c) => {
                  const cd = definition(c.kind),
                    limit = c.expiresAt === undefined ? null : c.expiresAt - s.turns[s.active];
                  return (
                    <button
                      key={c.id}
                      className={`hand-card ${cd.spell !== undefined ? 'spell-card' : ''} ${cd.weapon !== undefined ? 'weapon-card' : ''} ${cd.tier === 'ultimate' || c.kind === 'firelord' || c.kind === 'u12p' ? 'ultimate-card' : ''} ${c.id === cardId ? 'chosen' : ''}`}
                      aria-label={`选择${cd.name}${cd.weapon !== undefined ? '武器' : cd.spell !== undefined ? '法术' : '随从'}`}
                      aria-pressed={c.id === cardId}
                      disabled={!!reaction || !!s.winner}
                      onClick={() => chooseCard(c.id)}
                    >
                      <Rune kind={c.kind} owner={s.active} />
                      <div className="hand-card-main">
                        <div>
                          <h3>{cd.name}</h3>
                          <span className="summon-number">
                            {String(c.kind).startsWith('u')
                              ? String(c.kind).toUpperCase()
                              : typeof c.kind === 'number'
                                ? String(c.kind).padStart(2, '0')
                                : c.kind === '3p'
                                  ? '03′'
                                  : '◆'}
                          </span>
                        </div>
                        {isStored(cd) ? (
                          <p className={limit === 1 ? 'expires-soon' : ''}>
                            <Icon name={cd.weapon !== undefined ? 'sword' : 'clock'} size={12} />
                            {limit === null
                              ? '储限未注明 · 暂不限'
                              : limit === 1
                                ? '本回合到期'
                                : `剩余${limit}回合`}
                          </p>
                        ) : (
                          <p>
                            <span>
                              <Icon name="sword" size={11} />
                              {cd.attack}
                            </span>
                            <span>
                              <Icon name="heart" size={11} />
                              {cd.health}
                            </span>
                            <span>
                              <Icon name="target" size={11} />
                              {cd.range}
                            </span>
                          </p>
                        )}
                      </div>
                      {!isStored(cd) && (
                        <span className="deploy-label">{c.group ? '同批克隆 · ' : ''}待部署</span>
                      )}
                    </button>
                  );
                })}
                {hand.length === 0 && s.phase === 'play' && (
                  <div className="empty-hand">
                    <Icon name="check" size={28} />
                    <p>手牌已全部处理</p>
                  </div>
                )}
              </div>
              {craftCards.length >= 3 && s.phase === 'play' && (
                <button
                  className="craft-button"
                  disabled={!!reaction}
                  onClick={() =>
                    run({ type: 'craft', cardIds: craftCards.slice(0, 3).map((c) => c.id) })
                  }
                >
                  <Icon name="spark" />
                  3枚炎魔之心 → 炎魔之王
                </button>
              )}
            </section>
            <section className="panel battle-log">
              <div className="panel-heading">
                <h2>战场纪事</h2>
                <button className="text-button" onClick={() => setModal('log')}>
                  全部
                  <Icon name="arrow" size={13} />
                </button>
              </div>
              <ol>
                {s.log
                  .slice(-5)
                  .reverse()
                  .map((line, i) => (
                    <li key={`${i}-${line}`}>
                      <i />
                      <p>{line.split('·').slice(1).join('·').trim()}</p>
                    </li>
                  ))}
              </ol>
              <div className="log-footer">
                <i className="live-dot" />
                {storageKey ? '自动存档已启用' : '由宿主管理存档'}
              </div>
            </section>
            <div className="save-tools">
              <button onClick={exportSave}>
                <Icon name="download" size={14} />
                导出存档
              </button>
              <button onClick={() => file.current?.click()}>
                <Icon name="upload" size={14} />
                载入存档
              </button>
              <input
                hidden
                type="file"
                accept=".json,application/json"
                ref={file}
                onChange={(e) => void importSave(e.target.files?.[0])}
              />
            </div>
          </aside>
        </div>
        <footer className="game-footer">
          <span>
            <i className="live-dot" />
            浩劫2.0 · 离线单HTML · 同屏双人
          </span>
          <button onClick={() => setModal('rules')}>规则与待确认事项</button>
          <span>SEED {s.seed}</span>
        </footer>
      </main>
      {reveals.length > 0 && (
        <div
          className={`summon-reveal ${reveals.some((e) => e.ultimate) ? 'ultimate-reveal' : ''}`}
          role="status"
        >
          <span>{reveals.some((e) => e.ultimate) ? 'ULTIMATE SUMMON' : 'SUMMON'}</span>
          <Icon name="spark" size={28} />
          <strong>{reveals.map((e) => e.text).join(' · ')}</strong>
        </div>
      )}
      {notice && (
        <div className="toast" role="status">
          <Icon name="help" size={18} />
          <p>{notice}</p>
          <button className="icon-button" aria-label="关闭提示" onClick={() => setNotice('')}>
            <Icon name="x" size={16} />
          </button>
        </div>
      )}
      {modal === 'codex' && <Codex onClose={() => setModal(null)} />}
      {modal === 'rules' && <Rules onClose={() => setModal(null)} />}
      {modal === 'log' && (
        <Modal
          title="战场纪事"
          subtitle="最近180条事件；悔棋时一并恢复。"
          onClose={() => setModal(null)}
        >
          <ol className="full-log">
            {s.log.map((line, i) => (
              <li key={i}>{line}</li>
            ))}
          </ol>
        </Modal>
      )}
      {modal === 'new' && (
        <Modal
          title="浩劫，再起"
          subtitle="当前局面会被替换。重要对局请先导出存档。"
          onClose={() => setModal(null)}
        >
          <div className="new-match-art">
            <Icon name="sword" size={44} />
            <span>26普通召唤 · 28终极召唤 · 117格战场</span>
          </div>
          <label className="seed-field">
            对局种子<span>留空随机</span>
            <input
              type="number"
              step="1"
              value={seedText}
              onChange={(e) => setSeedText(e.target.value)}
              placeholder="例如 20260907"
            />
          </label>
          <button className="primary full-width" onClick={() => newGame(false)}>
            开始正式对局
            <Icon name="arrow" />
          </button>
          <button className="secondary full-width" onClick={() => newGame(true)}>
            载入演示棋局
          </button>
          <p className="fine-print">
            正式对局空棋盘、300血、0人头。演示局包含终极随从、装备、叠放和可兑换人头。
          </p>
        </Modal>
      )}
    </div>
  );
}
