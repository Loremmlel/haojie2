import { matchSettings, ownsComputerDecision, rewindMatch } from '../../match/history';
import { LOCAL_MATCH, type MatchSettings } from '../../match/settings';
import { useComputer } from '../opponent/useComputer';
import { useMemo, useRef, useState } from 'react';
import type { ActionSpec, Command, Point, Session } from '../../engine';
import {
  actionError,
  applyCommand,
  cardActions,
  createDemoGame,
  createGame,
  createSession,
  occupants,
  reactionAction,
  targetAt,
  unitActions,
} from '../../engine';
import { randomSeed, useGameSession } from '../session/useGameSession';
import type { Intent } from './selection';
import { advanceIntent, canChoose, commandFor, startIntent } from './selection';
import type { GameModal, HaojieGameProps } from './types';
import { useGameHotkeys } from './useGameHotkeys';

/** UI selections are transient; every rule-changing action crosses the engine command boundary. */
export function useGameController(props: HaojieGameProps) {
  const game = useGameSession(props);
  const { session, live, setNotice } = game;
  const scope = useRef<HTMLDivElement>(null);
  const [intent, setIntent] = useState<Intent>({ kind: 'none' });
  const [selectedId, setSelectedId] = useState<string | null>(null),
    [cardId, setCardId] = useState<string | null>(null);
  const [modal, setModal] = useState<GameModal>(null);
  const s = session.present,
    controller = s.pending[0]?.owner ?? s.active,
    hand = s.hands[s.active],
    unit = s.units.find((u) => u.id === selectedId),
    card = hand.find((c) => c.id === cardId),
    reaction = s.pending[0];
  const computer = useComputer({
    session,
    live,
    apply: (c) => {
      game.execute(c);
      cancel();
      setSelectedId(c.unitId ?? null);
    },
    modal,
    notice: setNotice,
  });
  const activeIntent =
    reaction && !ownsComputerDecision(session) ? startIntent(reactionAction(s)!) : intent;
  const actions = useMemo(
    () =>
      ownsComputerDecision(session)
        ? []
        : card
          ? cardActions(s, card)
          : unit
            ? unitActions(s, unit)
            : [],
    [s, card, unit],
  );
  const endError = useMemo(() => {
    if (ownsComputerDecision(session)) return '当前由AI决策。';
    try {
      applyCommand(s, { type: 'end' });
      return null;
    } catch (e) {
      return e instanceof Error ? e.message : '不能结束回合';
    }
  }, [s]);

  function cancel() {
    setIntent({ kind: 'none' });
    setCardId(null);
  }
  function replace(next: Session, explicit = false) {
    computer.cancel();
    game.replace(next, explicit);
    cancel();
  }
  function rewind(forward = false) {
    const next = rewindMatch(live.current, forward);
    if (next === live.current) return;
    replace(next);
    if (ownsComputerDecision(next)) computer.pause();
    setNotice(
      forward ? '已重做，随机结果保持不变。' : '已悔棋，人头、武器、效果计时与随机数全部恢复。',
    );
  }
  useGameHotkeys(scope, !!modal, rewind, cancel);
  function run(c: Command) {
    if (ownsComputerDecision(live.current)) {
      setNotice('当前由AI决策，可查看棋盘或悔棋。');
      return;
    }
    try {
      const before = live.current,
        next = game.execute(c);
      setNotice('');
      setCardId(null);
      setIntent({ kind: 'none' });
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
    if (ownsComputerDecision(live.current)) return;
    const error = actionError(s, a);
    if (error) {
      setNotice(error);
      return;
    }
    if (!a.steps.length) run(a.command);
    else setIntent(startIntent(a));
  }
  function chooseCard(id: string) {
    if (reaction || ownsComputerDecision(live.current)) return;
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

  function importSession(next: Session) {
    replace(next, true);
    setSelectedId(null);
    computer.resume();
    setNotice('浩劫存档已载入，包含完整悔棋历史。');
  }
  function newGame(seedText: string, demo: boolean, match: MatchSettings = LOCAL_MATCH) {
    try {
      replace(
        createSession(
          demo ? createDemoGame() : createGame(seedText.trim() ? Number(seedText) : randomSeed()),
          match,
        ),
        true,
      );
      computer.resume();
      setSelectedId(null);
      setModal(null);
      setNotice(
        demo
          ? '已载入演示局，包含法师、装备、叠放军团和6人头。'
          : '浩劫新局开始。先召唤，再部署与行动。',
      );
    } catch (e) {
      setNotice(e instanceof Error ? e.message : '无法开始对局。');
    }
  }
  return {
    ...game,
    computer,
    match: matchSettings(session),
    resumeComputer: () => {
      if (live.current.future.length) replace({ ...live.current, future: [] });
      computer.resume();
    },
    scope,
    state: s,
    controller,
    intent,
    activeIntent,
    selectedId,
    cardId,
    modal,
    setModal,
    actions,
    endError,
    setSelectedId,
    setIntent,
    cancel,
    rewind,
    run,
    chooseAction,
    chooseCard,
    onCell,
    newGame,
    importSession,
  };
}
