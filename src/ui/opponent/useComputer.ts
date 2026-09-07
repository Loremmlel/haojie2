import { useEffect, useRef, useState } from 'react';
import { AiClient } from '../../ai/client';
import { DIFFICULTIES } from '../../ai/difficulty';
import { decisionOwner, fingerprint, observe } from '../../ai/observation';
import type { Decision, PlanStep } from '../../ai/types';
import type { Command, Player, Session } from '../../engine';
import { matchSettings, ownsComputerDecision } from '../../match/history';
import type { GameModal } from '../game/types';
import { actionDelay, waitForPresentation } from './pacing';
interface Port {
  session: Session;
  live: { current: Session };
  apply: (command: Command) => void;
  modal: GameModal;
  notice: (message: string) => void;
}
/** Cancels stale work on every state revision. Workers never receive the saved game's RNG. */
export function useComputer({ session, live, apply, modal, notice }: Port) {
  const client = useRef<AiClient | null>(null),
    revision = useRef(0),
    cache = useRef<PlanStep[]>([]);
  const presentation = useRef<AbortController | null>(null);
  const lastAction = useRef<{ command: Command; ply: number; owner: Player } | null>(null);
  const port = useRef({ apply, notice });
  port.current = { apply, notice };
  const budget = useRef({ ply: -1, nodes: 0, ms: 0, commands: 0 });
  const [paused, setPaused] = useState(false),
    [thinking, setThinking] = useState(false);
  const [stats, setStats] = useState<Decision['stats'] | null>(null),
    [backend, setBackend] = useState('worker');
  const busy = ownsComputerDecision(session),
    match = matchSettings(session);
  function cancel(pause = false) {
    revision.current++;
    presentation.current?.abort();
    presentation.current = null;
    lastAction.current = null;
    client.current?.cancel();
    cache.current = [];
    budget.current = { ply: -1, nodes: 0, ms: 0, commands: 0 };
    setThinking(false);
    if (pause) setPaused(true);
  }
  useEffect(() => {
    if (!busy || paused || modal || session.future.length) {
      lastAction.current = null;
      setThinking(false);
      return;
    }
    const id = ++revision.current,
      source = session.present;
    const player = decisionOwner(source),
      cfg = DIFFICULTIES[match.difficulty];
    if (budget.current.ply !== source.ply)
      budget.current = { ply: source.ply, nodes: 0, ms: 0, commands: 0 };
    setThinking(true);
    const startedDecision = performance.now();
    const abort = new AbortController();
    presentation.current = abort;
    const previous = lastAction.current;
    const first = !previous || previous.ply !== source.ply || previous.owner !== player;
    let alive = true;
    const timer = setTimeout(
      async () => {
        if (!alive || abort.signal.aborted) return;
        try {
          const predicted = cache.current[0];
          let command: Command | null = null;
          if (predicted?.before === fingerprint(source)) {
            command = predicted.command;
            cache.current.shift();
          } else {
            cache.current = [];
            client.current ??= new AiClient();
            const remainingNodes = cfg.turnNodes - budget.current.nodes,
              remainingMs = cfg.turnMs - budget.current.ms;
            const activeUnits = source.units.filter(
              (u) =>
                u.owner === player &&
                (u.operations === 0 || u.mode === 'attack' || u.mode === 'move'),
            ).length;
            const allowance = Math.max(
              80,
              Math.min(cfg.nodes, remainingNodes / Math.max(1, Math.min(4, activeUnits))),
            );
            const milliseconds = Math.max(35, Math.min(cfg.decisionMs, remainingMs));
            const started = performance.now();
            const result = await client.current.plan({
              id,
              observation: observe(source),
              side: player,
              difficulty: match.difficulty,
              limits: { simulations: allowance, milliseconds },
            });
            if (!alive || id !== revision.current || live.current.present !== source) return;
            budget.current.nodes += result.stats.simulations;
            budget.current.ms += performance.now() - started;
            setStats(result.stats);
            setBackend(client.current.mode);
            command = result.command;
            cache.current = result.plan.slice(1);
          }
          if (
            !alive ||
            id !== revision.current ||
            live.current.present !== source ||
            !ownsComputerDecision(live.current)
          )
            return;
          if (!command) throw new Error('没有找到合法操作。');
          const minimum = actionDelay(command, {
            first,
            previous: first ? null : previous.command,
            events: source.events,
          });
          setThinking(false);
          await waitForPresentation(minimum - (performance.now() - startedDecision), abort.signal);
          // Pausing/undo/import can happen after planning, while we are only presenting a pause.
          if (
            !alive ||
            abort.signal.aborted ||
            id !== revision.current ||
            live.current.present !== source ||
            !ownsComputerDecision(live.current)
          )
            return;
          if (++budget.current.commands > 200) {
            setPaused(true);
            port.current.notice('AI本轮操作较多，已暂停。可以继续计算或悔棋。');
            return;
          }
          lastAction.current = { command, ply: source.ply, owner: player };
          port.current.apply(command);
        } catch (e) {
          if (!alive || (e instanceof DOMException && e.name === 'AbortError')) return;
          cache.current = [];
          setPaused(true);
          port.current.notice(
            `AI已暂停，局面未被跳过：${e instanceof Error ? e.message : '计算失败'}。可点击继续AI重试或悔棋。`,
          );
        } finally {
          if (alive && id === revision.current) setThinking(false);
        }
      },
      0, // Start computing promptly; presentation and search run on overlapping clocks.
    );
    return () => {
      alive = false;
      clearTimeout(timer);
      abort.abort();
      if (presentation.current === abort) presentation.current = null;
      revision.current++;
      client.current?.cancel();
    };
  }, [session, paused, modal, busy, match.difficulty]);
  useEffect(
    () => () => {
      client.current?.cancel();
      client.current = null;
    },
    [],
  );
  function resume() {
    lastAction.current = null;
    cache.current = [];
    budget.current = { ply: -1, nodes: 0, ms: 0, commands: 0 };
    setPaused(false);
  }
  return {
    busy,
    paused,
    thinking,
    stats,
    backend,
    cancel,
    resume,
    pause: () => {
      cancel(true);
    },
  };
}
