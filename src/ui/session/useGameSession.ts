import { useEffect, useRef, useState } from 'react';
import type { Command, GameEvent, Session } from '../../engine';
import { createGame, createSession, dispatch } from '../../engine';
import { Soundscape } from '../audio/Soundscape';
import type { HaojieGameProps } from '../game/types';
import type { SaveStatus } from './storage';
import { DEFAULT_STORAGE_KEY, readStoredSession, SAVE_LABELS, writeStoredSession } from './storage';

export function randomSeed(): number {
  return typeof crypto !== 'undefined' && crypto.getRandomValues
    ? crypto.getRandomValues(new Uint32Array(1))[0] || 1
    : Date.now() >>> 0 || 1;
}
function boot(props: HaojieGameProps, key: string | null) {
  const fallback = () => createSession(props.initialState ?? createGame(randomSeed()));
  if (props.initialState || !key || typeof window === 'undefined')
    return { session: fallback(), notice: '', writable: true };
  try {
    return readStoredSession(window.localStorage, key, fallback);
  } catch {
    return { session: fallback(), notice: '浏览器禁止本机保存，请导出对局。', writable: false };
  }
}
/** Owns the authoritative snapshot, persistence and disposable audiovisual effects only. */
export function useGameSession(props: HaojieGameProps) {
  const storageKey = props.storageKey === undefined ? DEFAULT_STORAGE_KEY : props.storageKey;
  const [initial] = useState(() => boot(props, storageKey));
  const [session, setSession] = useState(initial.session),
    live = useRef(session);
  const [writable, setWritable] = useState(initial.writable);
  const [status, setStatus] = useState<SaveStatus>(
    !storageKey ? 'disabled' : initial.writable ? 'saved' : 'blocked',
  );
  const [notice, setNotice] = useState(initial.notice);
  const [events, setEvents] = useState<GameEvent[]>([]),
    [sound, setSound] = useState(false);
  const audio = useRef<Soundscape | null>(null),
    callback = useRef(props.onStateChange);
  callback.current = props.onStateChange;
  useEffect(() => {
    if (!storageKey) {
      setStatus('disabled');
      return;
    }
    if (!writable) {
      setStatus('blocked');
      return;
    }
    let result: SaveStatus;
    try {
      result = writeStoredSession(window.localStorage, storageKey, session);
    } catch {
      result = 'unavailable';
    }
    setStatus(result);
    if (result === 'snapshot') setNotice('空间有限，仅保存当前局面；完整历史请导出。');
    if (result === 'unavailable') setNotice('浏览器禁止本机保存，请用“导出存档”保存对局。');
  }, [session, storageKey, writable]);
  useEffect(() => {
    callback.current?.(session.present);
  }, [session]);
  useEffect(() => {
    if (!events.length) return;
    const timer = setTimeout(() => setEvents([]), 1700);
    return () => clearTimeout(timer);
  }, [events]);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(''), 6500);
    return () => clearTimeout(timer);
  }, [notice]);
  useEffect(
    () => () => {
      audio.current?.dispose();
      audio.current = null;
    },
    [],
  );
  function replace(next: Session, explicit = false) {
    live.current = next;
    setSession(next);
    setEvents([]);
    if (explicit) setWritable(true);
  }
  function execute(command: Command) {
    const next = dispatch(live.current, command);
    live.current = next;
    setSession(next);
    setNotice('');
    setEvents(next.present.events.slice(-90));
    if (sound) {
      audio.current ??= new Soundscape();
      audio.current.play(next.present.events, true);
    }
    return next;
  }
  return {
    session,
    live,
    replace,
    execute,
    notice,
    setNotice,
    events,
    sound,
    setSound,
    saveStatus: SAVE_LABELS[status],
  };
}
