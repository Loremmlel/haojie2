'use client';
import { useRef, useState } from 'react';
import { createGame, type Player } from '../../src/engine';
import { HaojieOnlineGame, type HaojieOnlineGameProps } from '../../src/ui/online/HaojieOnlineGame';
import { DemoRoom } from './room';

/** Mock only: real authority must run on the website server, never inside its browser bundle. */
export function TwoPlayerDemo() {
  const [room] = useState(() => new DemoRoom('two-player-example', createGame(20260920)));
  const [updates, setUpdates] = useState(() => ({ 1: room.update(1, 'snapshot'), 2: room.update(2, 'snapshot') }));
  const sequence = useRef(0);
  const send = (actor: Player): HaojieOnlineGameProps['onCommand'] => async (command, context) => {
    if (context.signal.aborted) throw new Error('请求已取消。');
    const receipt = room.submit(actor, {
      requestId: `example-${++sequence.current}`,
      baseRevision: context.baseRevision,
      command: JSON.parse(JSON.stringify(command)),
    });
    // A real transport may deliver this update before or after the Promise's receipt.
    const kind = receipt.ok ? 'update' : 'snapshot';
    setUpdates({ 1: room.update(1, kind), 2: room.update(2, kind) });
    return receipt;
  };
  return <>{([1, 2] as Player[]).map((player) => <section key={player} aria-label={`玩家${player}的客户端`}>
    <HaojieOnlineGame update={updates[player]} connection="connected" onCommand={send(player)} />
  </section>)}</>;
}
