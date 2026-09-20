import { StrictMode, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  ALL_CELLS,
  applyPlayerCommand,
  canDeployKind,
  commandError,
  createGame,
  expansionAnchors,
  template,
  type Command,
  type GameState,
  type Kind,
  type Player,
} from '../../src/engine';
import { HaojieOnlineGame } from '../../src/ui/online/HaojieOnlineGame';
import type { CommandContext, CommandReceipt, OnlineUpdate } from '../../src/ui/online/types';
import { DemoRoom } from '../../examples/online/room';
// Compile the documented component too; it is not mounted in this instrumented host.
export { TwoPlayerDemo } from '../../examples/online/TwoPlayerDemo';

type Scenario = 'classic' | 'combat' | 'shrine' | 'reaction' | 'giant' | 'path';
type Connection = 'connecting' | 'connected' | 'disconnected';
const copy = <T,>(value: T): T => JSON.parse(JSON.stringify(value)) as T;
function scenario(name: Scenario): GameState {
  if (name === 'classic') return createGame(19);
  if (name === 'shrine') return createGame(90, 'shrine');
  const s = createGame(19);
  s.phase = 'play';
  s.summonSlots = 0;
  s.ply = 5;
  s.turns = { 1: 3, 2: 2 };
  s.heads = { 1: 6, 2: 0 };
  s.log = [];
  s.events = [];
  const add = (kind: Kind, p: Player, x: number, y: number) => {
    const u = template(kind, p, 0, { x, y }, `fixture-${s.serial++}`);
    s.units.push(u);
    return u;
  };
  if (name === 'reaction') {
    s.pending = [
      {
        kind: 'death-shot',
        owner: 2,
        source: template(8, 2, 0, { x: 5, y: 8 }, 'dead'),
        amount: 10,
      },
    ];
  } else if (name === 'path') {
    const u = add('u6', 1, 3, 4);
    u.equipment.push('u28');
    add(1, 2, 3, 5);
    add(1, 2, 4, 5);
  } else {
    const fighter = add(9, 1, 3, 4);
    fighter.hp = fighter.maxHp = 500;
    const enemy = add(1, 2, 3, 5);
    enemy.hp = enemy.maxHp = 500;
    add('u7', 2, 2, 8);
    add(1, 1, 5, 8);
  }
  return s;
}
let identity = 0,
  sequence = 0;
let room = new DemoRoom(`test-${identity}`, scenario('classic'));
let updates = { 1: room.update(1, 'snapshot'), 2: room.update(2, 'snapshot') };
let connections: Record<Player, Connection> = { 1: 'connected', 2: 'connected' };
let history: Record<Player, OnlineUpdate[]> = { 1: [updates[1]], 2: [updates[2]] };
let holdAck = false,
  holdUpdates = false,
  rejectNext = false;
let deferred: (() => void)[] = [];
let render = () => {};
const attempts: { actor: Player; command: Command; baseRevision: number }[] = [];
let aborted = 0;
function publish(kind: OnlineUpdate['kind'] = 'update') {
  updates = { 1: room.update(1, kind), 2: room.update(2, kind) };
  history[1].push(updates[1]);
  history[2].push(updates[2]);
  render();
}
function send(actor: Player, command: Command, context: CommandContext): Promise<CommandReceipt> {
  const localRoom = room;
  attempts.push({ actor, command: copy(command), baseRevision: context.baseRevision });
  context.signal.addEventListener(
    'abort',
    () => {
      aborted++;
    },
    { once: true },
  );
  const receipt: CommandReceipt = rejectNext
    ? { ok: false, message: '测试宿主拒绝了这次操作。' }
    : localRoom.submit(actor, {
        requestId: `test-request-${++sequence}`,
        baseRevision: context.baseRevision,
        command: copy(command),
      });
  rejectNext = false;
  if (!holdUpdates) publish(receipt.ok ? 'update' : 'snapshot');
  return holdAck
    ? new Promise((resolve) => {
        deferred.push(() => resolve(receipt));
      })
    : Promise.resolve(receipt);
}
const controls = {
  reset(name: Scenario) {
    room = new DemoRoom(`test-${++identity}`, scenario(name));
    updates = { 1: room.update(1, 'snapshot'), 2: room.update(2, 'snapshot') };
    history = { 1: [updates[1]], 2: [updates[2]] };
    connections = { 1: 'connected', 2: 'connected' };
    holdAck = false;
    holdUpdates = false;
    rejectNext = false;
    attempts.length = 0;
    render();
  },
  configure(value: { holdAck?: boolean; holdUpdates?: boolean; rejectNext?: boolean }) {
    if (value.holdAck !== undefined) holdAck = value.holdAck;
    if (value.holdUpdates !== undefined) holdUpdates = value.holdUpdates;
    if (value.rejectNext !== undefined) rejectNext = value.rejectNext;
  },
  state: () => room.inspect(),
  views: () => copy(updates),
  attempts: () => copy(attempts),
  aborted: () => aborted,
  revision: () => room.revision,
  flushAcks() {
    const queued = deferred;
    deferred = [];
    for (const resolve of queued) resolve();
  },
  flushUpdates(kind: OnlineUpdate['kind'] = 'update') {
    publish(kind);
  },
  duplicate(player: Player) {
    updates = { ...updates, [player]: copy(updates[player]) };
    render();
  },
  stale(player: Player) {
    updates = { ...updates, [player]: copy(history[player][0]) };
    render();
  },
  connect(player: Player, connection: Connection) {
    connections = { ...connections, [player]: connection };
    if (connection === 'connected') publish('snapshot');
    else render();
  },
  /** A server-generated update used only for fault injection, still crosses the actor boundary. */
  server(actor: Player, command: Command) {
    const receipt = room.submit(actor, {
      requestId: `server-${++sequence}`,
      baseRevision: room.revision,
      command: copy(command),
    });
    if (!receipt.ok) throw new Error(receipt.message);
    publish();
    return receipt;
  },
  expandCommand(): Command {
    const s = room.inspect(),
      bw = s.units.find((u) => u.kind === 'u7')!,
      target = s.units.find((u) => u.x === 5 && u.y === 8)!;
    return { type: 'skill', unitId: bw.id, targetId: target.id, ...expansionAnchors(s, target)[0] };
  },
  nextDeployment() {
    const s = room.inspect(),
      card = s.hands[s.active].find((c) => canDeployKind(c.kind));
    if (!card) return null;
    for (const p of ALL_CELLS) {
      const command: Command = { type: 'deploy', cardId: card.id, ...p };
      if (commandError(s, command) === null)
        return { index: s.hands[s.active].indexOf(card), command, point: p };
    }
    throw new Error('No fixture deployment');
  },
  checkCommand(actor: Player, command: Command) {
    return applyPlayerCommand(room.inspect(), actor, command);
  },
};
declare global {
  interface Window {
    onlineDemo: typeof controls;
  }
}
window.onlineDemo = controls;
function Host() {
  const [, refresh] = useState(0);
  render = () => refresh((n) => n + 1);
  return (
    <>
      {([1, 2] as Player[]).map((p) => (
        <section id={`player-${p}`} key={p} style={{ marginBottom: 30 }}>
          <HaojieOnlineGame
            update={updates[p]}
            connection={connections[p]}
            onCommand={(c, context) => send(p, c, context)}
          />
        </section>
      ))}
    </>
  );
}
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Host />
  </StrictMode>,
);
