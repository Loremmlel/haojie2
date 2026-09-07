import { OpponentBar } from './opponent/OpponentBar';
import { Battlefield } from './board/Battlefield';
import { BattleLog } from './feedback/BattleLog';
import { Feedback } from './feedback/Feedback';
import { GameDialogs } from './feedback/GameDialogs';
import { GameHeader } from './game/GameHeader';
import { Scoreboard } from './game/Scoreboard';
import type { HaojieGameProps } from './game/types';
import { useGameController } from './game/useGameController';
import { VictoryBanner } from './game/VictoryBanner';
import { HandPanel } from './hand/HandPanel';
import { Inspector } from './inspector/Inspector';
import { SaveTools } from './session/SaveTools';
import { playerStyle } from './shared/visuals';
import './styles/index.css';
export type { HaojieGameProps } from './game/types';

/** Public React composition root. No rule mutations, storage or animation timers live here. */
export function HaojieGame(props: HaojieGameProps) {
  const game = useGameController(props);
  const { state, session } = game;
  return (
    <div ref={game.scope} className="hj-game v2-game" style={playerStyle(game.controller)}>
      <GameHeader
        sound={game.sound}
        onToggleSound={() => game.setSound((v) => !v)}
        onOpen={game.setModal}
      />
      <main className="game-container">
        <div className="match-heading">
          <p className="eyebrow">
            <span />
            {game.match.mode === 'ai' ? 'LOCAL AI DUEL' : 'LOCAL TWO-PLAYER DUEL'}
          </p>
          <p>
            普通召唤 × 26 <i /> 终极召唤 × 28
          </p>
        </div>
        <OpponentBar
          ended={!!state.winner}
          match={game.match}
          busy={game.computer.busy}
          thinking={game.computer.thinking}
          paused={game.computer.paused || !!session.future.length}
          backend={game.computer.backend}
          onToggle={() =>
            game.computer.paused || session.future.length
              ? game.resumeComputer()
              : game.computer.pause()
          }
          onNew={() => game.setModal('new')}
        />
        <Scoreboard state={state} human={game.match.mode === 'ai' ? game.match.human : undefined} />
        <VictoryBanner state={state} onNewGame={() => game.setModal('new')} />
        <div className="play-layout">
          <Inspector
            state={state}
            selectedId={game.selectedId}
            cardId={game.cardId}
            intent={game.intent}
            activeIntent={game.activeIntent}
            actions={game.actions}
            setSelectedId={game.setSelectedId}
            setIntent={game.setIntent}
            chooseAction={game.chooseAction}
            onRules={() => game.setModal('rules')}
          />
          <Battlefield
            state={state}
            intent={game.intent}
            activeIntent={game.activeIntent}
            selectedId={game.selectedId}
            onCell={game.onCell}
            events={game.events}
            canUndo={!!session.past.length}
            canRedo={!!session.future.length}
            rewind={game.rewind}
            run={game.run}
            onCancel={game.cancel}
            endError={game.endError}
            readOnly={game.computer.busy}
          />
          <aside className="hand-rail">
            <HandPanel
              state={state}
              cardId={game.cardId}
              chooseCard={game.chooseCard}
              run={game.run}
              readOnly={game.computer.busy}
            />
            <BattleLog
              state={state}
              saveStatus={game.saveStatus}
              onOpen={() => game.setModal('log')}
            />
            <SaveTools live={game.live} onImport={game.importSession} onError={game.setNotice} />
          </aside>
        </div>
        <footer className="game-footer">
          <span>
            <i className="live-dot" />
            浩劫2.1 · 离线单HTML · 双人 / 本地AI
          </span>
          <button onClick={() => game.setModal('rules')}>规则与实施说明</button>
          <span>SEED {state.seed}</span>
        </footer>
      </main>
      <Feedback events={game.events} notice={game.notice} onDismiss={() => game.setNotice('')} />
      <GameDialogs
        key={game.modal ?? 'closed'}
        modal={game.modal}
        state={state}
        match={game.match}
        onClose={() => game.setModal(null)}
        startGame={game.newGame}
      />
    </div>
  );
}
