import type { GameState } from '../../engine';
import { faction } from '../../engine';
import { Icon } from '../shared/visuals';

export function VictoryBanner({
  state: s,
  onNewGame,
}: {
  state: GameState;
  onNewGame: () => void;
}) {
  if (!s.winner) return null;
  return (
    <section className="victory-banner" role="alert">
      <Icon name="crown" size={35} />
      <div>
        <h2>{s.winner === 'draw' ? '同归于尽，平局。' : `${faction(s.winner)}获胜`}</h2>
        <p>浩劫落幕。仍可悔棋，重新推演最后一步。</p>
      </div>
      <button className="primary" onClick={() => onNewGame()}>
        再来一局
      </button>
    </section>
  );
}
