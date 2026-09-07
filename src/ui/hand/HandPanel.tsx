import type { Command, GameState } from '../../engine';
import { faction } from '../../engine';
import { Icon } from '../shared/visuals';
import { HandCard } from './HandCard';
import { SummonControls } from './SummonControls';

export function HandPanel({
  state: s,
  cardId,
  chooseCard,
  run,
  readOnly = false,
}: {
  state: GameState;
  cardId: string | null;
  chooseCard: (id: string) => void;
  run: (command: Command) => void;
  readOnly?: boolean;
}) {
  const hand = s.hands[s.active],
    reaction = s.pending[0],
    craftCards = hand.filter((c) => c.kind === 'u28');
  return (
    <section className={`panel hand-panel ${s.phase === 'summon' ? 'summoning' : ''}`}>
      <div className="panel-heading">
        <h2>
          {s.phase === 'summon' ? '召唤仪式' : '本回合手牌'}
          <b>{hand.length}</b>
        </h2>
        <Icon name="spark" size={17} />
      </div>
      <SummonControls state={s} run={run} readOnly={readOnly} />
      <p className="hand-intro">
        <span className={`tiny-side p${s.active}`} />
        {faction(s.active)}
        <span>随从必须部署 · 法术与武器可储存</span>
      </p>
      <div className="hand-cards">
        {hand.map((c) => (
          <HandCard
            key={c.id}
            card={c}
            owner={s.active}
            turn={s.turns[s.active]}
            selected={c.id === cardId}
            disabled={readOnly || !!reaction || !!s.winner}
            onChoose={chooseCard}
          />
        ))}
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
          disabled={readOnly || !!reaction}
          onClick={() => run({ type: 'craft', cardIds: craftCards.slice(0, 3).map((c) => c.id) })}
        >
          <Icon name="spark" />
          3枚炎魔之心 → 炎魔之王
        </button>
      )}
    </section>
  );
}
