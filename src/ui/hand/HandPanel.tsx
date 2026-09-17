import type { ActionSpec, Command, GameState } from '../../engine';
import { faction } from '../../engine';
import { Icon } from '../shared/visuals';
import { HandCard } from './HandCard';
import { SynthesisControls } from './SynthesisControls';
import { SummonControls } from './SummonControls';

export function HandPanel({
  state: s,
  cardId,
  chooseCard,
  chooseAction,
  run,
  readOnly = false,
}: {
  state: GameState;
  cardId: string | null;
  chooseCard: (id: string) => void;
  chooseAction: (a: ActionSpec) => void;
  run: (command: Command) => void;
  readOnly?: boolean;
}) {
  const hand = s.hands[s.active],
    reaction = s.pending[0];
  return (
    <section className={`panel hand-panel ${s.phase === 'summon' ? 'summoning' : ''}`}>
      <div className="panel-heading">
        <h2>
          {s.phase === 'synthesis'
            ? '回合开始 · 合成'
            : s.phase === 'summon'
              ? '召唤仪式'
              : '本回合手牌'}
          {s.phase !== 'synthesis' && <b>{hand.length}</b>}
        </h2>
        <Icon name="spark" size={17} />
      </div>
      {s.phase === 'synthesis' && (
        <SynthesisControls
          key={`${s.ply}:${s.serial}`}
          state={s}
          run={run}
          chooseAction={chooseAction}
          readOnly={readOnly}
        />
      )}
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
            disabled={readOnly || !!reaction || !!s.winner || s.phase === 'synthesis'}
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
    </section>
  );
}
