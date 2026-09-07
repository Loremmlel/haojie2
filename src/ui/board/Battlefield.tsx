import type { Command, GameEvent, GameState, Point } from '../../engine';
import { definition, faction, isStored } from '../../engine';
import type { Intent } from '../game/selection';
import { instruction } from '../game/selection';
import { Icon } from '../shared/visuals';
import { Board } from './Board';

export function Battlefield({
  state: s,
  intent,
  activeIntent,
  selectedId,
  onCell,
  events,
  canUndo,
  canRedo,
  rewind,
  run,
  onCancel,
  endError,
}: {
  state: GameState;
  intent: Intent;
  activeIntent: Intent;
  selectedId: string | null;
  onCell: (point: Point) => void;
  events: GameEvent[];
  canUndo: boolean;
  canRedo: boolean;
  rewind: (forward?: boolean) => void;
  run: (command: Command) => void;
  onCancel: () => void;
  endError: string | null;
}) {
  const reaction = s.pending[0],
    minions = s.hands[s.active].filter((c) => !isStored(definition(c.kind))).length;
  return (
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
          <button className="icon-button" aria-label="取消当前操作" onClick={() => onCancel()}>
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
          <button aria-label="悔棋" disabled={!canUndo} onClick={() => rewind()}>
            <Icon name="undo" />
            悔棋
          </button>
          <button aria-label="重做" disabled={!canRedo} onClick={() => rewind(true)}>
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
  );
}
