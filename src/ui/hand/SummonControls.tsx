import type { Command, GameState } from '../../engine';
import { Icon } from '../shared/visuals';

export function SummonControls({
  state: s,
  run,
  readOnly = false,
}: {
  state: GameState;
  run: (command: Command) => void;
  readOnly?: boolean;
}) {
  const reaction = s.pending[0];
  if ((s.phase !== 'summon' && s.summonSlots <= 0) || s.winner) return null;
  return (
    <div className="summon-controls">
      <p>
        {s.phase === 'play' ? '本回合额外' : '剩余'} <strong>{s.summonSlots}</strong> 次召唤 · 持有{' '}
        <strong>{s.heads[s.active]}</strong> 人头
      </p>
      <button
        className="secondary"
        disabled={readOnly || s.summonSlots <= 0 || !!reaction}
        onClick={() => run({ type: 'summon' })}
      >
        <Icon name="plus" />
        普通召唤
      </button>
      <button
        className="ultimate-summon"
        disabled={readOnly || s.summonSlots <= 0 || s.heads[s.active] < 2 || !!reaction}
        onClick={() => run({ type: 'summon', ultimate: true })}
      >
        <Icon name="spark" />
        终极召唤 <span>−2 人头</span>
      </button>
      {s.phase === 'summon' && (
        <button
          className="primary"
          disabled={readOnly || s.summonSlots !== 0 || !!reaction}
          onClick={() => run({ type: 'begin' })}
        >
          完成召唤，开始行动
          <Icon name="arrow" />
        </button>
      )}
      <small>
        {s.phase === 'play'
          ? '额外机会仅本回合有效；付2人头可升级，费用在揭示结果前支付。'
          : '费用在揭示结果之前支付，可在开始行动前使用改判。'}
      </small>
    </div>
  );
}
