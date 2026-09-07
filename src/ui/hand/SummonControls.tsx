import type { Command, GameState } from '../../engine';
import { Icon } from '../shared/visuals';

export function SummonControls({
  state: s,
  run,
}: {
  state: GameState;
  run: (command: Command) => void;
}) {
  const reaction = s.pending[0];
  if (s.phase !== 'summon' || s.winner) return null;
  return (
    <div className="summon-controls">
      <p>
        剩余 <strong>{s.summonSlots}</strong> 次召唤 · 持有 <strong>{s.heads[s.active]}</strong>{' '}
        人头
      </p>
      <button
        className="secondary"
        disabled={s.summonSlots <= 0 || !!reaction}
        onClick={() => run({ type: 'summon' })}
      >
        <Icon name="plus" />
        普通召唤
      </button>
      <button
        className="ultimate-summon"
        disabled={s.summonSlots <= 0 || s.heads[s.active] < 2 || !!reaction}
        onClick={() => run({ type: 'summon', ultimate: true })}
      >
        <Icon name="spark" />
        终极召唤 <span>−2 人头</span>
      </button>
      <button
        className="primary"
        disabled={s.summonSlots !== 0 || !!reaction}
        onClick={() => run({ type: 'begin' })}
      >
        完成召唤，开始行动
        <Icon name="arrow" />
      </button>
      <small>费用在揭示结果之前支付，可在开始行动前使用改判。</small>
    </div>
  );
}
