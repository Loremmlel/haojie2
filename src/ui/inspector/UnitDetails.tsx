import type { GameState, Unit } from '../../engine';
import { definition, getStats, now } from '../../engine';
import { Icon, numberLabel } from '../shared/visuals';

export function UnitDetails({ state: s, unit: inspected }: { state: GameState; unit: Unit }) {
  const effectNames: Record<string, string> = {
    attack: '攻击强化',
    immune: '金身',
    execute: '死吧！',
    convert: '策反',
    mark: '投石标记',
    freeze: '冰冻 · 中立',
    burn: '灼烧',
    stun: '眩晕',
    'inner-fire': '心灵之火',
  };
  const modeNames: Record<string, string> = {
    none: '尚未选择',
    move: '移动模式',
    attack: '攻击模式',
    skill: '技能模式',
    charge: '蓄力模式',
  };

  const stats = getStats(s, inspected);
  return (
    <>
      <>
        <div className="live-stats">
          {[
            ['sword', '攻击', Math.round(stats.attack * 10) / 10],
            ['heart', '生命', `${Math.round(inspected.hp * 10) / 10}/${inspected.maxHp}`],
            ['target', '射程', stats.range],
            ['clock', '攻击次数', stats.actions],
            ['move', '移动', numberLabel(stats.move)],
          ].map(([icon, label, value]) => (
            <div key={label}>
              <span>
                <Icon name={String(icon)} size={13} />
                {label}
              </span>
              <b>{value}</b>
            </div>
          ))}
        </div>
        <div className="operation-state">
          <span>
            {modeNames[inspected.mode]}
            {stats.sleeping ? ' · 休整中' : ''}
          </span>
          <b>
            操作 {stats.operationsLeft}/{stats.operationLimit}
          </b>
        </div>
        {inspected.mode === 'attack' && (
          <p className="sequence-hint">本次攻击还剩 {stats.remaining} 次，不能切换移动或技能。</p>
        )}
        {inspected.mode === 'move' && inspected.moves > 0 && (
          <p className="sequence-hint">冲撞还剩 {inspected.moves} 步；必须最终回到空地。</p>
        )}
      </>
      <>
        <div className="status-tags">
          {inspected.silenced && <span>原技能已沉默</span>}
          {inspected.guardUsed && <span>名刀已用</span>}
          {inspected.effects.map((e, i) => (
            <span key={i}>
              {e.from > now(s, inspected) ? '下回合 · ' : ''}
              {effectNames[e.type]}
            </span>
          ))}
          {inspected.equipment.map((k) => (
            <span className="equipment-tag" key={k}>
              {definition(k).name}
            </span>
          ))}
          {inspected.kills > 0 && <span>击杀 {inspected.kills}</span>}
        </div>
        {inspected.charge > 0 && (
          <div className="charge-meter">
            <span>
              {inspected.chargeType === 'move'
                ? '移动'
                : inspected.chargeType === 'skill'
                  ? '技能'
                  : '攻击'}
              蓄力{' '}
              <b>
                {inspected.charge} 层 · 就绪 {inspected.readyCharge}
              </b>
            </span>
            <div>
              {Array.from(
                {
                  length:
                    inspected.kind === 4
                      ? 5
                      : inspected.kind === 'u2'
                        ? 4
                        : inspected.kind === 21
                          ? 2
                          : 1,
                },
                (_, i) => (
                  <i key={i} className={i < inspected.charge ? 'filled' : ''} />
                ),
              )}
            </div>
          </div>
        )}
      </>
    </>
  );
}
