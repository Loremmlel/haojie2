import type { Dispatch, SetStateAction } from 'react';
import type { ActionSpec, GameState } from '../../engine';
import { definition, faction, getStats, occupants } from '../../engine';
import type { Intent } from '../game/selection';
import { DefinitionStats } from '../shared/DefinitionStats';
import { Icon, Rune } from '../shared/visuals';
import { ActionPanel } from './ActionPanel';
import { UnitDetails } from './UnitDetails';

export function Inspector({
  state: s,
  selectedId,
  cardId,
  intent,
  activeIntent,
  actions,
  setSelectedId,
  setIntent,
  chooseAction,
  onRules,
}: {
  state: GameState;
  selectedId: string | null;
  cardId: string | null;
  intent: Intent;
  activeIntent: Intent;
  actions: ActionSpec[];
  setSelectedId: Dispatch<SetStateAction<string | null>>;
  setIntent: Dispatch<SetStateAction<Intent>>;
  chooseAction: (action: ActionSpec) => void;
  onRules: () => void;
}) {
  const unit = s.units.find((u) => u.id === selectedId),
    card = s.hands[s.active].find((c) => c.id === cardId),
    reaction = s.pending[0];
  const inspected = reaction?.source ?? unit,
    d = card ? definition(card.kind) : inspected ? definition(inspected.kind) : undefined;
  const stats = inspected ? getStats(s, inspected) : null;
  const selectedBase = selectedId === 'base-1' ? 1 : selectedId === 'base-2' ? 2 : null;
  const deaths = inspected
    ? s.deaths.filter(
        (r) => r.owner === inspected.owner && !r.revived && r.ply < s.ply && r.ply >= s.ply - 4,
      )
    : [];

  return (
    <aside className="inspection-rail" tabIndex={0} aria-label="棋子详情与操作">
      <section className="panel inspector">
        <div className="panel-heading">
          <h2>{reaction ? '待结算效果' : card ? '召唤详情' : '棋子情报'}</h2>
          <span>{d?.tier === 'ultimate' ? 'ULTIMATE' : 'INSPECT'}</span>
        </div>
        {d ? (
          <>
            <div className="inspector-profile">
              <Rune kind={d.id} owner={card ? s.active : inspected!.owner} large />
              <span className="role-chip">
                {d.role}
                {card
                  ? ' · 手牌'
                  : ` · ${inspected && stats?.frozen ? '冰冻中立' : faction(inspected!.owner)}`}
              </span>
              <h3>{d.name}</h3>
            </div>
            {inspected && !card ? (
              <UnitDetails state={s} unit={inspected} />
            ) : (
              <DefinitionStats d={d} />
            )}
            <details className="ability-details">
              <summary>
                能力与规则 <Icon name="book" size={12} />
              </summary>
              <p className="ability-copy">{d.description}</p>
            </details>
            {unit && occupants(s, unit).length > 1 && (
              <div className="stack-selector">
                <span>同格棋子 · 点击切换</span>
                {occupants(s, unit).map((v, i) => (
                  <button
                    className={v.id === selectedId ? 'active' : ''}
                    onClick={() => {
                      setSelectedId(v.id);
                      setIntent({ kind: 'none' });
                    }}
                    key={v.id}
                  >
                    {i + 1}
                    {v.id === selectedId ? ' ✓' : ''}
                  </button>
                ))}
              </div>
            )}
            {card?.kind === 1 && intent.kind === 'select' && intent.draft.type === 'deploy' && (
              <label className="charge-choice">
                <input
                  type="checkbox"
                  checked={!!intent.draft.charge}
                  onChange={(e) =>
                    setIntent({
                      ...intent,
                      draft: { ...intent.draft, charge: e.target.checked },
                    })
                  }
                />
                <span>
                  <b>支付10生命，获得冲锋</b>
                  <small>当前与最大生命变为40</small>
                </span>
              </label>
            )}
          </>
        ) : selectedBase ? (
          <div className="base-inspector">
            <Icon name="crown" size={45} />
            <h3>{faction(selectedBase)}基地</h3>
            <strong>{s.bases[selectedBase]} / 300</strong>
            <p>普通治疗不能作用基地；灵魂法师虹吸是文档明确允许的例外。</p>
          </div>
        ) : (
          <div className="inspector-empty">
            <div className="empty-orbit">
              <Icon name="target" size={42} />
            </div>
            <h3>在浩劫中，落子。</h3>
            <p>每一枚棋子选择一种操作。击杀积累人头，下一回合开启终极召唤。</p>
            <button className="text-button" onClick={() => onRules()}>
              先了解新版规则 <Icon name="arrow" size={14} />
            </button>
          </div>
        )}
        <ActionPanel
          state={s}
          actions={actions}
          intent={intent}
          activeIntent={activeIntent}
          setIntent={setIntent}
          chooseAction={chooseAction}
          deaths={deaths}
        />
      </section>
      <section className="field-note">
        <span>THE RECKONING</span>
        <h3>一次选择，一场浩劫。</h3>
        <p>第四属性是攻击次数，不是自由行动点。主动蓄力不会与移动自动叠加。</p>
        <div>
          <kbd>Esc</kbd> 取消选点 <kbd>Ctrl / ⌘ Z</kbd> 悔棋
        </div>
      </section>
    </aside>
  );
}
