import type { Dispatch, SetStateAction } from 'react';
import type { ActionSpec, GameState } from '../../engine';
import { actionError, definition } from '../../engine';
import type { Intent } from '../game/selection';
import { Icon } from '../shared/visuals';

export function ActionPanel({
  state: s,
  actions,
  intent,
  activeIntent,
  setIntent,
  chooseAction,
  deaths,
}: {
  state: GameState;
  actions: ActionSpec[];
  intent: Intent;
  activeIntent: Intent;
  setIntent: Dispatch<SetStateAction<Intent>>;
  chooseAction: (action: ActionSpec) => void;
  deaths: GameState['deaths'];
}) {
  const reaction = s.pending[0];
  return (
    <>
      {!reaction && actions.length > 0 && (
        <div className="unit-actions">
          <h4>选择操作 · 剩余攻击不能换模式</h4>
          <div className="v2-actions">
            {actions.map((a) => {
              const error = actionError(s, a);
              return (
                <button
                  key={a.id}
                  disabled={!!error}
                  title={error ?? a.label}
                  aria-pressed={intent.kind === 'select' && intent.action.id === a.id}
                  onClick={() => chooseAction(a)}
                  className={a.free ? 'free-action' : ''}
                >
                  <Icon name={a.icon} size={14} />
                  {a.label}
                </button>
              );
            })}
          </div>
        </div>
      )}
      {activeIntent.kind === 'select' &&
        activeIntent.action.steps[activeIntent.index]?.kind === 'death' && (
          <div className="revival-picker">
            <h4>可复活的近期阵亡友方</h4>
            {deaths.length ? (
              deaths.map((r) => (
                <button
                  key={r.id}
                  onClick={() =>
                    setIntent({
                      ...activeIntent,
                      draft: { ...activeIntent.draft, deathId: r.id },
                      index: activeIntent.index + 1,
                    })
                  }
                >
                  {definition(r.kind).name}
                  <small>第{Math.ceil(r.ply / 2)}轮阵亡</small>
                </button>
              ))
            ) : (
              <p>最近两个己方回合窗口内没有可复活目标。</p>
            )}
          </div>
        )}
    </>
  );
}
