import type { GameState, Unit } from '../../engine';
import { unitStatus } from './unit-status';

export function UnitStatus({ state, unit }: { state: GameState; unit: Unit }) {
  const rows = unitStatus(state, unit);
  return (
    <section className="unit-status" aria-label="棋子当前状态">
      <h4>当前状态</h4>
      {rows.length ? (
        <ul>
          {rows.map((row) => (
            <li key={row.key} className={row.pending ? 'upcoming' : undefined}>
              <b>
                {row.pending ? '待生效 · ' : ''}
                {row.label}
              </b>
              <span>{row.detail}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p>暂无持续效果、装备或外部技能影响。</p>
      )}
    </section>
  );
}
