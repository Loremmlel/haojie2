import type { GameEvent, Point } from '../../engine';
export function Effects({ events, reduced }: { events: GameEvent[]; reduced: boolean }) {
  const xy = (p: Point) => ({ x: (p.x - 0.5) * 100, y: (p.y - 0.5) * 100 });
  return (
    <svg className="effects-layer" viewBox="0 0 900 1300" aria-hidden="true">
      {events
        .filter((e) => e.to)
        .map((e) => {
          const to = xy(e.to!),
            from = e.from ? xy(e.from) : to;
          const route = (e as GameEvent & { path?: Point[] }).path?.map(xy) ?? [from, to];
          const path = route.map((p, i) => `${i ? 'L' : 'M'}${p.x} ${p.y}`).join(' ');
          return (
            <g key={e.id} className={`fx fx-${e.type} p${e.owner ?? 1}`}>
              {e.type === 'attack' && (
                <>
                  <path d={path} className="shot-trail" />
                  <circle r="8" fill="currentColor" className="shot-core">
                    {!reduced && <animateMotion path={path} dur="0.42s" fill="freeze" />}
                  </circle>
                  <circle cx={to.x} cy={to.y} r="23" className="impact-ring" />
                </>
              )}
              {e.type === 'move' && <path d={path} className="move-trail" />}
              {['spawn', 'shield', 'skill', 'death'].includes(e.type) && (
                <g transform={`translate(${to.x} ${to.y})`}>
                  <circle
                    r={e.text === '爆弹' ? 92 : e.type === 'death' ? 40 : 35}
                    className={`aura-ring ${e.text === '爆弹' ? 'blast' : ''}`}
                  />
                  {[0, 1, 2, 3, 4, 5, 6, 7].map((n) => (
                    <path
                      key={n}
                      d="M 0 -32 L 0 -52"
                      transform={`rotate(${n * 45})`}
                      className="spark-ray"
                    />
                  ))}
                  {e.text && e.type !== 'spawn' && (
                    <text y="-40" className="effect-label">
                      {e.text}
                    </text>
                  )}
                </g>
              )}
              {(e.type === 'damage' || e.type === 'heal') && (
                <g transform={`translate(${to.x} ${to.y - 24})`}>
                  <text className={`damage-number ${e.type === 'heal' ? 'healing' : ''}`}>
                    {e.type === 'heal' ? '+' : '−'}
                    {e.amount}
                  </text>
                </g>
              )}
            </g>
          );
        })}
    </svg>
  );
}
