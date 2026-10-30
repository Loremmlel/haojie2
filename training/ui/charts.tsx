import { useEffect, useMemo, useRef, useState } from 'react';

export type Point = {
  id: number;
  t: number;
  gap?: boolean;
  [key: string]: number | boolean | null | undefined;
};
export type Series = { key: string; label: string; color: string };
export type Sample = { t: number; raw: number; value: number } | null;

/** 按真实时间间隔平滑；缺测和运行边界不连接。 */
export function smooth(points: Point[], key: string, seconds: number): Sample[] {
  const result: Sample[] = [];
  let last: { t: number; value: number } | undefined;
  for (const p of points) {
    if (p.gap) {
      result.push(null);
      last = undefined;
      continue;
    }
    const raw = p[key];
    if (typeof raw !== 'number' || !Number.isFinite(raw)) continue;
    if (last && p.t - last.t > 15) {
      result.push(null);
      last = undefined;
    }
    const alpha = !last || seconds === 0 ? 1 : 1 - Math.exp(-(p.t - last.t) / seconds);
    const value = last ? last.value + alpha * (raw - last.value) : raw;
    result.push({ t: p.t, raw, value });
    last = { t: p.t, value };
  }
  return result;
}

/** 分段三次 Hermite 曲线；相邻区间同号时限制斜率，避免制造峰值。 */
export function curve(points: [number, number][]): string {
  if (!points.length) return '';
  if (points.length === 1) return `M${points[0].join(',')}h0.01`;
  const slopes = points.slice(1).map((p, i) => (p[1] - points[i][1]) / (p[0] - points[i][0] || 1));
  const tangents = points.map((_, i) =>
    i === 0
      ? slopes[0]
      : i === points.length - 1
        ? slopes[i - 1]
        : slopes[i - 1] * slopes[i] <= 0
          ? 0
          : (2 * slopes[i - 1] * slopes[i]) / (slopes[i - 1] + slopes[i]),
  );
  return points.slice(1).reduce(
    (path, p, i) => {
      const a = points[i],
        dx = (p[0] - a[0]) / 3;
      return `${path}C${a[0] + dx},${a[1] + dx * tangents[i]} ${p[0] - dx},${p[1] - dx * tangents[i + 1]} ${p[0]},${p[1]}`;
    },
    `M${points[0].join(',')}`,
  );
}

export const number = (value: number | undefined | null, digits = 1) =>
  value == null || !Number.isFinite(value)
    ? '—'
    : new Intl.NumberFormat('zh-CN', { maximumFractionDigits: digits }).format(value);
const clock = (t: number, seconds = false) =>
  new Date(t * 1000).toLocaleTimeString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit',
    ...(seconds ? { second: '2-digit' as const } : {}),
  });

export function Chart({
  title,
  subtitle,
  points,
  series,
  seconds,
  raw,
  unit = '',
  ceiling,
}: {
  title: string;
  subtitle: string;
  points: Point[];
  series: Series[];
  seconds: number;
  raw: boolean;
  unit?: string;
  ceiling?: number;
}) {
  const container = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(600);
  const [cursor, setCursor] = useState<number | null>(null);
  const [hidden, setHidden] = useState<string[]>([]);
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) =>
      setWidth(Math.max(260, entry.contentRect.width)),
    );
    if (container.current) observer.observe(container.current);
    return () => observer.disconnect();
  }, []);
  const data = useMemo(
    () => series.map((s) => ({ ...s, samples: smooth(points, s.key, seconds) })),
    [points, series, seconds],
  );
  const visible = data.filter((s) => !hidden.includes(s.key));
  const values = visible.flatMap((s) =>
    s.samples.flatMap((p) => (p ? [p.value, ...(raw ? [p.raw] : [])] : [])),
  );
  const magnitude = Math.max(...values, 0);
  const target = (magnitude || 1) * 1.1;
  const step = 10 ** Math.floor(Math.log10(target)) / 2;
  const max = ceiling || (magnitude ? Math.ceil(target / step) * step : 1);
  const start = points[0]?.t || 0,
    end = Math.max(points.at(-1)?.t || 0, start + 1);
  const height = 220,
    left = 44,
    right = width - 12,
    bottom = height - 28;
  const x = (t: number) => left + ((t - start) / (end - start)) * (right - left);
  const y = (v: number) => bottom - (Math.max(0, v) / max) * (bottom - 16);
  const paths = (samples: Sample[], original: boolean) => {
    const groups: [number, number][][] = [[]];
    const stride = Math.max(1, Math.ceil(samples.length / 600));
    samples.forEach((p, i) => {
      if (!p) {
        if (groups.at(-1)!.length) groups.push([]);
      } else if (i % stride === 0 || i === samples.length - 1)
        groups.at(-1)!.push([x(p.t), y(original ? p.raw : p.value)]);
    });
    return groups
      .filter((g) => g.length)
      .map((g) => (original ? `M${g.map((p) => p.join(',')).join('L')}` : curve(g)));
  };
  const activeTime = cursor == null ? null : start + cursor * (end - start);
  return (
    <section className="chart-panel" aria-label={title}>
      <div className="chart-heading">
        <div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
        <span className="unit">{unit}</span>
      </div>
      <div className="legend">
        {data.map((s) => (
          <button
            key={s.key}
            type="button"
            aria-pressed={!hidden.includes(s.key)}
            onClick={() =>
              setHidden((v) => (v.includes(s.key) ? v.filter((k) => k !== s.key) : [...v, s.key]))
            }
          >
            <i style={{ background: s.color }} />
            {s.label}
            <strong>{number(s.samples.filter((p) => p !== null).at(-1)?.value, 3)}</strong>
          </button>
        ))}
      </div>
      <div
        className="plot"
        ref={container}
        tabIndex={0}
        role="group"
        aria-label={`${title}，左右方向键查看观测，Escape 退出`}
        onPointerMove={(e) => {
          const box = e.currentTarget.getBoundingClientRect();
          setCursor(Math.max(0, Math.min(1, (e.clientX - box.left - left) / (right - left))));
        }}
        onPointerLeave={() => setCursor(null)}
        onBlur={() => setCursor(null)}
        onKeyDown={(e) => {
          if (e.key === 'Escape') setCursor(null);
          if (['ArrowLeft', 'ArrowRight'].includes(e.key)) {
            e.preventDefault();
            setCursor((v) =>
              Math.max(0, Math.min(1, (v ?? 1) + (e.key === 'ArrowLeft' ? -0.025 : 0.025))),
            );
          }
        }}
      >
        <svg viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
          {[0, 1, 2, 3, 4].map((i) => (
            <g key={i}>
              <line
                x1={left}
                x2={right}
                y1={y((max * i) / 4)}
                y2={y((max * i) / 4)}
                className="grid-line"
              />
              <text x={left - 10} y={y((max * i) / 4) + 4} textAnchor="end">
                {number((max * i) / 4, max < 1 ? 4 : 2)}
              </text>
            </g>
          ))}
          {[0, 0.5, 1].map((f) => (
            <text
              key={f}
              x={x(start + f * (end - start))}
              y={height - 4}
              textAnchor={f === 0 ? 'start' : f === 1 ? 'end' : 'middle'}
            >
              {points.length ? clock(start + f * (end - start), end - start < 120) : '—'}
            </text>
          ))}
          {visible.map((s) => (
            <g key={s.key}>
              {raw &&
                paths(s.samples, true).map((d, i) => (
                  <path key={`r${i}`} d={d} stroke={s.color} className="raw-line" />
                ))}
              {paths(s.samples, false).map((d, i) => (
                <path key={i} d={d} stroke={s.color} className="trend-line" />
              ))}
            </g>
          ))}
          {activeTime != null && (
            <line
              x1={x(activeTime)}
              x2={x(activeTime)}
              y1={12}
              y2={bottom}
              className="cursor-line"
            />
          )}
        </svg>
        {!values.length && (
          <div className="chart-empty">
            {points.length ? '该阶段尚无有效观测' : '开始训练后，曲线将在这里生长'}
            <small>仅展示真实采样</small>
          </div>
        )}
        {activeTime != null && values.length > 0 && (
          <div
            className="chart-tooltip"
            style={{ left: Math.max(8, Math.min(width - 195, x(activeTime) + 12)) }}
          >
            <strong>{clock(activeTime, true)} · 原始 / 趋势</strong>
            {visible.map((s) => {
              const nearest = s.samples
                .filter((p) => p != null)
                .reduce<
                  Exclude<Sample, null> | undefined
                >((best, p) => (!best || Math.abs(p.t - activeTime) < Math.abs(best.t - activeTime) ? p : best), undefined);
              return (
                <div key={s.key}>
                  {s.label}
                  <b>
                    {nearest && Math.abs(nearest.t - activeTime) <= 15
                      ? `${number(nearest.raw, 3)} / ${number(nearest.value, 3)}`
                      : '—'}
                  </b>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}
