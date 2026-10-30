import { useEffect, useRef, useState } from "react";
import { Chart, number, type Point, type Series } from "./charts";

type Config = Record<string, number | string>;
type Field = {
  key: string;
  label: string;
  group: string;
  min: number;
  max: number;
  step: number;
  help: string;
  integer: boolean;
  initial: boolean;
};
type Schema = { defaults: Config; fields: Field[] };
type Snapshot = {
  run: string | null;
  state: string;
  message: string;
  revision: number;
  session: string;
  config: Config;
  stats: Record<string, any>;
  running: boolean;
  checkpoint: boolean;
  history: Point[];
  sequence: number;
  logs: string[];
  runs: { name: string; updated: number }[];
  cpu_count: number;
  memory_gib: number;
};
const names: Record<string, string> = {
  idle: "待开始",
  starting: "正在启动",
  running: "训练中",
  stopping: "保存中",
  restarting: "应用参数中",
  saved: "已保存",
  error: "运行异常",
};
const loss: Series[] = [
  { key: "policy", label: "策略", color: "var(--teal)" },
  { key: "value", label: "价值", color: "var(--ember)" },
  { key: "aux", label: "辅助", color: "var(--gold)" },
];
const throughput: Series[] = [
  { key: "cps", label: "命令", color: "var(--teal)" },
  { key: "sps", label: "训练样本", color: "var(--ember)" },
];
const utilization: Series[] = [
  { key: "cpu", label: "训练 CPU", color: "var(--teal)" },
  { key: "gpu", label: "整卡 GPU", color: "var(--ember)" },
];
const memory: Series[] = [
  { key: "ram", label: "训练 RAM", color: "var(--teal)" },
  { key: "vram", label: "整卡显存", color: "var(--ember)" },
  { key: "replay", label: "回放", color: "var(--gold)" },
];
const freshName = () =>
  `训练-${new Date().toLocaleString("sv-SE").replace(/[-: ]/g, "").slice(0, 14)}`;

async function api(path: string, body?: unknown) {
  const response = await fetch(path, {
    ...(body
      ? {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Haojie-Control": "1",
          },
          body: JSON.stringify(body),
        }
      : {}),
    signal: AbortSignal.timeout(10000),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "服务请求失败");
  return data;
}

export default function App() {
  const [schema, setSchema] = useState<Schema>();
  const [state, setState] = useState<Snapshot>();
  const [points, setPoints] = useState<Point[]>([]);
  const [draft, setDraft] = useState<Config>({});
  const [creating, setCreating] = useState(true);
  const [name, setName] = useState(freshName);
  const [busy, setBusy] = useState(false);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState("");
  const [range, setRange] = useState(3600);
  const [smoothing, setSmoothing] = useState(60);
  const [raw, setRaw] = useState(false);
  const latest = useRef<Snapshot | undefined>(undefined);
  const initialized = useRef(false);

  const accept = (data: Snapshot) => {
    if (
      latest.current?.session === data.session &&
      data.revision < latest.current.revision
    )
      return;
    const previous = latest.current;
    latest.current = data;
    if (previous?.run && !data.run) {
      setCreating(true);
      setName(freshName());
    }
    setState(data);
    setPoints((old) =>
      (previous?.session !== data.session || previous?.run !== data.run
        ? data.history
        : [
            ...old,
            ...data.history.filter((p) => p.id > (previous?.sequence || 0)),
          ]
      ).slice(-17280),
    );
    setConnected(true);
    if (!initialized.current) {
      initialized.current = true;
      setCreating(!data.run);
      let saved: {
        run?: string;
        draft?: Config;
        creating?: boolean;
        name?: string;
      } = {};
      try {
        saved = JSON.parse(
          sessionStorage.getItem("haojie-training-draft") || "{}",
        );
      } catch {
        /* 损坏草稿不影响服务端训练。 */
      }
      setDraft(
        saved.run === (data.run || "") && saved.draft
          ? saved.draft
          : data.config,
      );
      if (saved.run === (data.run || "") && saved.creating != null) {
        setCreating(saved.creating);
        if (saved.name) setName(saved.name);
      }
    }
  };
  useEffect(() => {
    let cancelled = false,
      timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const [data, definitions] = await Promise.all([
          api(`/api/state?since=${latest.current?.sequence || 0}`),
          schema ? Promise.resolve(null) : api("/api/schema"),
        ]);
        if (!cancelled) {
          accept(data);
          if (definitions) setSchema(definitions);
        }
      } catch {
        if (!cancelled) setConnected(false);
      }
      if (!cancelled) timer = setTimeout(poll, 2000);
    };
    void poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [!!schema]);
  useEffect(() => {
    if (initialized.current) {
      try {
        sessionStorage.setItem(
          "haojie-training-draft",
          JSON.stringify({ run: state?.run || "", draft, creating, name }),
        );
      } catch {
        /* 页面存储不可用时仍可操作服务。 */
      }
    }
  }, [draft, creating, name, state?.run]);

  const pending =
    busy || ["starting", "stopping", "restarting"].includes(state?.state || "");
  const baseline = creating ? schema?.defaults : state?.config;
  const changed =
    baseline && Object.keys(draft).some((k) => draft[k] !== baseline[k]);
  const act = async (action: string, selected?: string) => {
    setError("");
    if (["start", "resume", "apply"].includes(action) && schema) {
      const invalid = schema.fields.find(
        (f) =>
          typeof draft[f.key] !== "number" ||
          !Number.isFinite(Number(draft[f.key])) ||
          Number(draft[f.key]) < f.min ||
          Number(draft[f.key]) > f.max ||
          (f.integer && !Number.isInteger(Number(draft[f.key]))),
      );
      if (invalid) {
        setError(
          `${invalid.label}须在 ${invalid.min}–${invalid.max} 之间${invalid.integer ? "且为整数" : ""}`,
        );
        document.getElementById(invalid.key)?.focus();
        return;
      }
    }
    setBusy(true);
    try {
      const data = await api("/api/control", {
        action,
        name: selected || name,
        ...(action !== "stop" && action !== "select" ? { config: draft } : {}),
        revision: latest.current?.revision ?? 0,
        session: latest.current?.session,
      });
      accept(data);
      if (action !== "stop") {
        if (action !== "apply") setDraft(data.config);
        setCreating(false);
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : "操作失败，请重试";
      setError(message);
      const field = schema?.fields.find((f) => message.startsWith(f.label));
      if (field) document.getElementById(field.key)?.focus();
    } finally {
      setBusy(false);
    }
  };
  const activePoints = points.filter(
    (p) => p.t >= (points.at(-1)?.t || Date.now() / 1000) - range,
  );
  const recent = points.at(-1);
  const stat = state?.stats || {};
  const groups = [...new Set(schema?.fields.map((f) => f.group))];
  const stage = stat.phase || "collect";

  return (
    <div className="console-shell">
      <header className="topbar">
        <a href="/" className="brand" aria-label="浩劫训练台首页">
          <span className="brand-mark" aria-hidden="true">
            劫
          </span>
          <span>
            浩劫<small>训练台</small>
          </span>
        </a>
        <div className="run-selector">
          <label htmlFor="run-select">当前训练</label>
          <select
            id="run-select"
            value={creating ? "" : state?.run || ""}
            disabled={!connected || pending || state?.running}
            onChange={(e) => {
              if (changed) {
                setError("有未应用的参数修改，请先应用或撤销修改再切换训练。");
                return;
              }
              if (!e.target.value) {
                setCreating(true);
                setDraft(schema?.defaults || {});
                setName(freshName());
              } else void act("select", e.target.value);
            }}
          >
            <option value="">新建训练</option>
            {state?.runs.map((r) => (
              <option key={r.name} value={r.name}>
                {r.name}
              </option>
            ))}
          </select>
        </div>
        <span className={`state-pill ${state?.state || "idle"}`}>
          <i />
          {connected
            ? names[state?.state || "idle"]
            : state
              ? "连接中断"
              : "正在连接"}
        </span>
        <div className="main-actions">
          {state?.running ? (
            <button
              className="button outline"
              type="button"
              disabled={!connected || pending}
              onClick={() => void act("stop")}
            >
              <span aria-hidden="true">Ⅱ</span> 保存并停止
            </button>
          ) : (
            <button
              className="button primary"
              form="parameters"
              type="submit"
              disabled={
                !connected ||
                pending ||
                !schema ||
                (!creating && !state?.checkpoint)
              }
            >
              <span aria-hidden="true">▶</span>{" "}
              {creating ? "开始训练" : "继续训练"}
            </button>
          )}
        </div>
      </header>
      <div className="page-heading">
        <div>
          <div className="eyebrow">普通局 · 5.50M · 原生 BF16</div>
          <h1>看见每一步学习。</h1>
          <p>观测训练进度，让算力落在合适的位置。</p>
        </div>
        <div className="machine">
          <span>
            {state?.cpu_count || "—"} 逻辑核心 / {state?.memory_gib || "—"} GiB
          </span>
          <small>单训练进程 · Rust 线程池 · PyTorch</small>
        </div>
      </div>
      <div
        className={`feedback ${error || (!connected && state) ? "failure" : ""}`}
        role={error ? "alert" : "status"}
        aria-live="polite"
      >
        {error ||
          (!connected && state
            ? "暂时无法连接控制服务。保留当前读数，恢复连接后可继续操作。"
            : state?.message || "正在连接本地训练服务…")}
      </div>
      <div className="workspace">
        <main className="monitor">
          <div className="stage-track" aria-label="训练阶段">
            {[
              [
                "collect",
                "示范采集",
                `${number(stat.demonstrations, 0)} / ${state?.config.warmup_games ?? 8} 局`,
              ],
              [
                "imitation",
                "模仿学习",
                `${number(stat.imitation_updates, 0)} / ${state?.config.warmup_updates ?? 512} 次`,
              ],
              ["self", "自对弈", "当前 / 历史 / 手工对手"],
            ].map(([key, label, hint], i) => (
              <div key={key} className={stage === key ? "current" : ""}>
                <span className="stage-symbol" aria-hidden="true">
                  {i === 0 ? "◇" : i === 1 ? "◈" : "↔"}
                </span>
                <span>
                  {label}
                  <small>{hint}</small>
                </span>
                {i < 2 && <b aria-hidden="true">→</b>}
              </div>
            ))}
          </div>
          <div className="metrics">
            <Metric
              label="累计更新"
              value={number(stat.updates, 0)}
              unit="次"
              hint={`${number(stat.trained_samples, 0)} 个样本已消费`}
            />
            <Metric
              label="自然终局"
              value={number(stat.completed, 0)}
              unit="局"
              hint={`${number(stat.truncated, 0)} 局截断，保持未知`}
            />
            <Metric
              label="命令吞吐"
              value={number(recent?.cps as number)}
              unit="/ 秒"
              hint="相邻真实快照计数差"
            />
            <Metric
              label="样本吞吐"
              value={number(recent?.sps as number)}
              unit="/ 秒"
              hint="实际训练消费速度"
            />
          </div>
          <div className="chart-controls">
            <h2>
              训练趋势 <small>服务内存 · 最多 24 小时</small>
            </h2>
            <div>
              <label>
                范围
                <select
                  aria-label="图表时间范围"
                  value={range}
                  onChange={(e) => setRange(+e.target.value)}
                >
                  <option value={600}>10 分钟</option>
                  <option value={3600}>1 小时</option>
                  <option value={21600}>6 小时</option>
                  <option value={86400}>24 小时</option>
                </select>
              </label>
              <label>
                平滑
                <select
                  aria-label="曲线平滑时长"
                  value={smoothing}
                  onChange={(e) => setSmoothing(+e.target.value)}
                >
                  {[15, 30, 60, 120].map((n) => (
                    <option key={n} value={n}>
                      {n} 秒
                    </option>
                  ))}
                </select>
              </label>
              <label className="check-label">
                <input
                  type="checkbox"
                  checked={raw}
                  onChange={(e) => setRaw(e.target.checked)}
                />
                原始线
              </label>
            </div>
          </div>
          <div className="charts">
            <Chart
              title="学习损失"
              subtitle="按真实更新采样 · 时间加权 EMA"
              points={activePoints}
              series={loss}
              seconds={smoothing}
              raw={raw}
            />
            <Chart
              title="采集与学习"
              subtitle="消费速度与决策速度"
              unit="每秒"
              points={activePoints}
              series={throughput}
              seconds={smoothing}
              raw={raw}
            />
            <Chart
              title="计算资源"
              subtitle="CPU 为训练进程 / 整机容量，GPU 为整卡"
              unit="%"
              ceiling={100}
              points={activePoints}
              series={utilization}
              seconds={smoothing}
              raw={raw}
            />
            <Chart
              title="内存与显存"
              subtitle="回放内存随窗口自动回收"
              unit="GiB"
              points={activePoints}
              series={memory}
              seconds={smoothing}
              raw={raw}
            />
          </div>
          <details className="run-log">
            <summary>
              运行记录 <span>{state?.logs.length || 0} 条</span>
            </summary>
            <pre>{state?.logs.join("\n") || "暂无异常或额外运行消息。"}</pre>
          </details>
          <p className="lifecycle-note">
            刷新或关闭页面不停止训练。服务重启后从最近恢复点继续，曲线重新采集。
          </p>
        </main>
        <aside className="parameters">
          <div className="parameter-title">
            <h2>训练参数</h2>
            <span>{state?.running ? "保存后应用" : "启动配置"}</span>
          </div>
          <div className="concurrency">
            <span>并行对局</span>
            <strong>
              {Number(draft.workers || 0) * Number(draft.games_per_worker || 0)}
              <small>目标</small>
            </strong>
            <p>
              {number(stat.active_games, 0)} 局在途 ·{" "}
              {number(stat.queued_games, 0)} 局待恢复
            </p>
            <div>
              {draft.workers || "—"} 线程 <b>×</b>{" "}
              {draft.games_per_worker || "—"} 局 / 线程
            </div>
          </div>
          <form
            id="parameters"
            noValidate
            onSubmit={(e) => {
              e.preventDefault();
              void act(
                state?.running ? "apply" : creating ? "start" : "resume",
              );
            }}
          >
            {creating && (
              <label className="field new-name">
                训练名称
                <input
                  id="run-name"
                  value={name}
                  maxLength={64}
                  onChange={(e) => setName(e.target.value)}
                  autoComplete="off"
                />
                <small>新建独立目录，保存点不会覆盖其他训练。</small>
              </label>
            )}
            {groups.map((group, index) => (
              <details
                key={group}
                open={index === 0}
                className="parameter-group"
              >
                <summary>
                  {group}
                  <span aria-hidden="true">＋</span>
                </summary>
                <div className="fields">
                  {schema?.fields
                    .filter((f) => f.group === group)
                    .map((f) => (
                      <label className="field" key={f.key} htmlFor={f.key}>
                        {f.label}
                        <input
                          id={f.key}
                          type="number"
                          min={f.min}
                          max={f.max}
                          step={f.step}
                          value={draft[f.key] ?? ""}
                          disabled={pending || (!creating && f.initial)}
                          aria-describedby={`${f.key}-help`}
                          aria-invalid={error.startsWith(f.label) || undefined}
                          onChange={(e) =>
                            setDraft((v) => ({
                              ...v,
                              [f.key]:
                                e.target.value === ""
                                  ? ""
                                  : Number(e.target.value),
                            }))
                          }
                        />
                        <small id={`${f.key}-help`}>{f.help}</small>
                      </label>
                    ))}
                </div>
              </details>
            ))}
            <div className="parameter-footer">
              <p>
                {state?.running
                  ? "应用时保存并短暂重启。原有对局、回放和优化器状态继续保留。"
                  : "配置从同一训练引擎读取，精度固定为原生 BF16。"}
              </p>
              {state?.running && (
                <button
                  className="button primary"
                  type="submit"
                  disabled={!connected || pending || !changed}
                >
                  保存并应用参数
                </button>
              )}
              <button
                className="button ghost"
                type="button"
                disabled={pending || !schema}
                onClick={() => {
                  setDraft(creating ? schema!.defaults : state!.config);
                  setError("");
                }}
              >
                撤销未应用修改
              </button>
            </div>
          </form>
        </aside>
      </div>
      <footer className="footer">
        <span>浩劫 / TRAINING STUDIO</span>
        <span>内存回放 · 历史权重归档 · 单个滚动恢复点</span>
      </footer>
    </div>
  );
}

function Metric({
  label,
  value,
  unit,
  hint,
}: {
  label: string;
  value: string;
  unit: string;
  hint: string;
}) {
  return (
    <div className="metric">
      <span>{label}</span>
      <div>
        <strong>{value}</strong>
        <small>{unit}</small>
      </div>
      <p>{hint}</p>
    </div>
  );
}
