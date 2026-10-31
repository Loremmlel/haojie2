type Game = {
  difficulty: string;
  pair: number;
  seed: number;
  learner: number;
  winner?: number | 'draw' | null;
  steps?: number;
  error?: string;
};
type Score = {
  pairs: number;
  wins: number;
  losses: number;
  draws: number;
  unknown: number;
  errors: number;
  win_rate?: number | null;
  score_rate?: number | null;
  score_interval?: [number, number];
  verdict?: string;
};
export type EvaluationStatus = {
  state: string;
  version?: number;
  started_at: number;
  completed?: number;
  total?: number;
  current?: Game | null;
  results?: Game[];
  summary?: Record<string, Score>;
  error?: string;
  opponents?: number[];
  games?: number;
  simulations?: number;
  depth?: number;
  pending?: number;
};
const difficulty: Record<string, string> = { easy: '简单', medium: '普通', hard: '困难' };
const states: Record<string, string> = {
  queued: '等待测评',
  starting: '正在启动',
  running: '对弈中',
  completed: '本轮结束',
  failed: '评测异常',
};
const verdicts: Record<string, string> = {
  ahead: '本轮领先',
  behind: '本轮落后',
  uncertain: '差异未明',
  incomplete: '样本未齐',
};
const percent = (value?: number | null) => (value == null ? '—' : `${(value * 100).toFixed(1)}%`);
const outcome = (game: Game) =>
  game.error
    ? '失败'
    : game.winner == null
      ? '未知'
      : game.winner === 'draw'
        ? '平'
        : game.winner === game.learner
          ? '胜'
          : '负';

export function Evaluation({
  status,
  interval,
  kind = 'manual',
  games = 32,
}: {
  status?: EvaluationStatus | null;
  interval: number;
  kind?: 'manual' | 'history';
  games?: number;
}) {
  const history = kind === 'history';
  const id = `${kind}-evaluation-title`;
  const opponents = history
    ? Object.fromEntries(
        (status?.opponents ?? []).map((v, i, values) => [
          String(v),
          `v${v} · ${i === 0 ? (values.length === 1 ? '基线 / 上一版' : '基线') : '上一版'}`,
        ]),
      )
    : difficulty;
  const label = (key: string) => opponents[key] ?? `v${key}`;
  return (
    <section className="evaluation" aria-labelledby={id}>
      <div className="evaluation-heading">
        <h2 id={id}>{history ? '历史模型对弈' : '手工 AI 对弈'}</h2>
        <span>
          {status
            ? states[status.state] || status.state
            : (history ? games : interval)
              ? '等待首轮'
              : '已关闭'}
        </span>
      </div>
      <p>
        {history
          ? `每次历史归档对比基线与上一版 · 每个对手 ${status?.games ?? games} 局`
          : interval
            ? `自对弈阶段每 ${Number((interval / 60).toFixed(2))} 分钟一轮`
            : '不再启动新一轮'}{' '}
        {history ? ' · 按归档顺序逐轮执行' : ' · 三档各 4 局 · 独立运行，上一轮结束后才启动下一轮'}
      </p>
      {status ? (
        <>
          <p className="evaluation-progress" role="status">
            冻结版本 {status.version ?? '—'} · {status.completed ?? 0} /{' '}
            {status.total ?? (history ? 0 : 12)} 局{' · '}
            {new Date(status.started_at * 1000).toLocaleString('zh-CN')}
            {status.current && status.state === 'running' && (
              <>
                <br />
                当前：{label(status.current.difficulty)} · 第 {status.current.pair + 1} 对 · 模型
                {status.current.learner === 1 ? '先手' : '后手'} · {status.current.steps ?? 0} 步
              </>
            )}
          </p>
          {status.error && <p role="alert">{status.error}</p>}
          {history && (
            <p>
              双方 MCTS {status.simulations ?? '—'} 次 / 深度 {status.depth ?? '—'} · 排队{' '}
              {status.pending ?? 0} 轮
            </p>
          )}
          <p>胜负只统计完整镜像对；超限保留未知，错误单独列出。</p>
          <div
            className="evaluation-table"
            tabIndex={0}
            role="region"
            aria-label={history ? '本轮历史模型结果' : '本轮各难度结果'}
          >
            <table>
              <thead>
                <tr>
                  <th scope="col">{history ? '对手' : '难度'}</th>
                  <th scope="col">完整对</th>
                  <th scope="col">胜 / 负 / 平</th>
                  <th scope="col">未知 / 失败</th>
                  {history && (
                    <>
                      <th scope="col">胜率 / 得分率</th>
                      <th scope="col">得分率 95% 区间</th>
                      <th scope="col">进步判断</th>
                    </>
                  )}
                </tr>
              </thead>
              <tbody>
                {Object.entries(opponents).map(([key, label]) => {
                  const score = status.summary?.[key];
                  return (
                    <tr key={key}>
                      <th scope="row">{label}</th>
                      <td>
                        {score?.pairs ?? 0} / {history ? (status.games ?? games) / 2 : 2}
                      </td>
                      <td>
                        {score?.wins ?? 0} / {score?.losses ?? 0} / {score?.draws ?? 0}
                      </td>
                      <td>
                        {score?.unknown ?? 0} / {score?.errors ?? 0}
                      </td>
                      {history && (
                        <>
                          <td>
                            {percent(score?.win_rate)} / {percent(score?.score_rate)}
                          </td>
                          <td>
                            {score?.score_interval
                              ? score.score_interval.map(percent).join(' – ')
                              : '—'}
                          </td>
                          <td>{verdicts[score?.verdict ?? 'incomplete']}</td>
                        </>
                      )}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {history && <p>平局计半分；区间按独立镜像对计算，仅完整一轮给出领先或落后判断。</p>}
          {!!status.results?.length && (
            <details className="evaluation-games">
              <summary>逐局记录与镜像种子</summary>
              <ul>
                {status.results.map((game) => (
                  <li key={`${game.difficulty}-${game.pair}-${game.learner}`}>
                    <strong>
                      {label(game.difficulty)} · 第 {game.pair + 1} 对 ·{' '}
                      {game.learner === 1 ? '先手' : '后手'}：{outcome(game)}
                    </strong>
                    <span>
                      种子 {game.seed} · {game.steps ?? '—'} 步{game.error && ` · ${game.error}`}
                    </span>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </>
      ) : (
        <p>
          {history
            ? '首个训练后版本保存为基线，下一次历史归档后开始对比。'
            : '启动训练并进入自对弈后开始计时。'}
          每轮冻结权重，镜像对共享种子并交换先后手。
        </p>
      )}
      <p>停止训练后，已开始的评测继续完成。评测共用本机 CPU / GPU。</p>
      {history && (
        <p>尚未开始的轮次在继续训练后依次执行。完整记录保存在训练目录 history-evaluations。</p>
      )}
    </section>
  );
}
