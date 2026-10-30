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
};
const difficulty: Record<string, string> = { easy: '简单', medium: '普通', hard: '困难' };
const states: Record<string, string> = {
  starting: '正在启动',
  running: '对弈中',
  completed: '本轮结束',
  failed: '评测异常',
};
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
}: {
  status?: EvaluationStatus | null;
  interval: number;
}) {
  return (
    <section className="evaluation" aria-labelledby="evaluation-title">
      <div className="evaluation-heading">
        <h2 id="evaluation-title">手工 AI 对弈</h2>
        <span>
          {status ? states[status.state] || status.state : interval ? '等待首轮' : '定时已关闭'}
        </span>
      </div>
      <p>
        {interval
          ? `自对弈阶段每 ${Number((interval / 60).toFixed(2))} 分钟一轮`
          : '不再启动新一轮'}{' '}
        · 三档各 4 局 · 独立运行，上一轮结束后才启动下一轮
      </p>
      {status ? (
        <>
          <p className="evaluation-progress" role="status">
            冻结版本 {status.version ?? '—'} · {status.completed ?? 0} / {status.total ?? 12} 局
            {' · '}
            {new Date(status.started_at * 1000).toLocaleString('zh-CN')}
            {status.current && status.state === 'running' && (
              <>
                <br />
                当前：{difficulty[status.current.difficulty]} · 第 {status.current.pair + 1} 对 ·
                模型{status.current.learner === 1 ? '先手' : '后手'} · {status.current.steps ?? 0}{' '}
                步
              </>
            )}
          </p>
          {status.error && <p role="alert">{status.error}</p>}
          <p>胜负只统计完整镜像对；超限保留未知，错误单独列出。</p>
          <div className="evaluation-table" tabIndex={0} role="region" aria-label="本轮各难度结果">
            <table>
              <thead>
                <tr>
                  <th scope="col">难度</th>
                  <th scope="col">完整对</th>
                  <th scope="col">胜 / 负 / 平</th>
                  <th scope="col">未知 / 失败</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(difficulty).map(([key, label]) => {
                  const score = status.summary?.[key];
                  return (
                    <tr key={key}>
                      <th scope="row">{label}</th>
                      <td>{score?.pairs ?? 0} / 2</td>
                      <td>
                        {score?.wins ?? 0} / {score?.losses ?? 0} / {score?.draws ?? 0}
                      </td>
                      <td>
                        {score?.unknown ?? 0} / {score?.errors ?? 0}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {!!status.results?.length && (
            <details className="evaluation-games">
              <summary>逐局记录与镜像种子</summary>
              <ul>
                {status.results.map((game) => (
                  <li key={`${game.difficulty}-${game.pair}-${game.learner}`}>
                    <strong>
                      {difficulty[game.difficulty]} · 第 {game.pair + 1} 对 ·{' '}
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
        <p>启动训练并进入自对弈后开始计时。每轮固定同一模型，镜像对共享种子并交换先后手。</p>
      )}
      <p>停止训练后，已开始的评测继续完成。评测共用本机 CPU / GPU。</p>
    </section>
  );
}
