const $ = (id) => document.getElementById(id);
const states = {
  idle: '空闲',
  starting: '启动中',
  sampling: '采样中',
  updating: '更新中',
  evaluating: '评测中',
  pausing: '暂停中',
  paused: '已暂停',
  error: '错误 / 已停算',
};
const names = {
  easy: '简单',
  medium: '中等',
  hard: '困难',
  baseline: '初始化基准',
};
const outcomes = {
  win: '胜',
  loss: '负',
  draw: '规则平局',
  unfinished: '未完成',
  error: '错误',
};
const reasons = {
  terminal: '真实终局',
  commands: '命令超限',
  plies: '回合超限',
  cancelled: '已取消',
  'decode-budget': '解码超限',
  error: '错误',
  'time-budget': '单局时间预算耗尽',
  'resource-budget': '推理帧内存预算不足',
};
const number = (n) => new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 2 }).format(n ?? 0);
const bytes = (n) => `${number((n ?? 0) / 2 ** 20)} MiB`;
const percent = (n) => (n == null ? '未评测' : `${number(n * 100)}%`);
const date = (n) => (n ? new Date(n * 1000).toLocaleString('zh-CN', { hour12: false }) : '暂无');
let token,
  busy = false,
  connected = false,
  current,
  dirty = false;
const text = (id, value) => {
  $(id).textContent = value;
};

function showPage(name) {
  const pages = [...document.querySelectorAll('[data-page]')];
  if (!pages.some((page) => page.dataset.page === name)) name = 'overview';
  for (const page of pages) page.hidden = page.dataset.page !== name;
  document.querySelectorAll('[data-page-target]').forEach((button) => {
    if (button.dataset.pageTarget === name) button.setAttribute('aria-current', 'page');
    else button.removeAttribute('aria-current');
  });
  $('workspace').scrollTop = 0;
}
window.addEventListener('hashchange', () => showPage(location.hash.slice(1)));
showPage(location.hash.slice(1));

function chart(id, lines, percentAxis = false) {
  const root = $(id);
  root.replaceChildren();
  if (!lines.some((line) => line.points.length)) {
    root.textContent = '暂无数据 · 等待实际完成结果';
    return;
  }
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', '0 0 500 175');
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', root.getAttribute('aria-label'));
  const all = lines.flatMap((line) => line.points),
    maxX = Math.max(1, ...all.map((p) => p.x));
  const maxY = percentAxis ? 1 : Math.max(0.01, ...all.map((p) => p.y));
  for (const line of lines) {
    const path = document.createElementNS(ns, 'polyline');
    path.setAttribute(
      'points',
      line.points.map((p) => `${25 + (p.x / maxX) * 465},${150 - (p.y / maxY) * 130}`).join(' '),
    );
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', line.color);
    path.setAttribute('stroke-width', '2');
    svg.append(path);
    for (const p of line.points) {
      const dot = document.createElementNS(ns, 'circle');
      dot.setAttribute('cx', 25 + (p.x / maxX) * 465);
      dot.setAttribute('cy', 150 - (p.y / maxY) * 130);
      dot.setAttribute('r', '2.5');
      dot.setAttribute('fill', line.color);
      const title = document.createElementNS(ns, 'title');
      title.textContent = `${line.name} · 更新 ${p.x} · ${percentAxis ? percent(p.y) : number(p.y)}`;
      dot.append(title);
      svg.append(dot);
    }
  }
  for (const [x, y, value] of [
    [2, 14, percentAxis ? '100%' : number(maxY)],
    [2, 154, '0'],
    [430, 172, `更新 ${maxX}`],
  ]) {
    const label = document.createElementNS(ns, 'text');
    label.setAttribute('x', x);
    label.setAttribute('y', y);
    label.setAttribute('font-size', '10');
    label.setAttribute('fill', 'currentColor');
    label.textContent = value;
    svg.append(label);
  }
  root.append(svg);
}

function controls() {
  const active =
    current &&
    ['starting', 'sampling', 'updating', 'evaluating', 'pausing'].includes(current.state);
  document.querySelectorAll('[data-action]').forEach((button) => {
    const action = button.dataset.action;
    button.disabled =
      busy ||
      !connected ||
      !current ||
      (action === 'start' && active) ||
      (action === 'pause' && !active) ||
      (action === 'evaluate' &&
        (!current.evaluation_ready || current.eval_pending || current.evaluation_job)) ||
      (action === 'cancel-evaluation' && !current.evaluation_job && !current.eval_pending) ||
      (action === 'save' && current.save_pending);
    if (action === 'confirm-candidate')
      button.disabled ||= !current?.candidate || !!current?.evaluation_job;
    if (action === 'export') button.disabled ||= !current?.version;
    if (action === 'cleanup') button.disabled ||= active;
    if (action === 'restore') button.disabled ||= active || !current?.recoveries.length;
    if (action === 'switch-experiment' || action === 'delete-experiment')
      button.disabled ||=
        active ||
        !$('experiment-choice').value ||
        $('experiment-choice').value === current?.experiment;
  });
  $('new-training').disabled = busy || !connected || active;
  $('experiment-choice').disabled = busy || !connected || active;
  for (const field of $('settings').elements) field.disabled = busy || !connected || active;
  if (current?.version) $('settings').elements.method.disabled = true;
}

function render(s) {
  current = s;
  const matchups = Object.entries(s.counts.matchups ?? {});
  const historical = matchups.reduce(
    (n, [key, count]) => n + (key.includes('/historical/') ? count : 0),
    0,
  );
  const assigned = matchups.reduce((n, [, count]) => n + count, 0);
  const experiments = s.experiments ?? [];
  const choice = $('experiment-choice');
  const signature = JSON.stringify([s.experiment, experiments]);
  if (choice.dataset.signature !== signature) {
    const selected = choice.value;
    choice.replaceChildren(
      ...experiments.map((entry) => {
        const option = document.createElement('option');
        option.value = entry.id;
        option.textContent = `${entry.id}${entry.id === s.experiment ? '（当前）' : ''} · ${bytes(entry.bytes)} · ${entry.recoveries}个恢复点`;
        return option;
      }),
    );
    choice.value = experiments.some((entry) => entry.id === selected) ? selected : s.experiment;
    choice.dataset.signature = signature;
  }
  text(
    'active-games',
    (s.runtime.active ?? [])
      .map(
        (g) => `局${g.task} · ply ${g.ply ?? '初始化'} · ${Math.round(g.seconds)}秒 · 等待实际终局`,
      )
      .join('；'),
  );
  text(
    'quick-runtime',
    `${states[s.state]} · ${s.runtime.active?.length ?? 0} 局进行中 · 实际分配：当前自对弈${assigned - historical} / 历史${historical}（${percent(assigned ? historical / assigned : null)}） · RAM ${bytes(s.resources.rss)}`,
  );
  text(
    'quick-resource',
    `样本池 ${bytes(s.pool.bytes)} / ${bytes(s.pool.limit)}；磁盘 ${bytes(s.disk.used)} / ${bytes(s.disk.limit)}`,
  );
  const recent = s.evaluations.at(-1);
  text(
    'quick-evaluation',
    recent
      ? `更新 ${recent.updates} · ${Object.entries(names)
          .map(
            ([key, name]) =>
              `${name} ${percent(recent.results[key].score_rate)}（${recent.results[key].n}局）`,
          )
          .join(' · ')}`
      : '尚未评测 · 训练损失不代表对战棋力',
  );
  text('state', states[s.state]);
  text('device', `${s.device.toUpperCase()} · 单一服务会话`);
  $('error').hidden = !s.error;
  text('error', s.error ?? '');
  text(
    'method',
    s.config.method.startsWith('decomposed-mc-q-')
      ? '分解动作 MC 收益回归 · 实验'
      : '已执行动作模仿 · 接线基线',
  );
  text(
    'learning-status',
    s.learning_status + (s.research_fixture ? '（验收夹具模式，不作为棋力）' : ''),
  );
  const loss = s.losses.at(-1);
  text(
    'loss-description',
    loss
      ? `${s.config.method.startsWith('decomposed-mc-q-') ? 'Q均方误差' : '动作交叉熵'} ${loss.policy.toFixed(4)} / 价值均方误差 ${loss.value.toFixed(4)} · 更新${loss.step}`
      : '等待真实终局与更新，尚无损失记录',
  );
  text('games', number(s.counts.games));
  text(
    'discarded',
    `截断 ${number(s.counts.truncated)} · 暂停/错误丢弃 ${number(s.counts.discarded)}`,
  );
  text('updates', number(s.updates));
  text(
    'update-progress',
    `行为 ${s.version?.slice(0, 12) ?? '尚未初始化'} · 发布${s.generation}次 · 本轮 ${s.update_progress}/${s.update_budget}`,
  );
  text('samples', number(s.pool.samples));
  text(
    'retention',
    `累计保留 ${number(s.counts.retained_samples)} / ${number(s.counts.seen_samples)} · 实际消费 ${number(s.pool.consumed)}`,
  );
  text('saved', number(s.saved_step));
  text('save-time', `上次保存 ${date(s.saved_at)}`);
  text(
    'rollback',
    `若现在退出，可能丢失 ${number(s.unsaved_updates)} 次更新、${number(s.unsaved_games)} 个终局及RAM样本。${s.restored ? '已从磁盘恢复。' : ''}`,
  );
  text(
    'next-work',
    `下次评测：再完成 ${s.next_eval_games} 局并有新更新；保存：${s.save_pending ? '请求已合并，等待安全边界' : `${Math.ceil(s.next_save_seconds)} 秒后有新更新时`}`,
  );
  const css = getComputedStyle(document.body),
    teal = css.getPropertyValue('--teal'),
    ember = css.getPropertyValue('--ember'),
    gold = css.getPropertyValue('--gold');
  chart('loss-chart', [
    {
      name: '候选/策略',
      color: teal,
      points: s.losses.map((r) => ({ x: r.step, y: r.policy })),
    },
    {
      name: '价值',
      color: ember,
      points: s.losses.map((r) => ({ x: r.step, y: r.value })),
    },
  ]);
  const last = s.evaluations.at(-1),
    series = s.evaluations.filter((r) => r.series === last?.series);
  chart(
    'strength-chart',
    Object.entries(names).map(([key, name], i) => ({
      name,
      color: [teal, gold, ember, 'currentColor'][i],
      points: series
        .filter((r) => r.results[key]?.n)
        .map((r) => ({ x: r.updates, y: r.results[key].score_rate })),
    })),
    true,
  );
  text(
    'eval-ready',
    s.evaluation_ready
      ? '真实各档教师 · 固定工作预算 · 非决策方沿用生产Pass（独立调度系列）'
      : '未就绪：安装 mini-racer==0.14.1；训练可独立运行',
  );
  text(
    'evaluation-job',
    s.evaluation_job
      ? `冻结 ${s.evaluation_job.model.slice(0, 12)} · ${s.evaluation_job.confirmation ? '确认' : '趋势'} ${s.evaluation_job.completed_tasks}/${s.evaluation_job.planned}局 · 每局最多${s.evaluation_job.seconds_per_game ?? s.config.eval_seconds}秒，训练轮次之间继续`
      : '没有待评测任务',
  );
  text(
    'candidate-status',
    s.candidate
      ? `趋势候选 ${s.candidate.model.slice(0, 12)} · 得分率${percent(s.candidate.score)}。点击确认，保持同一权重完成每档至少20局；不是已确认最佳。`
      : '暂无可确认候选；先取得三档真实完成结果。初始化权重另作固定基准。',
  );
  $('strength').replaceChildren();
  for (const [key, name] of Object.entries(names)) {
    const r = last?.results[key],
      row = document.createElement('tr');
    for (const value of [
      name,
      percent(r?.win_rate),
      percent(r?.score_rate),
      r ? `${r.n} / ${r.total ?? r.n + r.unfinished + r.errors}` : '0 / 0',
    ]) {
      const td = document.createElement('td');
      td.textContent = value;
      row.append(td);
    }
    $('strength').append(row);
  }
  text('disk-label', `${bytes(s.disk.used)} / 20 GB（约18.63 GiB）`);
  $('disk-meter').value = s.disk.used + s.disk.reserved;
  $('disk-meter').max = s.disk.limit;
  text(
    'disk-detail',
    `已预留 ${bytes(s.disk.reserved)} · 60秒均速 ${bytes(s.disk.write_rate)}/s · 本次累计写入 ${bytes(s.disk.written)}`,
  );
  text(
    'disk-categories',
    Object.entries(s.disk.categories)
      .map(([key, size]) => `${key} ${bytes(size)}`)
      .join(' · '),
  );
  text(
    'memory',
    `RAM样本池 ${bytes(s.pool.bytes)} / ${bytes(s.pool.limit)} · 每样本最多复用 ${s.pool.reuse} 次`,
  );
  text(
    'compute',
    `进程树RAM ${bytes(s.resources.rss)} · CPU ${number(s.resources.cpu_percent)}% · GPU利用率：${s.resources.gpu_utilization == null ? '不可用' : number(s.resources.gpu_utilization) + '%'}`,
  );
  text(
    'consumption',
    `已结束局生成${s.counts.seen_samples} · 在途${s.runtime.inflight_samples ?? 0} · 合格${s.counts.eligible_samples ?? 0} · 终局保留${s.counts.retained_samples} · 消费${s.pool.consumed} · 用尽${s.pool.exhausted} · 年龄淘汰${s.pool.age_evicted} · 容量淘汰${s.pool.capacity_evicted} · 超大拒收${s.counts.oversize_samples}`,
  );
  text(
    'coverage',
    `实际消费：${
      Object.entries(s.pool.coverage ?? {})
        .map(([k, v]) => `${k} ${v}`)
        .join(' / ') || '等待终局'
    }；年龄上限${s.pool.max_age}个行为版本。`,
  );
  text(
    'matchups',
    `实际分配：${
      Object.entries(s.counts.matchups ?? {})
        .map(([k, v]) => `${k} ${v}`)
        .join(' / ') || '尚未分配'
    }；无不同历史权重回退${s.counts.history_fallback ?? 0}局。`,
  );
  text(
    'exports',
    s.exports?.length
      ? `最新导出：${s.disk.root}/${s.exports.at(-1)}`
      : '导出仅保留最新一份并计入配额',
  );
  text(
    'inflight',
    `在途样本 ${bytes(s.runtime.inflight_bytes)} · 活跃局 ${s.runtime.active?.length ?? 0} · 显存 ${bytes(s.resources.gpu_allocated)} / 预留 ${bytes(s.resources.gpu_reserved)}`,
  );
  text('version', `当前行为 / 学习器：${s.version?.slice(0, 16) ?? '尚未初始化'}`);
  text(
    'opponents',
    `历史对手：${s.history.length}/${s.config.history_size} · ${s.history.map((n) => Number(n.split('/').at(-1).split('.')[0])).join('、') || '尚未纳入'} 更新`,
  );
  text('checkpoints', `完整恢复点 ${s.recoveries.length} 份（保留最近2份）`);
  text(
    'best',
    s.best.length
      ? '最佳评测权重已确认（独立于当前学习器）'
      : '最佳模型：尚无符合规则的评测（三档各至少20局且三档完成率100%）',
  );
  text(
    'managed-root',
    `实验 ${s.experiment} · 受管根目录：${s.disk.root}。训练引擎：${s.readiness.engine}。采样${s.execution.sampling} / 更新${s.execution.updating}。`,
  );
  text(
    'external',
    `未纳管旧文件（仅启动时盘点，未自动删除）：${s.external.map((r) => `${r.path}：${bytes(r.bytes)}`).join('；') || '指定范围内未发现'}。20GB仅约束受管产物，不代表机器全部训练文件小于20GB。`,
  );
  if (!dirty)
    for (const field of $('settings').elements)
      if (field.name && s.config[field.name] != null) field.value = s.config[field.name];
  $('logs').replaceChildren(
    ...s.logs
      .slice()
      .reverse()
      .map((r) => {
        const li = document.createElement('li');
        li.textContent = `${date(r.time)} · ${r.message}`;
        return li;
      }),
  );
  if (last) {
    const items = [document.createElement('p')];
    items[0].textContent = `更新 ${last.updates} · 训练 ${last.games_trained} 局 · 模型 ${last.model.slice(0, 12)} · 系列 ${last.series.slice(0, 8)}`;
    for (const [key, r] of Object.entries(last.results)) {
      const p = document.createElement('p');
      p.textContent = `${names[key]}：${r.wins}胜 / ${r.losses}负 / ${r.draws}平 · 未完成 ${r.unfinished} · 错误 ${r.errors} · 超时 ${r.timeouts ?? 0} / 预算截断 ${r.truncated ?? 0} · 完成率 ${percent(r.completion)}`;
      items.push(p);
    }
    for (const r of last.games ?? []) {
      const p = document.createElement('p');
      p.textContent = `${names[r.difficulty]} · ${r.start.rules === 'classic' ? '经典' : '神龛'} · 模型执${r.side === 1 ? '先' : '后'} · ${outcomes[r.result] ?? r.result}${r.reason ? ` / ${reasons[r.reason] ?? r.reason}` : ''}${r.error ? ` / ${r.error}` : ''}`;
      if (r.budget)
        p.textContent += ` · 预设工作预算 ${r.budget.nodes} / 回合 ${r.budget.turnNodes} 节点`;
      items.push(p);
    }
    $('evaluation-details').replaceChildren(...items);
  }
  controls();
}

async function command(action, values) {
  busy = true;
  controls();
  text('feedback', '正在提交…');
  try {
    const response = await fetch('/api/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Session-Token': token },
      body: JSON.stringify({ action, values }),
      signal: AbortSignal.timeout(5000),
    });
    const result = await response.json();
    if (!response.ok) throw Error(result.error);
    text(
      'feedback',
      action === 'pause'
        ? '暂停请求已接受；正在取消在途计算。'
        : action === 'save'
          ? '保存请求已接受；以“上次保存”更新为完成依据。'
          : '请求已接受。',
    );
    if (action === 'settings') dirty = false;
    if (action === 'new-experiment') {
      dirty = false;
      $('new-dialog').close();
    }
    return true;
  } catch (error) {
    text('feedback', error.message);
    return false;
  } finally {
    busy = false;
    controls();
  }
}
document
  .querySelectorAll('[data-action]')
  .forEach((b) =>
    b.addEventListener('click', () =>
      command(
        b.dataset.action,
        b.dataset.action.endsWith('-experiment')
          ? { experiment: $('experiment-choice').value }
          : undefined,
      ),
    ),
  );
$('experiment-choice').addEventListener('change', controls);
$('new-training').addEventListener('click', () => $('new-dialog').showModal());
$('new-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (event.submitter.id === 'new-cancel') {
    $('new-dialog').close();
    return;
  }
  if (busy) return;
  const form = $('new-form').elements;
  await command('new-experiment', {
    source: form.source.value,
    device: form.device.value,
    config: { classic_percent: Number(form.rules.value) },
  });
});
$('settings').addEventListener('input', () => {
  dirty = true;
});
$('settings').addEventListener('submit', (event) => {
  event.preventDefault();
  const values = {};
  let invalid;
  for (const field of $('settings').elements)
    if (field.name && !field.disabled) {
      field.setAttribute('aria-invalid', String(!field.validity.valid));
      if (!field.validity.valid) invalid ??= field;
      values[field.name] = field.name === 'method' ? field.value : Number(field.value);
    }
  text('form-error', invalid ? '请在标注的范围内填写数值。' : '');
  if (invalid) {
    invalid.focus();
    return;
  }
  command('settings', values);
});
async function poll() {
  try {
    if (!token) {
      const session = await fetch('/api/session', {
        signal: AbortSignal.timeout(5000),
      });
      token = (await session.json()).token;
    }
    const response = await fetch('/api/status', {
      signal: AbortSignal.timeout(5000),
    });
    if (!response.ok) throw Error('服务返回错误');
    connected = true;
    text('connection', '● 已连接本地服务');
    render(await response.json());
  } catch {
    connected = false;
    token = null;
    text('connection', '连接已断开 · 训练状态未知，正在重连');
    controls();
  }
  setTimeout(poll, 1000);
}
poll();
