(() => {
  const identity = __PANEL_ID__;
  const root = document.getElementById(identity);
  const find = (selector) => root.querySelector(selector);
  const finite = (value) => typeof value === 'number' && Number.isFinite(value);
  const number = (value, digits = 0) =>
    finite(value) ? value.toLocaleString('zh-CN', { maximumFractionDigits: digits }) : '—';
  const phases = {
    not_started: '尚未启动',
    initializing: '正在初始化',
    running: '训练运行中',
    stopping: '正在保存并停止',
    stopped: '已停止',
    failed: '训练异常退出',
    stale: '状态更新延迟',
    unknown: '状态未知',
  };
  let running = true,
    busy = false,
    polling = false,
    closed = false,
    timer,
    status,
    draft = false,
    evaluationDraft = false,
    disconnected = false;
  let queue = Promise.resolve();
  const notice = (message, error = false) => {
    find('[data-notice]').textContent = message;
    find('[data-notice]').setAttribute('role', error ? 'alert' : 'status');
  };
  function buttons() {
    find('[data-refresh]').disabled = closed;
    root
      .querySelectorAll(
        '.toolbar:first-of-type [data-action], .toolbar:first-of-type button[type=submit], .toolbar:first-of-type input',
      )
      .forEach((element) => {
        element.disabled =
          busy ||
          closed ||
          disconnected ||
          !(element.dataset.action === 'stop' ? status?.can_stop : status?.can_control) ||
          (element.dataset.action === 'save' && status?.save_pending);
      });
    const evaluation = status?.evaluation;
    const disabled = busy || closed || disconnected || !evaluation;
    find('[data-eval-action="evaluate"]').disabled =
      disabled || evaluation?.pending || evaluation?.active;
    find('[data-eval-action="eval_cancel"]').disabled =
      disabled || !(evaluation?.pending || evaluation?.active);
    find('[data-eval-form]')
      .querySelectorAll('input, button')
      .forEach((element) => {
        element.disabled = disabled;
      });
  }
  function chart(key, history, series, unit, xKey = 'seconds', percent = false) {
    const target = find(`[data-chart="${key}"]`);
    const values = history.flatMap((row) => series.map(([field]) => row[field])).filter(finite);
    if (!values.length) {
      target.innerHTML = '<p class="empty">等待实际数据</p>';
      return;
    }
    const xs = history.map((row) => row[xKey]).filter(finite);
    const lowX = Math.min(...xs),
      highX = Math.max(...xs);
    const lowY = Math.min(0, ...values),
      highY = Math.max(percent ? 100 : 0.01, ...values);
    let paths = '';
    for (const [field, color] of series) {
      let points = [],
        previous;
      const flush = () => {
        if (points.length)
          paths += `<polyline fill="none" stroke="var(--${color})" stroke-width="2" points="${points.join(' ')}"/>`;
        points = [];
      };
      for (const row of history) {
        if (!finite(row[field]) || !finite(row[xKey])) {
          flush();
          previous = undefined;
          continue;
        }
        if (previous !== undefined && row.seconds - previous > 15) flush();
        previous = row.seconds;
        const x = 48 + ((row[xKey] - lowX) / Math.max(1, highX - lowX)) * 428;
        const y = 150 - ((row[field] - lowY) / (highY - lowY)) * 125;
        points.push(`${x.toFixed(2)},${y.toFixed(2)}`);
        paths += `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="1.7" fill="var(--${color})"><title>${number(row[field], 3)}</title></circle>`;
      }
      flush();
    }
    const label = xKey === 'seconds' ? '运行时间 / 分钟' : '累计学习更新';
    const scale = xKey === 'seconds' ? 60 : 1;
    target.innerHTML = `<svg viewBox="0 0 500 190" role="img" aria-label="${target.parentElement.querySelector('h2').textContent}">
      <path d="M48 25V150H476" fill="none" stroke="var(--line)"/>
      <text x="1" y="16">${unit}</text><text x="1" y="35">${number(highY, 2)}</text><text x="1" y="153">${number(lowY, 2)}</text>
      <text x="48" y="170">${number(lowX / scale, 1)}</text><text x="476" y="170" text-anchor="end">${number(highX / scale, 1)}</text>
      <text x="260" y="187" text-anchor="middle">${label}</text>${paths}</svg>`;
  }
  function update(reply) {
    if (reply.closed) {
      closed = true;
      running = false;
      buttons();
      find('[data-refresh]').disabled = true;
      notice('此面板已由新的输出替代，请使用最新面板。');
      return;
    }
    if (reply.error) throw new Error(reply.error);
    const previous = status;
    status = reply;
    const m = reply.metrics,
      c = reply.configuration,
      history = reply.history;
    const latest = history.at(-1) || {};
    const set = (key, value) => {
      find(`[data-value="${key}"]`).textContent = value;
    };
    set('consume', number(latest.consume, 1));
    set('actor', number(latest.actor, 1));
    set('updates', number(m.updates));
    set('games', number(m.completed_games));
    set('ready', number(m.ready_steps));
    set('lag', number(m.consumed_lag_max));
    set('batch', `学习 batch ${number(m.batch ?? c.batch)}`);
    set('lag_limit', `上限 ${number(m.lag_limit)} 次更新`);
    find('[data-state]').textContent = phases[reply.phase] || '状态未知';
    find('[data-updated]').textContent = `读取于 ${new Date().toLocaleTimeString('zh-CN')}`;
    const target = m.target_concurrency ?? c.concurrency;
    if (!draft) find('input').value = finite(target) ? target : '';
    const saved = reply.saved_at
      ? new Date(reply.saved_at * 1000).toLocaleString('zh-CN')
      : '尚未保存';
    const r = reply.resources;
    find('[data-summary]').textContent =
      `${c.gpu || c.device || '等待设备'} · ${number(c.workers)} 个 lane · 活动 ${number(m.active)} / 目标 ${number(target)} 局 · 已停放 ${number(m.parked)} 局 · 显存 ${number(r.vram, 1)} / ${number(r.vram_total, 1)} GiB · GPU ${number(r.power, 1)} W · 最近保存：${saved}${reply.save_pending ? '（保存请求处理中）' : ''}`;
    chart(
      'throughput',
      history,
      [
        ['actor', 'teal'],
        ['consume', 'ember'],
        ['commands', 'gold'],
      ],
      '每秒',
    );
    chart('loss', history, [['loss', 'teal']], '总损失', 'updates');
    chart('ready', history, [['ready', 'teal']], '转换');
    chart('lag', history, [['lag', 'ember']], '版本差');
    chart(
      'utilization',
      history,
      [
        ['cpu', 'teal'],
        ['gpu', 'ember'],
      ],
      '%',
      'seconds',
      true,
    );
    chart(
      'memory',
      history,
      [
        ['ram', 'teal'],
        ['vram', 'ember'],
      ],
      'GiB',
    );
    if (!find('[data-raw]').open) find('[data-json]').textContent = JSON.stringify(m, null, 2);
    if (!find('[data-config]').open)
      find('[data-configuration]').textContent = JSON.stringify(
        { ...c, directory: reply.directory, checkpoint: reply.checkpoint },
        null,
        2,
      );
    if (!find('[data-log]').open) find('[data-logs]').textContent = reply.log || '暂无日志';
    const evaluation = reply.evaluation;
    if (evaluation) {
      if (!evaluationDraft) {
        for (const input of find('[data-eval-form]').querySelectorAll('input'))
          input.value = evaluation.settings[input.name];
      }
      const report = evaluation.report;
      const next =
        evaluation.settings.minutes && evaluation.next_due
          ? `下次计划 ${new Date(evaluation.next_due * 1000).toLocaleTimeString('zh-CN')}`
          : '定期测评已关闭';
      const phase = {
        initializing: '加载冻结模型',
        running: '测评进行中',
        finished: '本轮结束',
        cancelled: '已取消',
        failed: '测评失败',
      };
      find('[data-eval-status]').textContent =
        `${evaluation.message} ${next}${report ? ` · ${phase[report.phase] || report.phase} · 模型更新 ${number(report.updates)} · 第 ${number(report.current)} / ${report.games.length} 局${report.error ? ` · ${report.error}` : ''}` : ''}`;
      const body = find('[data-eval-results]');
      body.replaceChildren();
      for (const [key, label] of [
        ['easy', '简单'],
        ['medium', '普通'],
        ['hard', '困难'],
      ]) {
        const row = document.createElement('tr'),
          counts = report?.summary?.[key];
        const values = [
          label,
          counts ? `${counts.win} / ${counts.loss} / ${counts.draw}` : '—',
          counts ? `${counts.completed} / ${counts.total}` : '—',
          counts ? `${counts.unfinished} / ${counts.error}` : '—',
          finite(counts?.win_rate) ? `${number(counts.win_rate * 100, 1)}%` : '—',
        ];
        values.forEach((value) => {
          const cell = document.createElement('td');
          cell.textContent = value;
          row.append(cell);
        });
        body.append(row);
      }
      if (!find('[data-eval-detail]').open)
        find('[data-eval-json]').textContent = JSON.stringify(
          { directory: evaluation.directory, report },
          null,
          2,
        );
    }
    if (reply.phase === 'failed')
      notice(`训练异常退出（退出码 ${reply.exit_code ?? '未知'}），请展开运行日志。`, true);
    else if (reply.phase === 'stale' || reply.phase === 'unknown')
      notice('状态未知或更新延迟，保留上次观测。请查看日志或重新展示面板。', true);
    else if (reply.message) notice(reply.message);
    else if (previous?.save_pending && !reply.save_pending) notice('检查点保存完成。');
    else if (!previous)
      notice(
        reply.phase === 'not_started'
          ? '请在启动单元中开始训练，随后重新展示此面板。'
          : '已连接，每 5 秒更新；速率使用相邻状态的差值。',
      );
    buttons();
  }
  function call(action = 'status', value = null) {
    const invoke = async () => {
      const result = await google.colab.kernel.invokeFunction(
        'haojie.native_panel',
        [identity, action, value],
        {},
      );
      return result.data['application/json'];
    };
    const result = queue.then(invoke, invoke);
    queue = result.catch(() => {});
    return result;
  }
  async function poll() {
    clearTimeout(timer);
    if (!running || closed || !root.isConnected) return;
    if (!busy && !polling) {
      polling = true;
      try {
        const reply = await call();
        const recovered = disconnected;
        disconnected = false;
        update(reply);
        if (recovered) notice('连接已恢复，已读取最新状态。');
      } catch (error) {
        disconnected = true;
        find('[data-state]').textContent = '状态未知';
        notice(`连接中断，保留上次读数。${error.message}`, true);
      } finally {
        polling = false;
        buttons();
      }
    }
    if (running && !closed && root.isConnected) timer = setTimeout(poll, 5000);
  }
  async function submit(action, value = null) {
    if (busy || closed || disconnected) return;
    busy = true;
    buttons();
    notice('正在提交…');
    try {
      const reply = await call(action, value);
      update(reply);
      if (action === 'resize' && !reply.error && !reply.closed) draft = false;
      if (action === 'eval_config' && !reply.error && !reply.closed) evaluationDraft = false;
    } catch (error) {
      notice(`请求结果未知，请读取最新状态后确认。${error.message}`, true);
    } finally {
      busy = false;
      buttons();
    }
  }
  find('input').addEventListener('input', () => {
    draft = true;
    find('input').removeAttribute('aria-invalid');
  });
  find('form').addEventListener('submit', (event) => {
    event.preventDefault();
    const input = find('input'),
      value = input.valueAsNumber;
    if (!Number.isInteger(value) || value < 0 || value > 128) {
      input.setAttribute('aria-invalid', 'true');
      input.focus();
      notice('并发局数须为 0–128 的整数。', true);
      return;
    }
    submit('resize', value);
  });
  root
    .querySelectorAll('[data-action]')
    .forEach((button) => button.addEventListener('click', () => submit(button.dataset.action)));
  root
    .querySelectorAll('[data-eval-action]')
    .forEach((button) => button.addEventListener('click', () => submit(button.dataset.evalAction)));
  find('[data-eval-form]').addEventListener('submit', (event) => {
    event.preventDefault();
    const value = {};
    for (const input of event.currentTarget.querySelectorAll('input')) {
      if (!input.checkValidity() || !Number.isInteger(input.valueAsNumber)) {
        input.setAttribute('aria-invalid', 'true');
        input.focus();
        notice('请填写范围内的整数测评参数。', true);
        return;
      }
      input.removeAttribute('aria-invalid');
      value[input.name] = input.valueAsNumber;
    }
    submit('eval_config', value);
  });
  find('[data-eval-form]').addEventListener('input', () => {
    evaluationDraft = true;
  });
  find('[data-refresh]').addEventListener('click', () => {
    running = !running;
    find('[data-refresh]').textContent = running ? '暂停刷新' : '恢复刷新';
    if (running) poll();
    else {
      clearTimeout(timer);
      notice('已暂停面板刷新，训练继续运行。');
    }
  });
  poll();
})();
