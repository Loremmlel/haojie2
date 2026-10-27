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
    workersDraft = false,
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
          (element.dataset.action === 'save' && status?.save_pending) ||
          (element.closest('[data-workers-form]') && status?.workers_pending);
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
  const escape = (value) =>
    String(value).replace(
      /[&<>"']/g,
      (c) =>
        ({
          '&': '&amp;',
          '<': '&lt;',
          '>': '&gt;',
          '"': '&quot;',
          "'": '&#39;',
        })[c],
    );

  // 速率按计数区间的实际时长加权，资源按观测间隔加权；不跨缺测续算。
  function average(history, fields) {
    return history.map((row, i) => {
      const result = { ...row };
      for (const field of fields) {
        if (!finite(row[field])) continue;
        let sum = 0,
          duration = 0;
        for (let j = i; j >= 0; j--) {
          const sample = history[j],
            delta = sample.seconds - history[j - 1]?.seconds;
          if (!finite(sample[field]) || sample.seconds <= row.seconds - 30) break;
          const observed = finite(delta) && delta <= 15 ? delta : 5;
          const interval = ['cpu', 'gpu'].includes(field)
            ? observed
            : (sample.interval ?? observed);
          const weight = Math.max(0, Math.min(interval, sample.seconds - (row.seconds - 30)));
          sum += sample[field] * weight;
          duration += weight;
          if (delta > 15) break;
        }
        result[field] = duration ? sum / duration : undefined;
      }
      return result;
    });
  }

  function lossHistory(history) {
    let update, trend, previous;
    return history.flatMap((row) => {
      const interrupted =
        row.gap ||
        !finite(row.loss) ||
        !finite(row.updates) ||
        (previous !== undefined && row.seconds - previous > 15);
      previous = row.seconds;
      if (interrupted) {
        update = trend = undefined;
        if (!finite(row.loss) || !finite(row.updates) || row.gap)
          return [{ ...row, loss: undefined }];
      }
      if (row.updates === update) return [];
      update = row.updates;
      trend = trend === undefined ? row.loss : trend * 0.8 + row.loss * 0.2;
      const point = { ...row, original: row.loss, loss: trend };
      return interrupted ? [{ ...row, loss: undefined }, point] : [point];
    });
  }

  function gaps(history) {
    const result = [];
    let active;
    history.forEach((row, i) => {
      const previous = history[i - 1];
      if (previous && row.seconds - previous.seconds > 15 && !active)
        result.push({ start: previous.seconds, end: row.seconds, reason: '后台采样间隔过长' });
      if (row.gap && !active)
        active = { start: previous?.seconds ?? row.seconds, end: row.seconds, reason: row.gap };
      if (active) {
        active.end = row.seconds;
        if (row.gap && row.gap !== '等待新的训练状态') active.reason = row.gap;
        if (!row.gap) {
          result.push(active);
          active = undefined;
        }
      }
    });
    if (active) result.push(active);
    return result;
  }

  function chart(key, history, series, unit, options = {}) {
    const { xKey = 'seconds', percent = false, step = false, raw = [], missing = [] } = options;
    const target = find(`[data-chart="${key}"]`);
    const values = [...history, ...raw]
      .flatMap((row) => series.map(([field]) => row[field]))
      .filter(finite);
    if (!values.length) {
      target.innerHTML = '<p class="empty">等待实际数据</p>';
      return;
    }
    const xs = history.map((row) => row[xKey]).filter(finite);
    const lowX = Math.min(...xs),
      highX = Math.max(...xs);
    const min = Math.min(...values),
      max = Math.max(...values);
    const padding = Math.max((max - min) * 0.12, Math.abs(max) * 0.02, 0.01);
    const lowY = key === 'loss' ? min - padding : 0,
      highY = percent ? 100 : max + padding;
    const x = (value) => 60 + ((value - lowX) / Math.max(1, highX - lowX)) * 424;
    const y = (value) => 176 - ((value - lowY) / (highY - lowY)) * 148;
    let paths = '',
      grid = '';
    for (let i = 0; i <= 4; i++) {
      const value = lowY + ((highY - lowY) * i) / 4;
      grid += `<path d="M60 ${y(value)}H484" stroke="var(--line)"/>
        <text x="51" y="${y(value) + 4}" text-anchor="end">${number(value, 2)}</text>`;
      const tick = lowX + ((highX - lowX) * i) / 4;
      grid += `<text x="${x(tick)}" y="198" text-anchor="${i === 0 ? 'start' : i === 4 ? 'end' : 'middle'}">${number(tick / (xKey === 'seconds' ? 60 : 1), 1)}</text>`;
    }
    const shades =
      xKey === 'seconds'
        ? missing
            .map(
              (gap) =>
                `<rect class="missing" x="${x(gap.start)}" y="28" width="${Math.max(1, x(gap.end) - x(gap.start))}" height="148" fill="var(--muted)" opacity="0.18"><title>${escape(gap.reason)} · ${number(gap.end - gap.start, 1)} 秒</title></rect>`,
            )
            .join('')
        : '';
    const draw = (rows, opacity, original = false) => {
      for (const [field, color] of series) {
        let points = [],
          previous,
          last;
        const flush = () => {
          if (points.length)
            paths += `<polyline class="${original ? 'original' : 'trend'}" fill="none" stroke="var(--${color})" opacity="${opacity}" stroke-width="${original ? 1 : 2}" stroke-linejoin="round" stroke-linecap="round" points="${points.join(' ')}"/>`;
          points = [];
        };
        for (const row of rows) {
          if (!finite(row[field]) || !finite(row[xKey])) {
            flush();
            previous = last = undefined;
            continue;
          }
          if (xKey === 'seconds' && previous && row.seconds - previous.seconds > 15) {
            flush();
            previous = undefined;
          }
          const cx = x(row[xKey]).toFixed(2),
            cy = y(row[field]).toFixed(2);
          if (step && previous) points.push(`${cx},${y(previous[field]).toFixed(2)}`);
          points.push(`${cx},${cy}`);
          previous = row;
          last = { cx, cy, value: row[field] };
        }
        flush();
        if (last && !original)
          paths += `<circle cx="${last.cx}" cy="${last.cy}" r="2.5" fill="var(--${color})"><title>${number(last.value, 3)} ${unit}</title></circle>`;
      }
    };
    if (raw.length) draw(raw, 0.2, true);
    draw(history, 1);
    const label = xKey === 'seconds' ? '运行时间 / 分钟' : '累计学习更新';
    target.innerHTML = `<svg viewBox="0 0 500 220" role="img" aria-label="${escape(target.parentElement.querySelector('h2').textContent)}">
      <text x="60" y="16">${unit}</text>${shades}${grid}
      <text x="272" y="218" text-anchor="middle">${label}</text>${paths}</svg>`;
  }

  function plots() {
    const all = status.history,
      last = all.at(-1)?.seconds ?? 0;
    const visible = (row) => row.seconds >= last - Number(find('[data-range]').value) * 60;
    const history = all.filter(visible);
    const smoothed = average(all, ['actor', 'commands', 'consume', 'cpu', 'gpu']).filter(visible);
    const missing = gaps(history),
      raw = find('[data-original]').checked ? history : [];
    chart(
      'throughput',
      smoothed,
      [
        ['actor', 'teal'],
        ['consume', 'ember'],
        ['commands', 'gold'],
      ],
      '每秒',
      { raw, missing },
    );
    const losses = lossHistory(all).filter(visible);
    chart('loss', losses, [['loss', 'teal']], '总损失', {
      xKey: 'updates',
      raw: losses.map((row) => ({ ...row, loss: row.original })),
    });
    chart('ready', history, [['ready', 'teal']], '转换', { step: true, missing });
    chart(
      'utilization',
      smoothed,
      [
        ['cpu', 'teal'],
        ['gpu', 'ember'],
      ],
      '%',
      { percent: true, raw, missing },
    );
    chart(
      'memory',
      history,
      [
        ['ram', 'teal'],
        ['vram', 'ember'],
      ],
      'GiB',
      { missing },
    );
    const latest = average(all.slice(-8), ['actor', 'consume']).at(-1) || {};
    find('[data-value="consume"]').textContent = number(latest.consume, 1);
    find('[data-value="actor"]').textContent = number(latest.actor, 1);
    find('[data-quality]').textContent = !all.length
      ? '等待后台观测'
      : all.at(-1).gap
        ? '训练状态缺测中'
        : missing.length
          ? `所选范围内 ${missing.length} 段缺测`
          : '所选范围采样连续';
    find('[data-sample-age]').textContent =
      `训练状态：${number(status.status_age, 1)} 秒前 · 后台采样：${status.sampled_at ? new Date(status.sampled_at * 1000).toLocaleTimeString('zh-CN') : '等待采样'}${status.sample_error ? ' · 采样读取异常' : ''}`;
    find('[data-gaps]').replaceChildren(
      ...missing
        .slice(-3)
        .reverse()
        .map((gap) => {
          const item = document.createElement('li');
          item.textContent = `${number(gap.start / 60, 1)}–${number(gap.end / 60, 1)} 分钟 · ${gap.reason} · ${number(gap.end - gap.start, 1)} 秒`;
          return item;
        }),
    );
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
      c = reply.configuration;
    const set = (key, value) => {
      find(`[data-value="${key}"]`).textContent = value;
    };
    set('updates', number(m.updates));
    set('games', number(m.completed_games));
    set('ready', number(m.ready_steps));
    set('lag', number(m.consumed_lag_max));
    set(
      'batch',
      `计算小批 ${number(m.batch ?? c.batch)} 条 · 每批最多 ${number(m.update_games ?? c.update_games ?? 1)} 局`,
    );
    set('lag_limit', `上限 ${number(m.lag_limit)} 次更新`);
    find('[data-state]').textContent = phases[reply.phase] || '状态未知';
    find('[data-updated]').textContent = `状态文件 ${number(reply.status_age, 1)} 秒前更新`;
    const target = m.target_concurrency ?? c.concurrency;
    if (!draft) find('input').value = finite(target) ? target : '';
    const workers = m.workers ?? c.workers;
    if (!workersDraft) find('[data-workers]').value = finite(workers) ? workers : '';
    const saved = reply.saved_at
      ? new Date(reply.saved_at * 1000).toLocaleString('zh-CN')
      : '尚未保存';
    const r = reply.resources;
    find('[data-summary]').textContent =
      `${c.gpu || c.device || '等待设备'} · ${number(workers)} 个 Worker · 活动 ${number(m.active)} / 目标 ${number(target)} 局 · 已停放 ${number(m.parked)} 局 · 终局排队 ${number(m.ready_games)} 局 · 显存 ${number(r.vram, 1)} / ${number(r.vram_total, 1)} GiB · GPU ${number(r.power, 1)} W · 最近保存：${saved}${reply.save_pending ? '（保存请求处理中）' : ''}${m.stale_games ? ` · 过期未学习 ${number(m.stale_games)} 局 / ${number(m.stale_steps)} 条` : ''}${m.training_method === 'terminal-vtrace' && !m.updates ? ' · 本轮尚无新终局更新' : ''}`;
    plots();
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
      const rules = { classic: '经典', shrine: '神龛' }[report?.rules];
      find('[data-eval-status]').textContent =
        `${evaluation.message} ${next}${report ? ` · ${phase[report.phase] || report.phase}${rules ? ` · ${rules}` : ''}${finite(report.temperature) ? ` · 温度 ${number(report.temperature, 1)}` : ''} · 模型更新 ${number(report.updates)} · 第 ${number(report.current)} / ${report.games.length} 局${report.error ? ` · ${report.error}` : ''}` : ''}`;
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
          : '已连接；后台持续采样，页面每 5 秒刷新，速率显示最近 30 秒均值。',
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
      if (action === 'workers' && !reply.error && !reply.closed) workersDraft = false;
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
  find('[data-workers]').addEventListener('input', () => {
    workersDraft = true;
    find('[data-workers]').removeAttribute('aria-invalid');
  });
  find('[data-workers-form]').addEventListener('submit', (event) => {
    event.preventDefault();
    const input = find('[data-workers]'),
      value = input.valueAsNumber;
    if (!Number.isInteger(value) || value < 1 || value > 128) {
      input.setAttribute('aria-invalid', 'true');
      input.focus();
      notice('Worker 数须为 1–128 的整数。', true);
      return;
    }
    submit('workers', value);
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
      if (!input.checkValidity()) {
        input.setAttribute('aria-invalid', 'true');
        input.focus();
        notice(
          input.name === 'temperature'
            ? '测评温度须为 0–2，步长 0.1。'
            : '请填写范围内的整数测评参数。',
          true,
        );
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
  for (const selector of ['[data-range]', '[data-original]'])
    find(selector).addEventListener('change', () => {
      if (status) plots();
    });
  poll();
})();
