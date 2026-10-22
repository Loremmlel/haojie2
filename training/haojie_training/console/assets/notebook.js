(() => {
  const root = document.getElementById(__PANEL_ID__);
  const view = root.querySelector('[data-view]');
  const notice = root.querySelector('[data-notice]');
  const refresh = root.querySelector('[data-refresh]');
  let running = true,
    closed = false,
    busy = false,
    disconnected = false,
    timer,
    queue = Promise.resolve();

  function message(text, error = false) {
    notice.textContent = text;
    notice.setAttribute('role', error ? 'alert' : 'status');
  }

  // 仅更新已有节点；展开的 JSON 保持当前快照，保留选区、焦点及独立滚动位置。
  function patch(current, next) {
    if (!current || current.nodeType !== next.nodeType || current.nodeName !== next.nodeName) {
      current?.replaceWith(next.cloneNode(true));
      return;
    }
    if (current.nodeType === Node.TEXT_NODE) {
      if (current.data !== next.data) current.data = next.data;
      return;
    }
    if (current.nodeType !== Node.ELEMENT_NODE) return;
    if (current.hasAttribute('data-snapshot') && current.closest('details')?.open) return;
    for (const attr of [...current.attributes]) {
      if (attr.name !== 'open' && !next.hasAttribute(attr.name)) current.removeAttribute(attr.name);
    }
    for (const attr of next.attributes) {
      if (attr.name !== 'open' && current.getAttribute(attr.name) !== attr.value) {
        current.setAttribute(attr.name, attr.value);
      }
    }
    [...next.childNodes].forEach((child, index) => {
      const key = child.nodeType === Node.ELEMENT_NODE ? child.getAttribute('data-key') : null;
      let existing = key
        ? [...current.children].find((el) => el.getAttribute('data-key') === key)
        : current.childNodes[index];
      if (
        !existing ||
        (!key && existing.nodeType === Node.ELEMENT_NODE && existing.hasAttribute('data-key'))
      ) {
        current.insertBefore(child.cloneNode(true), current.childNodes[index] || null);
        return;
      }
      if (existing !== current.childNodes[index])
        current.insertBefore(existing, current.childNodes[index] || null);
      patch(existing, child);
    });
    while (current.childNodes.length > next.childNodes.length) current.lastChild.remove();
  }

  function update(reply) {
    if (reply.closed) {
      closed = true;
      running = false;
      clearTimeout(timer);
      root.querySelectorAll('button').forEach((button) => {
        button.disabled = true;
      });
      throw new Error('此面板已失效，请使用最近一次运行第 4 步生成的面板。');
    }
    if (reply.error) throw new Error(reply.error);
    const next = document.createElement('template');
    next.innerHTML = reply.html;
    const focused = document.activeElement;
    const selection = getSelection();
    const range =
      selection && !selection.isCollapsed
        ? {
            start: selection.anchorNode,
            from: selection.anchorOffset,
            end: selection.focusNode,
            to: selection.focusOffset,
          }
        : null;
    patch(view.querySelector('.hj-colab'), next.content.querySelector('.hj-colab'));
    if (focused?.isConnected && document.activeElement !== focused)
      focused.focus({ preventScroll: true });
    if (range?.start.isConnected && range.end.isConnected) {
      selection.setBaseAndExtent(range.start, range.from, range.end, range.to);
    }
  }

  function call(action = 'status', values = null) {
    const invoke = async () => {
      const reply = await google.colab.kernel.invokeFunction(
        'haojie.panel',
        [root.id, action, values],
        {},
      );
      return reply.data['application/json'];
    };
    const result = queue.then(invoke, invoke);
    queue = result.catch(() => {});
    return result;
  }

  async function poll() {
    clearTimeout(timer);
    if (!running || closed || !root.isConnected) return;
    if (!busy) {
      try {
        update(await call());
        if (disconnected) message('连接已恢复，已读取最新状态。');
        disconnected = false;
      } catch (error) {
        disconnected = true;
        message(`状态未知，保留上次读数。${error.message}`, true);
      }
    }
    if (running && !closed && root.isConnected) timer = setTimeout(poll, 5000);
  }

  async function submit(action, values) {
    if (busy || closed) return;
    busy = true;
    const buttons = root.querySelectorAll('[data-action], button[type=submit]');
    buttons.forEach((button) => {
      button.disabled = true;
    });
    message('正在提交…');
    try {
      const reply = await call(action, values);
      update(reply);
      message(reply.message);
    } catch (error) {
      message(error.message, true);
    } finally {
      busy = false;
      buttons.forEach((button) => {
        button.disabled = closed;
      });
    }
  }

  root.querySelectorAll('[data-action]').forEach((button) => {
    button.addEventListener('click', () => submit(button.dataset.action, null));
  });
  root.querySelectorAll('form').forEach((form) => {
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      const values = {};
      try {
        if (form.dataset.kind === 'json') {
          submit('settings', JSON.parse(form.querySelector('textarea').value));
          return;
        }
        for (const field of form.querySelectorAll('input')) {
          field.removeAttribute('aria-invalid');
          const value = field.valueAsNumber;
          if (!Number.isInteger(value) || value < Number(field.min) || value > Number(field.max)) {
            field.setAttribute('aria-invalid', 'true');
            field.focus();
            throw new Error(
              `${field.labels[0].textContent.trim()}须为 ${field.min}–${field.max} 的整数`,
            );
          }
          values[field.name] = value;
        }
        submit(form.dataset.kind === 'concurrency' ? 'concurrency' : 'settings', values);
      } catch (error) {
        message(`设置未提交：${error.message}`, true);
      }
    });
  });
  refresh.addEventListener('click', () => {
    running = !running;
    refresh.textContent = running ? '暂停刷新' : '继续刷新';
    clearTimeout(timer);
    message(running ? '自动刷新已继续。' : '自动刷新已暂停，训练继续运行。');
    if (running) poll();
  });
  timer = setTimeout(poll, 5000);
})();
