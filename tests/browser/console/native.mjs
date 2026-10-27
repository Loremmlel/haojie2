import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { chromium } from 'playwright';

// 页面使用真实渲染代码，传输与训练状态使用替身；不启动采集或学习。
const temporary = await mkdtemp(join(tmpdir(), 'haojie-native-panel-'));
const python =
  process.env.PYTHON ||
  (process.platform === 'win32' ? 'training/.venv/Scripts/python.exe' : 'python3');
const html = execFileSync(
  python,
  ['-B', '-c', 'from training.colab.panel import Panel; print(Panel("unused").render())'],
  { encoding: 'utf8', env: { ...process.env, PYTHONIOENCODING: 'utf-8' } },
);
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1180, height: 1000 } });
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.evaluate(() => {
    window.calls = [];
    window.mode = 'running';
    window.reply = {
      phase: 'running',
      can_control: true,
      can_stop: true,
      save_pending: false,
      saved_at: null,
      sampled_at: Date.now() / 1000,
      status_age: 1,
      directory: '/content/drive/MyDrive/haojie-training/test',
      checkpoint: '/content/run/checkpoint.pt',
      log: '',
      evaluation: {
        settings: { minutes: 30, pairs: 1, seconds: 120, temperature: 0.5 },
        pending: false,
        active: false,
        next_due: Date.now() / 1000 + 1800,
        message: '可开始测评',
        directory: '/content/run/evaluations/test',
        report: {
          phase: 'finished',
          updates: 120,
          rules: 'classic',
          temperature: 0.5,
          current: 6,
          games: Array.from({ length: 6 }, () => ({ rules: 'classic', result: 'unfinished' })),
          summary: {
            easy: {
              win: 1,
              loss: 1,
              draw: 0,
              completed: 2,
              total: 4,
              unfinished: 2,
              error: 0,
              win_rate: 0.5,
            },
          },
        },
      },
      configuration: { gpu: 'NVIDIA L4', workers: 6, batch: 512, concurrency: 16 },
      metrics: {
        updates: 120,
        completed_games: 7,
        ready_steps: 768,
        consumed_lag_max: 24,
        lag_limit: 64,
        target_concurrency: 16,
        active: 16,
        parked: 0,
      },
      resources: { gpu: 80, vram: 8.1, vram_total: 22.5, power: 66 },
      history: Array.from({ length: 30 }, (_, i) => ({
        seconds: i * 5,
        interval: 5,
        updates: i * 4,
        loss: 0.8 / (1 + i / 10),
        actor: 12000 + Math.sin(i) * 700,
        commands: 1800 + i * 5,
        consume: 5000 + i * 20,
        ready: 500 + i * 8,
        lag: Math.min(24, i),
        cpu: 60 + Math.sin(i) * 8,
        gpu: 80 + Math.cos(i) * 5,
        ram: 5 + i / 60,
        vram: 8 + i / 80,
      })),
    };
    window.google = {
      colab: {
        kernel: {
          invokeFunction: async (name, args) => {
            window.calls.push([name, ...args]);
            if (window.mode === 'offline') throw new Error('测试连接中断');
            if (window.mode === 'closed') return { data: { 'application/json': { closed: true } } };
            if (window.mode === 'error')
              return { data: { 'application/json': { error: '测试保存失败' } } };
            const response = structuredClone(window.reply);
            if (args[1] === 'resize') response.message = `并发请求已送达：${args[2]} 局`;
            return { data: { 'application/json': response } };
          },
        },
      },
    };
  });
  await page.setContent(html);
  await page.getByText('训练运行中', { exact: true }).waitFor();
  assert.equal(await page.locator('svg').count(), 5);
  assert.equal(await page.locator('[data-value="consume"]').innerText(), '5,530');
  assert.ok((await page.locator('svg circle').count()) <= 9);
  assert.equal(await page.locator('[data-chart="throughput"] .original').count(), 0);
  await page.getByLabel('显示原始波动', { exact: true }).check();
  assert.equal(await page.locator('[data-chart="throughput"] .original').count(), 3);
  await page.getByLabel('显示原始波动', { exact: true }).uncheck();
  // 不等间隔的观测必须按时长聚合，不能把三个速率做普通平均。
  await page.evaluate(() => {
    window.initialHistory = window.reply.history;
    window.reply.history = [
      { seconds: 0 },
      { seconds: 5, interval: 5, consume: 0 },
      { seconds: 20, interval: 15, consume: 100 },
      { seconds: 30, interval: 10, consume: 0 },
    ];
  });
  await page.waitForFunction(
    () => document.querySelector('[data-value="consume"]').textContent === '50',
  );
  await page.evaluate(() => {
    window.reply.history = window.initialHistory;
  });
  await page.waitForFunction(
    () => document.querySelector('[data-value="consume"]').textContent === '5,530',
  );
  const input = page.getByLabel('并发局数', { exact: true });
  await input.fill('32');
  await page.getByText('原始状态（展开时固定快照）', { exact: true }).click();
  const snapshot = await page.locator('[data-json]').textContent();
  await page.evaluate(() => {
    window.reply.metrics.updates = 999;
  });
  await input.focus();
  await page.waitForFunction(
    () => document.querySelector('[data-value="updates"]').textContent === '999',
  );
  assert.equal(await input.inputValue(), '32');
  assert.equal(await input.evaluate((element) => document.activeElement === element), true);
  assert.equal(await page.locator('[data-json]').textContent(), snapshot);
  await input.press('Enter');
  await page.waitForFunction(() =>
    window.calls.some((call) => call[2] === 'resize' && call[3] === 32),
  );
  const workers = page.getByLabel('Worker 数', { exact: true });
  await workers.fill('12');
  await workers.press('Enter');
  await page.waitForFunction(() =>
    window.calls.some((call) => call[2] === 'workers' && call[3] === 12),
  );
  await workers.fill('0');
  await workers.press('Enter');
  assert.equal(await workers.getAttribute('aria-invalid'), 'true');
  await page.evaluate(() => {
    window.reply.metrics.workers = 12;
  });
  await workers.fill('12');
  await workers.press('Enter');
  await page.waitForFunction(() =>
    document.querySelector('[data-summary]').textContent.includes('12 个 Worker'),
  );
  await input.fill('129');
  await input.press('Enter');
  assert.equal(await input.getAttribute('aria-invalid'), 'true');
  await input.fill('8');
  await page.getByRole('button', { name: '暂停刷新', exact: true }).click();
  const before = await page.evaluate(() => window.calls.length);
  await page.waitForTimeout(5100);
  assert.equal(await page.evaluate(() => window.calls.length), before);
  await page.evaluate(() => {
    window.mode = 'offline';
  });
  await page.getByRole('button', { name: '恢复刷新', exact: true }).click();
  await page.getByText('状态未知', { exact: true }).waitFor();
  assert.equal(await page.locator('svg').count(), 5);
  assert.equal(
    await page.getByRole('button', { name: '保存检查点', exact: true }).isDisabled(),
    true,
  );
  await page.evaluate(() => {
    window.mode = 'running';
  });
  await page.getByText('训练运行中', { exact: true }).waitFor();
  await page.getByRole('button', { name: '保存检查点', exact: true }).click();
  await page.waitForFunction(() => window.calls.some((call) => call[2] === 'save'));
  await page.getByRole('button', { name: '立即测评', exact: true }).click();
  await page.waitForFunction(() => window.calls.some((call) => call[2] === 'evaluate'));
  await page.getByLabel('间隔（分钟，0 关闭）', { exact: true }).fill('0');
  const temperature = page.getByLabel('测评温度', { exact: true });
  assert.equal(await temperature.inputValue(), '0.5');
  await temperature.fill('2.1');
  await page.getByRole('button', { name: '应用测评计划', exact: true }).click();
  assert.equal(await temperature.getAttribute('aria-invalid'), 'true');
  assert.equal(await temperature.evaluate((element) => element === document.activeElement), true);
  await temperature.fill('0.3');
  await page.getByRole('button', { name: '应用测评计划', exact: true }).click();
  await page.waitForFunction(() =>
    window.calls.some(
      (call) => call[2] === 'eval_config' && call[3].minutes === 0 && call[3].temperature === 0.3,
    ),
  );
  assert.equal(
    await page.locator('[data-eval-results]').getByText('50%', { exact: true }).count(),
    1,
  );
  assert.match(await page.locator('[data-eval-status]').textContent(), /经典.*第 6 \/ 6 局/);
  assert.match(await page.locator('[data-eval-status]').textContent(), /温度 0\.5/);
  await page.evaluate(() => {
    window.reply.evaluation.report.rules = 'shrine';
  });
  await page.waitForFunction(() =>
    document.querySelector('[data-eval-status]').textContent.includes('神龛'),
  );
  await page.evaluate(() => {
    window.reply.evaluation.active = true;
  });
  await page.waitForFunction(
    () => !document.querySelector('[data-eval-action="eval_cancel"]').disabled,
  );
  await page.getByRole('button', { name: '取消测评', exact: true }).click();
  await page.waitForFunction(() => window.calls.some((call) => call[2] === 'eval_cancel'));
  await page.getByRole('button', { name: '保存并停止', exact: true }).click();
  await page.waitForFunction(() => window.calls.some((call) => call[2] === 'stop'));
  await page.evaluate(() => {
    window.reply.history = Array.from({ length: 720 }, (_, i) => {
      const row = {
        seconds: i * 5,
        interval: 5,
        updates: Math.floor(i / 2),
        loss: 1.8 - i / 1800 + Math.sin(Math.floor(i / 2) * 2) * 0.2,
        actor: 12000 + Math.sin(i * 2) * 5000,
        commands: 1800 + Math.sin(i) * 600,
        consume: i % 6 === 0 ? 12000 : 0,
        ready: i % 45 === 0 ? 3000 : 80,
        cpu: 35 + Math.sin(i * 2) * 20,
        gpu: i % 8 < 2 ? 100 : 5,
        ram: 5 + i / 600,
        vram: 8 + i / 900,
      };
      if (i >= 530 && i <= 535) {
        for (const field of ['actor', 'commands', 'consume', 'loss', 'ready', 'updates'])
          delete row[field];
        row.gap = i === 530 ? '等待新的训练状态' : '状态文件更新延迟';
      }
      return row;
    });
  });
  await page.getByText('所选范围内 1 段缺测', { exact: true }).waitFor();
  assert.match(await page.locator('[data-gaps]').textContent(), /状态文件更新延迟.*35 秒/);
  assert.ok((await page.locator('[data-chart="throughput"] rect.missing').count()) > 0);
  assert.ok((await page.locator('svg circle').count()) <= 9);
  await page.getByLabel('时间范围', { exact: true }).selectOption('10');
  await page.getByText('所选范围采样连续', { exact: true }).waitFor();
  assert.equal(await page.locator('[data-chart="throughput"] rect.missing').count(), 0);
  assert.equal(await page.getByLabel('时间范围', { exact: true }).inputValue(), '10');
  await page.getByLabel('时间范围', { exact: true }).selectOption('30');
  await page.getByLabel('时间范围', { exact: true }).focus();
  await page.keyboard.press('ArrowDown');
  await page.keyboard.press('Enter');
  await page.getByLabel('时间范围', { exact: true }).selectOption('30');
  await page.screenshot({ path: join(temporary, 'desktop.png'), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.screenshot({ path: join(temporary, 'mobile.png'), fullPage: true });
  await page.evaluate(() => {
    window.reply.history = [];
    window.reply.metrics = {};
    window.reply.can_control = false;
    window.reply.phase = 'not_started';
  });
  await page.getByText('尚未启动', { exact: true }).waitFor();
  assert.equal(await page.getByText('等待实际数据', { exact: true }).count(), 5);
  await page.evaluate(() => {
    window.mode = 'closed';
  });
  await page.getByText('此面板已由新的输出替代，请使用最新面板。', { exact: true }).waitFor();
  assert.equal(await page.locator('[data-refresh]').isDisabled(), true);
  assert.deepEqual(errors, []);
  console.log(
    'Native Colab panel: weighted trends, raw overlay, gaps/ranges, controls, draft/focus, offline/reconnect, empty/old panel and 390px passed.',
  );
  if (process.env.PANEL_SCREENSHOTS) console.log(`Screenshots: ${temporary}`);
} finally {
  await browser.close();
  assert.ok(temporary.startsWith(join(tmpdir(), 'haojie-native-panel-')));
  if (!process.env.PANEL_SCREENSHOTS) await rm(temporary, { recursive: true });
}
