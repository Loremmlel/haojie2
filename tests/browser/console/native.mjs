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
      directory: '/content/drive/MyDrive/haojie-training/test',
      checkpoint: '/content/run/checkpoint.pt',
      log: '',
      evaluation: {
        settings: { minutes: 30, pairs: 1, seconds: 120 },
        pending: false,
        active: false,
        next_due: Date.now() / 1000 + 1800,
        message: '可开始测评',
        directory: '/content/run/evaluations/test',
        report: {
          phase: 'finished',
          updates: 120,
          current: 12,
          games: Array.from({ length: 12 }, () => ({ result: 'unfinished' })),
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
  assert.equal(await page.locator('svg').count(), 6);
  assert.equal(await page.locator('[data-value="consume"]').innerText(), '5,580');
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
  assert.equal(await page.locator('svg').count(), 6);
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
  await page.getByRole('button', { name: '应用测评计划', exact: true }).click();
  await page.waitForFunction(() =>
    window.calls.some((call) => call[2] === 'eval_config' && call[3].minutes === 0),
  );
  assert.equal(
    await page.locator('[data-eval-results]').getByText('50%', { exact: true }).count(),
    1,
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
  assert.equal(await page.getByText('等待实际数据', { exact: true }).count(), 6);
  await page.evaluate(() => {
    window.mode = 'closed';
  });
  await page.getByText('此面板已由新的输出替代，请使用最新面板。', { exact: true }).waitFor();
  assert.equal(await page.locator('[data-refresh]').isDisabled(), true);
  assert.deepEqual(errors, []);
  console.log(
    'Native Colab panel: charts, controls, draft/focus, frozen JSON, offline/reconnect, empty/old panel and 390px passed.',
  );
  if (process.env.PANEL_SCREENSHOTS) console.log(`Screenshots: ${temporary}`);
} finally {
  await browser.close();
  assert.ok(temporary.startsWith(join(tmpdir(), 'haojie-native-panel-')));
  if (!process.env.PANEL_SCREENSHOTS) await rm(temporary, { recursive: true });
}
