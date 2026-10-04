import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { chromium } from 'playwright';

const output = resolve('artifacts/console-browser-' + Date.now());
await mkdir(output, { recursive: true });
const python =
  process.env.HAOJIE_PYTHON ||
  (process.platform === 'win32' ? 'training/.venv/Scripts/python.exe' : 'python');
const engine = resolve(
  process.env.HAOJIE_NATIVE || 'artifacts/native-target/release/haojie-engine.exe',
);
const processes = [];
async function start(root, quota = 20_000_000_000) {
  const child = spawn(
    python,
    [
      'training/tests/console/serve_fixture.py',
      '--engine',
      engine,
      '--root',
      root,
      '--quota',
      String(quota),
    ],
    { windowsHide: true },
  );
  processes.push(child);
  let errors = '',
    line = '';
  child.stderr.on('data', (data) => {
    errors += data.toString().slice(0, 2000);
  });
  const url = await new Promise((done, reject) => {
    const timer = setTimeout(() => reject(Error('server startup timeout ' + errors)), 30000);
    child.on('exit', (code) => {
      clearTimeout(timer);
      reject(Error(`server exited ${code}: ${errors}`));
    });
    child.stdout.on('data', (data) => {
      line += data;
      if (line.includes('\n')) {
        clearTimeout(timer);
        done(JSON.parse(line.split('\n')[0]).url);
      }
    });
  });
  return { child, url };
}
const browser = await chromium.launch({ executablePath: process.env.BROWSER_PATH, headless: true });
const checks = [],
  errors = [];
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
page.on('pageerror', (error) => errors.push(String(error)));
const wait = async (url, predicate) => {
  for (let i = 0; i < 240; i++) {
    const state = await (await fetch(url + '/api/status')).json();
    if (predicate(state)) return state;
    assert.notEqual(state.state, 'error', state.error);
    await new Promise((done) => setTimeout(done, 250));
  }
  throw Error('状态等待超时');
};
try {
  const root = resolve(output, 'managed');
  const first = await start(root);
  const { url } = first;
  await page.goto(url);
  await page.waitForFunction(() =>
    document.getElementById('connection').textContent.includes('已连接'),
  );
  assert.match(await page.locator('#strength').textContent(), /未评测/);
  assert.ok(
    await page.evaluate(() => {
      const workspace = document.getElementById('workspace');
      return (
        workspace.scrollHeight <= workspace.clientHeight + 1 &&
        document.documentElement.scrollHeight <= innerHeight
      );
    }),
    '桌面概览应完整占一屏',
  );
  const unauthorized = await fetch(url + '/api/command', {
    method: 'POST',
    body: '{"action":"start"}',
  });
  assert.equal(unauthorized.status, 403);
  checks.push('空态与请求来源/令牌拒绝');
  await page.getByRole('button', { name: '开始 / 继续', exact: true }).click();
  const secondPage = await browser.newPage();
  await secondPage.goto(url);
  const trained = await wait(url, (s) => s.updates >= 2);
  assert.ok(trained.counts.retained_samples > 0);
  await page.reload();
  await page.getByRole('button', { name: '暂停', exact: true }).click();
  const pauseAt = performance.now();
  const paused = await wait(url, (s) => s.state === 'paused');
  const pauseMs = performance.now() - pauseAt;
  assert.equal(paused.runtime.active.length, 0);
  const oldUpdates = paused.updates;
  checks.push('页面开始实际模型更新、多页面同会话、刷新继续、短边界暂停');
  await page.getByRole('button', { name: '保存恢复点', exact: true }).click();
  const saved = await wait(url, (s) => s.saved_at != null && s.state === 'paused');
  await page.getByRole('button', { name: '开始 / 继续', exact: true }).click();
  await wait(url, (s) => s.updates > oldUpdates);
  await page.getByRole('button', { name: '暂停', exact: true }).click();
  const continued = await wait(url, (s) => s.state === 'paused');
  assert.notEqual(continued.version, saved.version);
  assert.ok(continued.unsaved_updates > 0);
  checks.push('同进程继续使用新权重，页面说明未保存回退');
  first.child.kill();
  await new Promise((done) => first.child.once('exit', done));
  const restarted = await start(root);
  await page.goto(restarted.url);
  await page.getByRole('button', { name: '保存恢复点', exact: true }).click();
  const restored = await wait(restarted.url, (s) => s.restored && s.state === 'paused');
  assert.equal(restored.updates, saved.updates);
  assert.equal(restored.version, saved.version);
  assert.equal(restored.pool.samples, 0);
  checks.push('独立进程恢复最近完整点，RAM池为空，未保存进度确实回退');
  await page.getByRole('link', { name: '设置', exact: true }).click();
  await page.locator('summary').click();
  await page.locator('input[name=max_commands]').fill('1');
  await page.locator('input[name=eval_seconds]').fill('30');
  await page.getByRole('button', { name: '应用设置', exact: true }).click();
  await wait(restarted.url, (s) => s.config.max_commands === 1);
  await page.getByRole('button', { name: '立即评测', exact: true }).click();
  const evaluated = await wait(
    restarted.url,
    (s) => s.evaluations.length > 0 && s.state === 'paused',
  );
  assert.equal(evaluated.counts.retained_samples, restored.counts.retained_samples);
  assert.equal(evaluated.evaluations[0].games.length, 16);
  assert.equal(evaluated.evaluations[0].results.easy.n, 0);
  assert.equal(evaluated.evaluations[0].results.hard.errors, 0);
  checks.push('三档真实AI及独立基准、两规则交换座位16局适配；限额局显示未完成，评测不入池');
  await page.getByRole('link', { name: '评测', exact: true }).click();
  await page.screenshot({ path: resolve(output, 'evaluation.png') });
  await page.getByRole('link', { name: '资源', exact: true }).click();
  await page.screenshot({ path: resolve(output, 'resources.png') });
  await page.getByRole('link', { name: '日志', exact: true }).click();
  await page.getByRole('link', { name: '设置', exact: true }).click();
  await page.locator('input[name=max_commands]').fill('500');
  await page.getByRole('button', { name: '应用设置', exact: true }).click();
  await wait(restarted.url, (s) => s.config.max_commands === 500);
  await page.getByRole('button', { name: '立即评测', exact: true }).click();
  await page.getByRole('button', { name: '取消评测', exact: true }).click();
  const cancelled = await wait(
    restarted.url,
    (s) => s.evaluations.length === 2 && s.state === 'paused',
  );
  assert.equal(cancelled.counts.retained_samples, restored.counts.retained_samples);
  assert.ok(Object.values(cancelled.evaluations[1].results).some((r) => r.unfinished > 0));
  checks.push('立即评测后的取消按钮确实停止本轮计算，不污染训练');
  await page.getByRole('link', { name: '资源', exact: true }).click();
  await page.getByRole('button', { name: '导出模型到受管目录', exact: true }).click();
  await wait(restarted.url, (s) => s.exports.length === 1 && s.state === 'paused');
  await page.getByRole('button', { name: '新建训练', exact: true }).click();
  await page.keyboard.press('Escape');
  assert.equal(await page.locator('#new-dialog').evaluate((dialog) => dialog.open), false);
  await page.getByRole('button', { name: '新建训练', exact: true }).click();
  await page.locator('#new-form select[name=source]').selectOption('current');
  await page.locator('#new-form select[name=device]').selectOption('cpu');
  await page.getByRole('button', { name: '建立实验', exact: true }).click();
  const inherited = await wait(
    restarted.url,
    (s) => s.experiment !== 'legacy' && s.state === 'paused',
  );
  assert.equal(inherited.version, cancelled.version);
  assert.equal(inherited.updates, 0);
  assert.equal(inherited.parent.optimizer, false);
  assert.equal(inherited.evaluations.length, 0);
  assert.equal(inherited.disk.root, cancelled.disk.root);
  await page.locator('#experiment-choice').selectOption('legacy');
  await page.getByRole('button', { name: '打开所选实验', exact: true }).click();
  const reopened = await wait(
    restarted.url,
    (s) => s.experiment === 'legacy' && s.state === 'paused',
  );
  assert.equal(reopened.updates, restored.updates);
  assert.equal(reopened.version, restored.version);
  await page.locator('#experiment-choice').selectOption(inherited.experiment);
  await page.getByRole('button', { name: '删除所选非当前实验', exact: true }).click();
  await wait(restarted.url, (s) =>
    s.experiments.every((entry) => entry.id !== inherited.experiment),
  );
  await page.getByRole('button', { name: '从恢复点重新加载', exact: true }).click();
  await wait(restarted.url, (s) => s.restored && s.state === 'paused');
  checks.push(
    '页面导出、新实验继承权重但重建优化器/曲线、共享配额、打开原实验恢复、明确删除非当前实验',
  );
  await page.getByRole('link', { name: '概览', exact: true }).click();
  await page.screenshot({ path: resolve(output, 'desktop.png'), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.getByRole('link', { name: '设置', exact: true }).click();
  await page.locator('input[name=environments]').focus();
  await page.keyboard.press('Escape');
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  assert.ok(
    await page.evaluate(() =>
      [...document.querySelectorAll('*')]
        .filter((el) => {
          const css = getComputedStyle(el);
          return /(auto|scroll)/.test(css.overflowY) && el.scrollHeight > el.clientHeight;
        })
        .every((el) => el.id === 'workspace'),
    ),
    '唯一内容滚动区',
  );
  await page.screenshot({ path: resolve(output, 'mobile.png'), fullPage: true });
  await page.context().setOffline(true);
  await page.waitForFunction(() =>
    document.getElementById('connection').textContent.includes('断开'),
  );
  assert.match(await page.locator('#connection').innerText(), /状态未知/);
  await page.context().setOffline(false);
  await page.waitForFunction(() =>
    document.getElementById('connection').textContent.includes('已连接'),
  );
  checks.push('桌面概览一屏、五页面固定控制区、唯一内容滚动区、390px无横溢、键盘、断网重连');
  const pressure = await start(resolve(output, 'pressure'), 700000);
  await page.goto(pressure.url);
  await page.getByRole('button', { name: '保存恢复点', exact: true }).click();
  const stressed = await wait(pressure.url, (s) => s.state === 'error');
  assert.match(stressed.error, /配额/);
  assert.ok(stressed.disk.used <= stressed.disk.limit);
  await page.waitForFunction(() => document.getElementById('error').textContent.includes('配额'));
  checks.push('小配额写前拒绝、错误可见、实际磁盘未越界');
  assert.deepEqual(errors, []);
  await writeFile(
    resolve(output, 'report.json'),
    JSON.stringify({ checks, pauseMs, saved, restored, evaluated, pressure: stressed }, null, 2),
  );
  console.log(JSON.stringify({ output, checks, pauseMs, disk: evaluated.disk }));
} finally {
  await browser.close();
  for (const child of processes) child.kill();
}
