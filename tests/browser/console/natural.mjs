import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { chromium } from 'playwright';

const url = process.env.CONSOLE_URL || 'http://127.0.0.1:8767';
const phase = process.env.NATURAL_PHASE || 'full';
assert.ok(['full', 'round', 'train', 'evaluation', 'compare'].includes(phase));
const output = resolve('artifacts/mc-q-delivery');
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ executablePath: process.env.BROWSER_PATH, headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
const status = async () => (await fetch(url + '/api/status')).json();
async function wait(predicate, minutes = 90) {
  const deadline = Date.now() + minutes * 60000;
  let printed = 0;
  while (Date.now() < deadline) {
    const s = await status();
    assert.notEqual(s.state, 'error', s.error);
    if (predicate(s)) return s;
    if (Date.now() - printed > 60000) {
      console.log(
        JSON.stringify({
          state: s.state,
          updates: s.updates,
          generation: s.generation,
          games: s.counts.games,
          evaluation: s.evaluation_job,
          active: s.runtime.active,
        }),
      );
      printed = Date.now();
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw Error('自然验收达到明确时间上限');
}
async function save(name, value) {
  await writeFile(resolve(output, name + '.json'), JSON.stringify(value, null, 2));
}
try {
  await page.goto(url);
  await page.waitForFunction(() =>
    document.getElementById('connection').textContent.includes('已连接'),
  );
  if (phase === 'train' && (await status()).state === 'idle') {
    const saved = JSON.parse(await readFile(resolve(output, 'natural-round1.json'), 'utf8'));
    await page.getByRole('link', { name: '资源', exact: true }).click();
    await page.getByRole('button', { name: '从恢复点重新加载', exact: true }).click();
    const restored = await wait((s) => s.restored && s.state === 'paused', 3);
    assert.equal(restored.updates, saved.updates);
    assert.equal(restored.version, saved.version);
    assert.equal(restored.pool.samples, 0);
    assert.equal(restored.execution.updating, saved.execution.updating);
    await save('natural-restored', restored);
    await page.getByRole('link', { name: '设置', exact: true }).click();
    await page.locator('summary').click();
    await page.locator('input[name=eval_games]').fill('4');
    await page.getByRole('button', { name: '应用设置', exact: true }).click();
    await wait((s) => s.config.eval_games === 4, 1);
  }
  if (phase === 'compare' && (await status()).state === 'idle') {
    const saved = JSON.parse(await readFile(resolve(output, 'natural-continued.json'), 'utf8'));
    await page.getByRole('link', { name: '资源', exact: true }).click();
    await page.getByRole('button', { name: '从恢复点重新加载', exact: true }).click();
    const restored = await wait((s) => s.restored && s.state === 'paused', 3);
    assert.equal(restored.updates, saved.updates);
    assert.equal(restored.version, saved.version);
    assert.equal(restored.pool.consumed, saved.pool.consumed);
    await save('natural-final-restored', restored);
    await page.getByRole('button', { name: '开始 / 继续', exact: true }).click();
    const periodic = await wait((s) => s.evaluation_job?.completed_tasks >= 1, 10);
    assert.equal(periodic.evaluation_job.model, restored.version);
    assert.equal(periodic.counts.tasks, restored.counts.tasks);
    await save('natural-periodic-triggered', periodic);
    await page.getByRole('button', { name: '暂停', exact: true }).click();
    await wait((s) => s.state === 'paused' && !s.evaluation_job, 2);
    await page.getByRole('button', { name: '开始 / 继续', exact: true }).click();
    const continued = await wait((s) => s.runtime.active.some((g) => g.commands >= 2), 3);
    assert.equal(continued.version, restored.version);
    assert.equal(continued.updates, restored.updates);
    await save('natural-memory-resumed-final', continued);
    await page.getByRole('button', { name: '暂停', exact: true }).click();
    await wait((s) => s.state === 'paused', 2);
    const savedAt = (await status()).saved_at;
    await page.getByRole('button', { name: '保存恢复点', exact: true }).click();
    await wait((s) => s.saved_at > savedAt && s.state === 'paused', 2);
  }
  if (phase === 'full') {
    await page.getByRole('button', { name: '新建训练', exact: true }).click();
    await page.getByRole('button', { name: '建立实验', exact: true }).click();
    await wait((s) => s.version && s.state === 'paused', 3);
    await page.screenshot({ path: resolve(output, 'ready.png') });
    await page.getByRole('button', { name: '立即评测', exact: true }).click();
    const before = await wait((s) => s.evaluations.length === 1 && !s.evaluation_job);
    await save('natural-before', before);
  }
  if (['full', 'round', 'train'].includes(phase)) {
    if (['idle', 'paused'].includes((await status()).state))
      await page.getByRole('button', { name: '开始 / 继续', exact: true }).click();
    const second = await browser.newPage();
    await second.goto(url);
    await page.reload();
    await wait(
      (s) =>
        s.generation >= (phase === 'round' ? 1 : 2) &&
        Object.values(s.counts.completed_matchups ?? {}).reduce((a, b) => a + b, 0) >=
          (phase === 'round' ? 4 : 8),
    );
    await page.getByRole('button', { name: '暂停', exact: true }).click();
    const began = performance.now();
    const trained = await wait((s) => s.state === 'paused', 1);
    trained.acceptance_pause_ms = performance.now() - began;
    await save(phase === 'round' ? 'natural-round1' : 'natural-trained', trained);
    await page.getByRole('button', { name: '保存恢复点', exact: true }).click();
    await wait((s) => s.saved_step === trained.updates && s.state === 'paused', 2);
    if (phase !== 'round') {
      await page.getByRole('button', { name: '开始 / 继续', exact: true }).click();
      const resumed = await wait((s) => s.state === 'sampling' && s.runtime.active.length > 0);
      assert.equal(resumed.version, trained.version);
      assert.equal(resumed.updates, trained.updates);
      assert.ok(resumed.counts.tasks > trained.counts.tasks);
      await save('natural-memory-resumed', resumed);
      await page.getByRole('button', { name: '暂停', exact: true }).click();
      const continued = await wait((s) => s.state === 'paused', 1);
      await save('natural-continued', continued);
      await page.getByRole('button', { name: '保存恢复点', exact: true }).click();
      await wait((s) => s.saved_step === continued.updates && s.state === 'paused', 2);
    }
  }
  if (['full', 'evaluation', 'compare'].includes(phase)) {
    const count = (await status()).evaluations.length;
    await page.getByRole('button', { name: '立即评测', exact: true }).click();
    const after = await wait((s) => s.evaluations.length > count && !s.evaluation_job);
    await save(process.env.NATURAL_REPORT || 'natural-after', after);
    await page.getByRole('link', { name: '评测', exact: true }).click();
    await page.screenshot({ path: resolve(output, 'natural-evaluation.png') });
    await page.getByRole('link', { name: '资源', exact: true }).click();
    await page.screenshot({ path: resolve(output, 'natural-resources.png') });
  }
  if (phase === 'compare') {
    const trained = await status();
    const report = trained.evaluations.at(-1);
    assert.equal(trained.candidate.model, trained.version);
    await page.getByRole('link', { name: '评测', exact: true }).click();
    await page.getByRole('button', { name: '确认冻结候选', exact: true }).click();
    const confirming = await wait((s) => s.evaluation_job?.confirmation, 2);
    assert.equal(confirming.evaluation_job.model, trained.version);
    assert.equal(confirming.evaluation_job.planned, 80);
    await save('natural-confirm-started', confirming);
    await page.getByRole('button', { name: '取消评测', exact: true }).click();
    const cancelled = await wait((s) => !s.evaluation_job && s.state === 'paused', 2);
    assert.equal(cancelled.updates, trained.updates);
    await save('natural-confirm-cancelled', cancelled);
    await page.getByRole('link', { name: '资源', exact: true }).click();
    await page.getByRole('button', { name: '导出模型到受管目录', exact: true }).click();
    await save('natural-exported', await wait((s) => s.exports.length === 1, 2));
    await page.getByRole('button', { name: '新建训练', exact: true }).click();
    await page.getByRole('button', { name: '建立实验', exact: true }).click();
    const control = await wait(
      (s) => s.experiment !== trained.experiment && s.state === 'paused',
      3,
    );
    const original = JSON.parse(await readFile(resolve(output, 'natural-before.json'), 'utf8'));
    assert.equal(control.version, original.version);
    assert.equal(control.updates, 0);
    await page.getByRole('button', { name: '立即评测', exact: true }).click();
    const evaluated = await wait((s) => s.evaluations.length === 1 && !s.evaluation_job);
    assert.equal(evaluated.evaluations[0].series, report.series);
    await save('natural-control-final', evaluated);
    await page.getByRole('link', { name: '资源', exact: true }).click();
    await page.locator('#experiment-choice').selectOption(trained.experiment);
    await page.getByRole('button', { name: '打开所选实验', exact: true }).click();
    const reopened = await wait(
      (s) => s.experiment === trained.experiment && s.state === 'paused',
      3,
    );
    assert.equal(reopened.version, trained.version);
    assert.equal(reopened.updates, trained.updates);
    assert.ok(reopened.evaluations.some((r) => r.time === report.time));
    await save('natural-final-paused', reopened);
  }
  assert.equal(errors.length, 0, errors.join('\n'));
  const final = await status();
  console.log(
    JSON.stringify({ complete: true, phase, updates: final.updates, games: final.counts.games }),
  );
} finally {
  await browser.close();
}
