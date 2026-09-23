import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readFile, writeFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium } from 'playwright';
execFileSync(process.execPath, ['--import', 'tsx', 'tests/browser/fixtures/feedback4-fixtures.ts']);
const renderOnly = process.env.HAOJIE_RENDER_ONLY === '1';
const browser = await chromium.launch({
  executablePath:
    process.env.BROWSER_PATH || (existsSync('/usr/bin/chromium') ? '/usr/bin/chromium' : undefined),
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});
const context = await browser.newContext({
  viewport: { width: 1440, height: 1080 },
  acceptDownloads: true,
  reducedMotion: 'reduce',
});
await context.setOffline(true);
const page = await context.newPage();
page.setDefaultTimeout(10000);
const errors = [],
  network = [],
  checks = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('request', (r) => {
  if (/^https?:/.test(r.url())) network.push(r.url());
});
const button = (name) => page.getByRole('button', { name, exact: true });
const cell = (x, y) => page.locator(`[data-cell="${x},${y}"]`);
async function state() {
  await page.evaluate(() => (window.feedbackExport = null));
  await button('导出存档').click();
  await page.waitForFunction(() => window.feedbackExport !== null);
  return JSON.parse(await page.evaluate(() => window.feedbackExport));
}
async function load(name) {
  await page.locator('input[type=file]').setInputFiles(`artifacts/feedback4-fixtures/${name}.json`);
  const expected = JSON.parse(await readFile(`artifacts/feedback4-fixtures/${name}.json`, 'utf8'));
  assert.deepEqual((await state()).present, expected.present);
}
try {
  if (renderOnly) await page.setContent(await readFile('dist/index.html', 'utf8'));
  else await page.goto(pathToFileURL(resolve('dist/index.html')).href);
  await page.getByRole('heading', { name: '浩劫3.0', exact: true }).waitFor();
  await page.evaluate(() => {
    const original = URL.createObjectURL.bind(URL);
    URL.createObjectURL = (blob) => {
      if (blob instanceof Blob && blob.type === 'application/json')
        blob.text().then((t) => (window.feedbackExport = t));
      return original(blob);
    };
  });
  for (const owner of [1, 2]) {
    await load(`cannon-${owner}`);
    const y = owner === 1 ? 10 : 4;
    await cell(5, y).click();
    await button('献祭射击').click();
    await cell(4, y).click();
    assert.match(await cell(5, owner === 1 ? 13 : 1).getAttribute('aria-label'), /可选择/);
    await cell(5, owner === 1 ? 13 : 1).click();
    assert.equal((await state()).present.bases[owner === 1 ? 2 : 1], 280);
    await button('悔棋').click();
    assert.equal((await state()).present.bases[owner === 1 ? 2 : 1], 300);
    checks.push(`player ${owner}: sacrifice then base-only lane, damage and undo`);
  }
  for (const [name, expected] of [
    ['row-one-one', false],
    ['row-two-zero', true],
  ]) {
    await load(name);
    await page.locator('.hand-card').click();
    await button('部署随从').click();
    assert.equal((await cell(8, 9).getAttribute('aria-label')).includes('可选择'), expected);
    if (expected) {
      await cell(8, 9).click();
      assert.ok((await state()).present.units.some((u) => u.x === 8 && u.y === 9));
    }
    checks.push(`${name}: deployment highlighter and accepted placement`);
  }
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 1080 });
    await load('live-deployment');
    await page.locator('.hand-card').click();
    await button('部署随从').click();
    assert.ok(!(await cell(8, 9).getAttribute('aria-label')).includes('可选择'));
    await page.keyboard.press('Escape');
    await cell(3, 8).click();
    await button('移动').click();
    await cell(3, 9).click();
    await page.locator('.hand-card').click();
    await button('部署随从').click();
    assert.match(await cell(8, 9).getAttribute('aria-label'), /可选择/);
    await page.screenshot({ path: `artifacts/live-deployment-${width}.png` });
    await cell(8, 9).focus();
    await page.keyboard.press('Enter');
    assert.ok((await state()).present.units.some((u) => u.x === 8 && u.y === 9));
    await button('悔棋').click();
    await button('悔棋').click();
    await page.locator('.hand-card').click();
    await button('部署随从').click();
    assert.ok(!(await cell(8, 9).getAttribute('aria-label')).includes('可选择'));
    await page.keyboard.press('Escape');
    checks.push(
      `${width}px: move unlocks deployment immediately, keyboard placement and undo restore it`,
    );
    await load('neutral-grave');
    assert.match(await cell(3, 5).getAttribute('aria-label'), /中立墓地/);
    await cell(3, 5).click();
    assert.match(await page.locator('.role-chip').innerText(), /中立/);
    assert.equal(await page.locator('.piece-wrap.p0').count(), 1);
    await page.locator('.hand-card').click();
    await button('献祭两名 · 召唤两次').click();
    assert.ok(!(await cell(3, 5).getAttribute('aria-label')).includes('可选择'));
    assert.match(await cell(5, 5).getAttribute('aria-label'), /可选择/);
    await page.screenshot({ path: `artifacts/neutral-grave-${width}.png` });
    assert.ok(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1),
    );
    await page.keyboard.press('Escape');
    checks.push(`${width}px: neutral grave is labeled and excluded from reforge`);
  }
  await page.setViewportSize({ width: 1440, height: 1080 });
  await load('hook');
  await cell(3, 4).click();
  await button('牵引').click();
  assert.ok(!(await cell(3, 6).getAttribute('aria-label')).includes('可选择'));
  checks.push('giant is not offered as an intermediate hook target');
  await page.screenshot({ path: 'artifacts/feedback4-hook.png' });
  await load('counter-status');
  await cell(3, 4).click();
  const statuses = page.getByLabel('棋子当前状态');
  await statuses.locator('summary').click();
  const weakened = statuses.locator('li').filter({ hasText: '攻击削弱' });
  assert.match(await weakened.innerText(), /攻击 -15/);
  assert.doesNotMatch(await weakened.innerText(), /\+-/);
  assert.match(
    await statuses.locator('li').filter({ hasText: '攻击强化' }).innerText(),
    /攻击 \+10/,
  );
  assert.match(
    await statuses
      .locator('li')
      .filter({ has: page.locator('b').getByRole('link', { name: '万法反制', exact: true }) })
      .innerText(),
    /冰冻中，不参与反制/,
  );
  await page.screenshot({ path: 'artifacts/counter-freeze-status.png' });
  const beforeReference = await state();
  await weakened.getByRole('link', { name: '攻击削弱', exact: true }).click();
  const keyword = page.getByRole('dialog', { name: '攻击削弱 · 状态与特性', exact: true });
  const instance = keyword.getByRole('region', { name: '本次状态' });
  assert.match(await instance.innerText(), /攻击 -15/);
  assert.match(await instance.innerText(), /记录未保存来源/);
  await keyword.getByRole('link', { name: '万法真君', exact: true }).click();
  await page
    .getByRole('dialog', { name: '万法真君', exact: true })
    .getByRole('button', { name: '返回上一介绍' })
    .click();
  await instance.waitFor();
  await page.keyboard.press('Escape');
  assert.deepEqual(await state(), beforeReference);
  checks.push('减攻与加攻分开命名，冰冻万法真君的状态明确显示反制停用');
  assert.deepEqual(errors, []);
  assert.deepEqual(network, []);
  for (const check of checks) console.log('✓ ' + check);
} finally {
  await writeFile(
    'artifacts/feedback4-browser-report.json',
    JSON.stringify(
      { mode: renderOnly ? 'memory-render' : 'offline-file', checks, errors, network },
      null,
      2,
    ),
  );
  await browser.close();
}
