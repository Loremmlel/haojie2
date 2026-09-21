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
  await load('hook');
  await cell(3, 4).click();
  await button('牵引').click();
  assert.ok(!(await cell(3, 6).getAttribute('aria-label')).includes('可选择'));
  checks.push('giant is not offered as an intermediate hook target');
  await page.screenshot({ path: 'artifacts/feedback4-hook.png' });
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
