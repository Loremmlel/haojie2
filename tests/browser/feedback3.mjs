import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readFile, writeFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium } from 'playwright';
const renderOnly = process.env.HAOJIE_RENDER_ONLY === '1';
execFileSync(process.execPath, ['--import', 'tsx', 'tests/browser/feedback3-fixtures.ts']);
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
const check = (name) => {
  checks.push(name);
  console.log('✓ ' + name);
};
async function state() {
  await page.evaluate(() => (window.feedbackExport = null));
  await button('导出存档').click();
  await page.waitForFunction(() => window.feedbackExport !== null);
  return JSON.parse(await page.evaluate(() => window.feedbackExport));
}
async function load(name) {
  const path = `artifacts/feedback3-fixtures/${name}.json`;
  await page.locator('input[type=file]').setInputFiles(path);
  const expected = JSON.parse(await readFile(path, 'utf8'));
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
  await load('giant');
  await cell(2, 3).click();
  assert.ok(await page.locator('.ability-copy').isVisible());
  assert.match(await page.locator('.ability-copy').innerText(), /任意另一棋子/);
  assert.equal(await page.locator('.unit-status').getAttribute('open'), null);
  await page.locator('.unit-actions button').filter({ hasText: '巨大化' }).click();
  await cell(6, 7).click();
  await cell(5, 6).hover();
  assert.equal((await state()).present.units[1].size, 1);
  await cell(5, 6).click();
  let n = (await state()).present;
  assert.deepEqual([n.units[1].x, n.units[1].y, n.units[1].size, n.units[1].hp], [5, 6, 2, 55]);
  assert.equal(n.units[0].operations, 0);
  await button('悔棋').click();
  assert.equal((await state()).present.units[1].size, 1);
  await button('重做').click();
  assert.equal((await state()).present.units[1].size, 2);
  check(
    'enemy growth: target then expansion anchor, atomic undo/redo, description immediately visible and status collapsible',
  );
  await load('large');
  await cell(3, 5).click();
  await button('移动').click();
  await button('→ 向右一格').click();
  n = (await state()).present;
  assert.deepEqual([n.units[0].x, n.units[0].moves, n.units[0].operations], [4, 1, 0]);
  await button('→ 向右一格').click();
  n = (await state()).present;
  assert.deepEqual([n.units[0].x, n.units[0].operations], [5, 1]);
  check(
    'large-body direction controls move exactly one small cell and remain available until operation is finished',
  );
  await load('heart');
  await cell(2, 5).click();
  await button('自选穿透路径').click();
  for (const [x, y] of [
    [2, 5],
    [3, 5],
    [3, 6],
    [4, 6],
  ])
    await cell(x, y).click();
  assert.equal((await state()).present.units[1].hp, 50);
  await button('撤回一格').click();
  await cell(4, 6).click();
  assert.ok(await button('确认攻击').isEnabled());
  await page.screenshot({ path: 'artifacts/feedback3-path-desktop.png' });
  await button('确认攻击').click();
  n = (await state()).present;
  assert.deepEqual(
    n.units.slice(1).map((u) => u.hp),
    [30, 30, 50, 50],
  );
  assert.ok(n.units[1].effects.some((e) => e.type === 'burn'));
  check(
    'draw/undo/confirm a bent piercing path, preview is state-free, no friendly fire or automatic path extension',
  );
  await load('healer');
  await cell(3, 5).click();
  await button('治疗自身').click();
  assert.equal((await state()).present.units[0].hp, 25);
  check('self-heal is an explicit one-click action');
  await load('interrupt');
  await cell(2, 3).click();
  assert.ok(await page.locator('.unit-actions button').filter({ hasText: '巨大化' }).isEnabled());
  assert.equal(await button('移动').count(), 0);
  await page.locator('.unit-actions button').filter({ hasText: '巨大化' }).click();
  await cell(6, 7).click();
  await cell(5, 6).click();
  assert.equal((await state()).present.units.find((u) => u.owner === 2).size, 2);
  check('human-owned free BW can interrupt AI turn without granting other actions');
  await load('giant');
  await page.setViewportSize({ width: 390, height: 844 });
  await cell(2, 3).click();
  await page.locator('.ability-copy').scrollIntoViewIfNeeded();
  assert.ok(await page.locator('.ability-copy').isVisible());
  assert.match(await page.locator('.ability-copy').innerText(), /任意另一棋子/);
  const width = await page.evaluate(() => ({
    w: innerWidth,
    doc: document.documentElement.scrollWidth,
  }));
  assert.ok(width.doc <= width.w + 1, JSON.stringify(width));
  await page.screenshot({ path: 'artifacts/feedback3-inspector-mobile.png', fullPage: true });
  check(
    '390px mobile: skill description remains visible without disclosure and no horizontal overflow',
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(network, []);
  check('no runtime errors or external network requests');
} finally {
  await writeFile(
    'artifacts/feedback3-browser-report.json',
    JSON.stringify(
      { mode: renderOnly ? 'memory-render' : 'offline-file', checks, errors, network },
      null,
      2,
    ),
  );
  await browser.close();
}
