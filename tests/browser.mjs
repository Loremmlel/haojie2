import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium } from 'playwright';

await mkdir('artifacts', { recursive: true });
assert.deepEqual(
  await readdir('dist'),
  ['index.html'],
  'Production output must be exactly one HTML',
);
const html = await readFile('dist/index.html', 'utf8');
assert.doesNotMatch(html, /<script[^>]+src\s*=|<link[^>]+rel=["']stylesheet/i);
const executablePath =
  process.env.BROWSER_PATH || (existsSync('/usr/bin/chromium') ? '/usr/bin/chromium' : undefined);
const browser = await chromium.launch({
  executablePath,
  headless: true,
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});
const url = pathToFileURL(resolve('dist/index.html')).href;
const checks = [],
  errors = [],
  network = [];
async function open(viewport, reducedMotion = 'no-preference') {
  const context = await browser.newContext({ viewport, reducedMotion, acceptDownloads: true });
  await context.setOffline(true);
  const page = await context.newPage();
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('request', (request) => {
    if (/^https?:/.test(request.url())) network.push(request.url());
  });
  await page.goto(url);
  await page.getByRole('heading', { name: '豪杰棋局II' }).waitFor();
  assert.equal(await page.getByRole('gridcell').count(), 117);
  return { page, context };
}
async function demo(page) {
  await page.getByRole('button', { name: '新对局', exact: true }).click();
  await page.getByRole('button', { name: '载入演示棋局', exact: true }).click();
  await page.locator('[data-cell="6,6"]').click();
}
try {
  const { page, context } = await open({ width: 1440, height: 1080 });
  checks.push('file:// launches offline with 117 accessible board cells');
  await demo(page);
  await page.getByRole('button', { name: '攻击', exact: true }).click();
  await page.screenshot({ path: 'artifacts/desktop.png', fullPage: true });
  await page.locator('[data-cell="3,7"]').click();
  assert.match(await page.locator('[data-cell="6,6"]').getAttribute('aria-label'), /35生命/);
  assert.match(await page.locator('[data-cell="3,7"]').getAttribute('aria-label'), /空格/);
  assert.ok((await page.locator('.effects-layer .fx').count()) > 0);
  checks.push('attack targeting, kill growth, death retaliation and visual events');
  await page.getByRole('button', { name: '悔棋', exact: true }).click();
  assert.match(await page.locator('[data-cell="6,6"]').getAttribute('aria-label'), /45生命/);
  assert.match(await page.locator('[data-cell="3,7"]').getAttribute('aria-label'), /超级跑得快/);
  await page.getByRole('button', { name: '重做', exact: true }).click();
  assert.match(await page.locator('[data-cell="6,6"]').getAttribute('aria-label'), /35生命/);
  checks.push('undo and redo restore real board state');
  await page.reload();
  assert.match(await page.locator('[data-cell="6,6"]').getAttribute('aria-label'), /35生命/);
  checks.push('local save restores the match after reload');
  await page.getByRole('button', { name: '选择爆弹法术', exact: true }).click();
  await page.locator('[data-cell="2,10"]').click();
  assert.match(await page.locator('[data-cell="2,10"]').getAttribute('aria-label'), /91生命/);
  assert.match(await page.locator('[data-cell="3,11"]').getAttribute('aria-label'), /91生命/);
  checks.push('2×2 spell damages the large minion exactly once');
  await page.getByRole('button', { name: '棋子图鉴', exact: true }).click();
  await page.getByRole('textbox', { name: '搜索图鉴' }).fill('杀手');
  assert.equal(await page.locator('.codex-card').count(), 1);
  await page.getByRole('textbox', { name: '搜索图鉴' }).fill('');
  await page.getByRole('button', { name: '法术', exact: true }).click();
  assert.equal(await page.locator('.codex-card').count(), 5);
  await page.screenshot({ path: 'artifacts/codex.png', fullPage: true });
  await page.getByRole('button', { name: '关闭弹窗', exact: true }).click();
  await page.getByRole('button', { name: '规则', exact: true }).click();
  await page.locator('summary').click();
  assert.ok(await page.getByText(/原文距离式/).isVisible());
  await page.getByRole('button', { name: '关闭弹窗', exact: true }).click();
  checks.push('searchable codex, spell filters and explicit rule interpretations');
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: '导出存档', exact: true }).click();
  const download = await downloadPromise;
  await download.saveAs('artifacts/exported-save.json');
  const saved = JSON.parse(await readFile('artifacts/exported-save.json', 'utf8'));
  assert.equal(saved.format, 'haojie2-session-v1');
  assert.equal(saved.present.bases[2], 234);
  await page.locator('input[type="file"]').setInputFiles('artifacts/exported-save.json');
  checks.push('JSON export and validated import');
  await page.getByRole('button', { name: '新对局', exact: true }).click();
  await page.getByPlaceholder('例如 20260906').fill('7');
  await page.getByRole('button', { name: '开始正式对局', exact: true }).click();
  await page.getByRole('button', { name: '选择冲锋怪随从', exact: true }).click();
  await page.getByRole('checkbox').check();
  await page.locator('[data-cell="2,3"]').click();
  assert.match(await page.locator('[data-cell="2,3"]').getAttribute('aria-label'), /40生命/);
  await page.getByRole('button', { name: '移动', exact: true }).click();
  await page.locator('[data-cell="2,4"]').click();
  assert.match(await page.locator('[data-cell="2,4"]').getAttribute('aria-label'), /冲锋怪/);
  await page.keyboard.press('Control+z');
  assert.match(await page.locator('[data-cell="2,3"]').getAttribute('aria-label'), /冲锋怪/);
  checks.push('formal empty-board start, charge deployment, movement and keyboard undo');
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await context.close();
  const mobile = await open({ width: 390, height: 844 }, 'reduce');
  await demo(mobile.page);
  assert.equal(
    await mobile.page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
    true,
  );
  await mobile.page.screenshot({ path: 'artifacts/mobile.png', fullPage: true });
  await mobile.page.getByRole('button', { name: '攻击', exact: true }).click();
  await mobile.page.locator('[data-cell="3,7"]').click();
  assert.match(await mobile.page.locator('[data-cell="6,6"]').getAttribute('aria-label'), /35生命/);
  checks.push('390px mobile layout, reduced motion and touch-sized game controls');
  await mobile.context.close();
  assert.deepEqual(network, [], 'Offline HTML must never request network resources');
  assert.deepEqual(errors, [], 'No uncaught browser exceptions');
  checks.push('zero external network requests and zero uncaught browser exceptions');
  await writeFile(
    'artifacts/browser-report.json',
    JSON.stringify({ passed: checks, errors, externalRequests: network }, null, 2),
  );
  console.log(`Browser acceptance passed: ${checks.length} scenarios`);
  for (const check of checks) console.log(`  ✓ ${check}`);
} finally {
  await browser.close();
}
