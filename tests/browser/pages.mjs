import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { existsSync } from 'node:fs';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { chromium } from 'playwright';
import { build } from 'esbuild';

await mkdir('artifacts', { recursive: true });
execFileSync(process.execPath, ['--import', 'tsx', 'tests/browser/fixtures.ts']);
const html = await readFile('index.html', 'utf8');
assert.equal(html, await readFile('dist/index.html', 'utf8'));
assert.ok(existsSync('.nojekyll'));
const harness = await build({
  entryPoints: ['tests/browser/host.tsx'],
  bundle: true,
  write: false,
  outdir: 'artifacts/host',
  format: 'iife',
  platform: 'browser',
  define: { 'process.env.NODE_ENV': '"production"' },
});
const hostHtml = `<html><head><style>${harness.outputFiles.find((f) => f.path.endsWith('.css')).text}</style></head><body><div id="root"></div><script>${harness.outputFiles.find((f) => f.path.endsWith('.js')).text}</script></body></html>`;
let release = 'first';
const server = createServer((request, response) => {
  response.writeHead(200, {
    'content-type': 'text/html; charset=utf-8',
    'cache-control': 'no-store',
  });
  response.end(
    request.url === '/host/'
      ? hostHtml
      : html.replace('</head>', `<meta name="test-release" content="${release}"></head>`),
  );
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({
  executablePath:
    process.env.BROWSER_PATH || (existsSync('/usr/bin/chromium') ? '/usr/bin/chromium' : undefined),
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});
const checks = [],
  errors = [],
  external = [];
const context = await browser.newContext({
  viewport: { width: 1440, height: 1080 },
  acceptDownloads: true,
  reducedMotion: 'reduce',
});
context.on('page', (p) => {
  p.on('pageerror', (e) => errors.push(e.message));
  p.on('request', (req) => {
    if (/^https?:/.test(req.url()) && !req.url().startsWith(origin + '/')) external.push(req.url());
  });
});
let page = await context.newPage();
page.setDefaultTimeout(10000);
const exported = async () => {
  const waiting = page.waitForEvent('download');
  await page.getByRole('button', { name: '导出存档', exact: true }).click();
  const download = await waiting;
  await download.saveAs('artifacts/pages-save.json');
  return JSON.parse(await readFile('artifacts/pages-save.json', 'utf8'));
};
try {
  await page.goto(origin + '/haojie2/');
  await page.getByRole('gridcell').first().waitFor();
  assert.equal(await page.getByRole('gridcell').count(), 117);
  await page.locator('input[type=file]').setInputFiles('artifacts/fixtures/archer.json');
  const first = await exported();
  assert.equal(first.present.phase, 'play');
  await page.locator('[data-cell="3,4"]').click();
  await page.getByRole('button', { name: '攻击', exact: true }).click();
  await page.locator('[data-cell="3,6"]').click();
  const playing = await exported();
  assert.ok(playing.past.length > 0);
  checks.push('main/root HTML serves the game at a project subpath, not README');
  release = 'second';
  await page.close();
  page = await context.newPage();
  page.setDefaultTimeout(10000);
  await page.goto(origin + '/haojie2/');
  await page.getByRole('gridcell').first().waitFor();
  assert.equal(await page.locator('meta[name=test-release]').getAttribute('content'), 'second');
  assert.deepEqual(await exported(), playing);
  assert.match(await page.locator('[data-cell="3,6"]').getAttribute('aria-label'), /60生命/);
  await page.getByRole('button', { name: '悔棋', exact: true }).click();
  assert.deepEqual((await exported()).present, first.present);
  checks.push('replacement HTML and a reopened tab retain the full match and undo history');
  const raw = '{broken-old-save';
  await page.evaluate((raw) => localStorage.setItem('haojie.session.v2', raw), raw);
  await page.reload();
  await page.getByRole('heading', { name: '浩劫2.0', exact: true }).waitFor();
  assert.equal(await page.evaluate(() => localStorage.getItem('haojie.session.v2')), raw);
  assert.match(await page.locator('.log-footer').innerText(), /暂停/);
  await page.getByRole('button', { name: '新对局', exact: true }).click();
  await page.getByRole('button', { name: '开始正式对局', exact: true }).click();
  const newSave = await exported();
  assert.equal(
    JSON.parse(await page.evaluate(() => localStorage.getItem('haojie.session.v2'))).format,
    newSave.format,
  );
  checks.push('unreadable stored data is not overwritten until the player explicitly starts anew');
  await page.goto(origin + '/host/');
  await page.waitForFunction(() => window.testStates?.length === 2);
  const games = page.locator('.hj-game'),
    left = games.nth(0),
    right = games.nth(1);
  assert.equal(await games.count(), 2);
  const initial = await page.evaluate(() => window.testStates);
  await left.locator('[data-cell="3,5"]').click();
  await left.getByRole('button', { name: '移动', exact: true }).click();
  await left.locator('[data-cell="2,5"]').click();
  const moved = await page.evaluate(() => window.testStates);
  assert.notDeepEqual(moved[0], initial[0]);
  assert.deepEqual(moved[1], initial[1]);
  await right.locator('[data-cell="3,5"]').focus();
  await page.keyboard.press('Control+z');
  assert.deepEqual(await page.evaluate(() => window.testStates), moved);
  await page.locator('#host-button').focus();
  await page.keyboard.press('Control+z');
  assert.deepEqual(await page.evaluate(() => window.testStates), moved);
  await left.getByRole('button', { name: '悔棋', exact: true }).focus();
  await page.keyboard.press('Control+z');
  assert.deepEqual(await page.evaluate(() => window.testStates), initial);
  checks.push('two embedded games keep state and shortcuts isolated from each other and the host');
  assert.equal(
    await page.locator('#host-button').evaluate((el) => getComputedStyle(el).borderRadius),
    '0px',
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(external, []);
  checks.push('embedded styles stay scoped; no unexpected external requests or uncaught errors');
  await writeFile(
    'artifacts/pages-report.json',
    JSON.stringify(
      {
        mode: 'http-same-origin-and-embedding',
        passed: checks,
        errors,
        externalRequests: external,
      },
      null,
      2,
    ),
  );
  console.log(checks.map((c) => '✓ ' + c).join('\n'));
} finally {
  await browser.close();
  await new Promise((resolve) => server.close(resolve));
}
