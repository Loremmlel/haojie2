import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium } from 'playwright';

const renderOnly = process.env.HAOJIE_RENDER_ONLY === '1';
await mkdir('artifacts', { recursive: true });
execFileSync(process.execPath, ['scripts/vfx/build-preview.mjs']);
execFileSync(process.execPath, ['--import', 'tsx', 'tests/browser/fixtures.ts']);
const browser = await chromium.launch({
  executablePath:
    process.env.BROWSER_PATH || (existsSync('/usr/bin/chromium') ? '/usr/bin/chromium' : undefined),
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});
const errors = [],
  network = [],
  checks = [];
const context = await browser.newContext({
  viewport: { width: 1280, height: 1000 },
  reducedMotion: 'no-preference',
});
await context.setOffline(true);
const page = await context.newPage();
page.setDefaultTimeout(7000);
page.on('pageerror', (error) => errors.push(error.message));
page.on('request', (r) => {
  if (/^https?:/.test(r.url())) network.push(r.url());
});
const button = (name) => page.getByRole('button', { name, exact: true });
const check = (name) => {
  checks.push(name);
  console.log(`✓ ${name}`);
};
async function open(path) {
  if (renderOnly) await page.setContent(await readFile(path, 'utf8'));
  else await page.goto(pathToFileURL(resolve(path)).href);
}
async function sample(name) {
  await button(name).click();
  await button('播放动作').click();
}
try {
  await open('artifacts/vfx-preview.html');
  await sample('射手箭矢');
  assert.equal(await page.locator('[data-fx=arrow]').count(), 1);
  const flight = () =>
    page.locator('[data-fx=arrow] animateMotion').evaluate((node) => {
      const m = node.parentElement.getCTM();
      return { x: m.e, y: m.f };
    });
  const p0 = await flight();
  await page.waitForTimeout(140);
  const p1 = await flight();
  assert.ok(
    Math.hypot(p1.x - p0.x, p1.y - p0.y) > 5,
    'projectile must actually move, not finish before insertion',
  );
  // Freezing is only for review screenshots, never an assertion about exact pixels.
  await page.evaluate(() => {
    document.querySelector('.effects-layer').pauseAnimations();
    for (const a of document.getAnimations()) a.pause();
  });
  await page.screenshot({ path: 'artifacts/vfx-arrow.png', fullPage: true });
  await page.evaluate(() => document.querySelector('.effects-layer').unpauseAnimations());
  check('arrow follows a moving SVG path even when mounted into a persistent SVG');
  for (const [name, kind, stage] of [
    ['近身斩击', 'slash'],
    ['赤焰重炮', 'cannon'],
    ['爆弹区域', 'bomb'],
    ['钩子牵引', 'hook'],
    ['策反施加', 'conversion', 'apply'],
    ['策反兑现', 'conversion', 'trigger'],
    ['处决施加', 'execution', 'apply'],
    ['处决触发', 'execution', 'trigger'],
    ['金身挡下', 'ward', 'blocked'],
    ['烈焰风暴', 'storm'],
    ['风暴再临', 'storm'],
    ['十字浩劫', 'cross'],
    ['法术反制', 'counter', 'blocked'],
  ]) {
    await sample(name);
    assert.ok(
      await page.locator(`[data-fx=${kind}]${stage ? `[data-stage=${stage}]` : ''}`).count(),
    );
    if (name === '钩子牵引') {
      assert.equal(
        await page.locator('[data-fx=hook] .vfx-contact').getAttribute('transform'),
        'translate(450 450)',
        'hook contact stays at the old victim square',
      );
      assert.equal(
        await page.locator('[data-fx=hook] .vfx-arrival').getAttribute('transform'),
        'translate(350 550)',
        'arrival is a distinct marker at the completed relocation square',
      );
    }
    if (name === '法术反制')
      assert.equal(await page.locator('[data-fx=bomb], [data-fx=damage]').count(), 0);
    if (name === '金身挡下') assert.equal(await page.locator('[data-fx=damage]').count(), 0);
    if (['赤焰重炮', '爆弹区域', '策反施加', '烈焰风暴', '十字浩劫'].includes(name)) {
      await page.waitForTimeout(name === '赤焰重炮' ? 340 : 200);
      await page.screenshot({ path: `artifacts/vfx-${kind}.png`, fullPage: true });
    }
  }
  check(
    'representative real-rule actions distinguish range, shape, pending seals, successful triggers and blocks',
  );
  // Click bursts programmatically to avoid Playwright actionability waiting on the animation itself.
  await button('近身斩击').click();
  await button('播放动作').evaluate((b) => {
    b.click();
    b.click();
  });
  assert.equal(await page.locator('[data-fx-batch]').count(), 2);
  await button('播放动作').evaluate((b) => {
    for (let n = 0; n < 30; n++) b.click();
  });
  assert.ok((await page.locator('[data-fx-batch]').count()) <= 4);
  assert.ok((await page.locator('[data-fx]').count()) <= 96);
  await button('复位 / 取消').click();
  assert.equal(await page.locator('[data-fx]').count(), 0);
  check('bursts coexist, have bounded work, and cancel without stale effects');
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await sample('射手箭矢');
  assert.equal(await page.locator('animateMotion').count(), 0);
  assert.ok(await page.locator('.vfx-number').isVisible());
  const animations = await page
    .locator('.board')
    .evaluate(
      (node) =>
        node.getAnimations({ subtree: true }).filter((a) => a.playState === 'running').length,
    );
  assert.equal(animations, 0);
  await page.setViewportSize({ width: 390, height: 844 });
  await sample('爆弹区域');
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.screenshot({ path: 'artifacts/vfx-mobile-reduced.png', fullPage: true });
  check('reduced motion retains numbers and area, has no moving projectiles, and fits mobile');

  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await page.setViewportSize({ width: 1440, height: 1080 });
  await open('dist/index.html');
  await page.locator('input[type=file]').setInputFiles('artifacts/fixtures/archer.json');
  const cell = (x, y) => page.locator(`[data-cell="${x},${y}"]`);
  await cell(3, 4).click();
  await button('攻击').click();
  await cell(3, 6).click();
  assert.match(await cell(3, 6).getAttribute('aria-label'), /60生命/);
  const first = await page.locator('[data-fx-batch]').first().getAttribute('data-fx-batch');
  await cell(5, 4).click();
  assert.equal(
    await page.locator(`[data-fx-batch="${first}"]`).count(),
    1,
    'second shot must not erase first shot',
  );
  assert.match(await cell(5, 4).getAttribute('aria-label'), /60生命/);
  await button('悔棋').click();
  assert.equal(await page.locator('[data-fx]').count(), 0);
  assert.match(await cell(5, 4).getAttribute('aria-label'), /70生命/);
  await button('重做').click();
  assert.equal(
    await page.locator('[data-fx]').count(),
    0,
    'redo replaces a snapshot, not an old animation',
  );
  await page.locator('input[type=file]').setInputFiles('artifacts/fixtures/archer.json');
  assert.equal(await page.locator('[data-fx]').count(), 0);
  await cell(3, 4).click();
  await button('攻击').click();
  await cell(3, 6).click();
  await page.waitForTimeout(1800);
  assert.equal(
    await page.locator('[data-fx]').count(),
    0,
    'independent expiry releases DOM without further input',
  );
  check(
    'production commands settle immediately; second shot, undo, redo, import and expiry are safe',
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(network, []);
  check('no uncaught errors or external requests');
  await writeFile(
    'artifacts/vfx-report.json',
    JSON.stringify(
      {
        mode: renderOnly ? 'in-memory-render-only' : 'offline-file-url',
        passed: checks,
        errors,
        externalRequests: network,
      },
      null,
      2,
    ),
  );
} finally {
  await browser.close();
}
