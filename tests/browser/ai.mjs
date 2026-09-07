import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium } from 'playwright';
const renderOnly = process.env.HAOJIE_RENDER_ONLY === '1';
await mkdir('artifacts', { recursive: true });
execFileSync(process.execPath, ['--import', 'tsx', 'tests/browser/ai-fixtures.ts']);
const html = await readFile('dist/index.html', 'utf8');
const browser = await chromium.launch({
  executablePath:
    process.env.BROWSER_PATH || (existsSync('/usr/bin/chromium') ? '/usr/bin/chromium' : undefined),
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});
const checks = [],
  errors = [],
  requests = [];
const context = await browser.newContext({
  viewport: { width: 1440, height: 1080 },
  acceptDownloads: true,
  reducedMotion: 'reduce',
});
await context.setOffline(true);
const page = await context.newPage();
page.setDefaultTimeout(10000);
page.on('pageerror', (e) => errors.push(e.message));
page.on('request', (r) => {
  if (/^https?:/.test(r.url())) requests.push(r.url());
});
const button = (name) => page.getByRole('button', { name, exact: true });
const exported = async () => {
  await page.evaluate(() => {
    window.aiExport = null;
  });
  await button('导出存档').click();
  await page.waitForFunction(() => window.aiExport !== null);
  return JSON.parse(await page.evaluate(() => window.aiExport));
};
const load = async (name) => {
  await page.locator('input[type=file]').setInputFiles(`artifacts/ai-fixtures/${name}.json`);
};
const waitForState = async (predicate, max = 200) => {
  let s;
  for (let n = 0; n < max; n++) {
    s = await exported();
    if (predicate(s)) return s;
    await page.waitForTimeout(40);
  }
  assert.ok(false, `State condition not reached: ${JSON.stringify(s?.present).slice(0, 800)}`);
};
const scenario = (label) => {
  checks.push(label);
  console.log('✓ ' + label);
};
async function installExportProbe() {
  await page.evaluate(() => {
    const original = URL.createObjectURL.bind(URL);
    URL.createObjectURL = (blob) => {
      if (blob instanceof Blob && blob.type === 'application/json')
        blob.text().then((text) => (window.aiExport = text));
      return original(blob);
    };
  });
}
try {
  if (renderOnly) await page.setContent(html);
  else await page.goto(pathToFileURL(resolve('dist/index.html')).href);
  await page.getByRole('gridcell').first().waitFor();
  await installExportProbe();
  const widths = [];
  for (const width of [900, 1090, 1100, 1279, 1280, 1440, 1680]) {
    await page.setViewportSize({ width, height: 900 });
    const box = await page.locator('.board').boundingBox();
    widths.push({ viewport: width, board: Math.round(box.width) });
    assert.equal(
      await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
      true,
      `overflow at ${width}`,
    );
  }
  for (let i = 1; i < widths.length; i++)
    assert.ok(widths[i].board >= widths[i - 1].board - 1, JSON.stringify(widths));
  assert.ok(widths.find((w) => w.viewport === 1440).board >= 620, JSON.stringify(widths));
  await writeFile('artifacts/layout-report.json', JSON.stringify(widths, null, 2));
  scenario('board never shrinks at 1090/1100/1280 breakpoints and exceeds 620px at 1440px');
  await page.setViewportSize({ width: 1440, height: 1080 });
  await page.evaluate(() => {
    const Native = window.Worker;
    window.workerMessages = [];
    window.Worker = class extends Native {
      postMessage(message, ...args) {
        window.workerMessages.push(message);
        super.postMessage(message, ...args);
      }
    };
  });
  for (const difficulty of ['easy', 'medium', 'hard']) {
    await button('新对局').click();
    await page.getByRole('radio', { name: '人机对战', exact: true }).check();
    await page.getByLabel('AI难度').selectOption(difficulty);
    await page.getByLabel('你的阵营').selectOption('2');
    await page.getByPlaceholder('例如 20260907').fill('7');
    await button('开始正式对局').click();
    const save = await waitForState((s) => s.present.active === 2 && s.present.ply === 2);
    assert.equal(save.match.mode, 'ai');
    assert.equal(save.match.difficulty, difficulty);
    assert.equal(save.match.human, 2);
    assert.ok(save.present.units.length > 0);
    assert.equal(
      save.present.hands[1].some(
        (c) => typeof c.kind === 'number' && ![8, 17, 18, 22, 25].includes(c.kind),
      ),
      false,
    );
  }
  const messages = await page.evaluate(() => window.workerMessages);
  assert.ok(messages.length > 0, 'Production build must use its inline worker, not only fallback');
  assert.ok(
    messages.every(
      (m) =>
        m.observation &&
        !('rng' in m.observation) &&
        !('seed' in m.observation) &&
        !('log' in m.observation),
    ),
  );
  scenario(
    'all three modes finish AI opening through the inline offline Worker without receiving true RNG',
  );
  await load('thinking');
  await button('暂停AI').click();
  const frozen = await exported();
  await page.waitForTimeout(550);
  assert.deepEqual(await exported(), frozen);
  await page.screenshot({ path: 'artifacts/ai-desktop.png', fullPage: true });
  await button('继续AI').click();
  await waitForState((s) => s.past.length > frozen.past.length);
  await button('暂停AI').click();
  scenario('pause cancels work, leaves state stable, and continue resumes legal operations');
  await load('response');
  const before = await exported();
  await button('结束回合').click();
  await waitForState((s) => s.present.ply >= 7 && s.present.active === 1);
  await button('悔棋').click();
  assert.deepEqual((await exported()).present, before.present);
  await page.waitForTimeout(450);
  assert.deepEqual((await exported()).present, before.present);
  await button('重做').click();
  assert.equal((await exported()).present.active, 1);
  scenario(
    'human undo removes the complete AI response, does not immediately replay it, and redo restores it',
  );
  await load('thinking');
  const auditBefore = await page.evaluate(() => window.workerMessages.length);
  await page.waitForFunction((n) => window.workerMessages.length > n, auditBefore);
  await button('新对局').click();
  await page.getByRole('radio', { name: '同屏双人', exact: true }).check();
  await button('开始正式对局').click();
  const replaced = await exported();
  await page.waitForTimeout(1200);
  assert.deepEqual(await exported(), replaced);
  scenario('opening a dialog and replacing the match invalidates in-flight worker results');
  await load('human-reaction');
  const reaction = await exported();
  assert.equal(reaction.present.pending[0].owner, 2);
  await page.waitForTimeout(450);
  assert.deepEqual(await exported(), reaction);
  await button('放弃此效果').click();
  await waitForState((s) => s.present.pending.length === 0);
  scenario('human death reactions remain human decisions even during the computer active turn');
  if (!renderOnly) {
    await load('thinking');
    await button('暂停AI').click();
    await page.reload();
    await installExportProbe();
    await waitForState((s) => s.present.active === 2);
    assert.equal((await exported()).match.difficulty, 'hard');
    scenario('file-origin refresh retains match settings and resumes the unfinished AI turn');
  }
  await button('新对局').click();
  await page.getByRole('radio', { name: '同屏双人', exact: true }).check();
  await button('开始正式对局').click();
  await page.evaluate(() => {
    window.Worker = function () {
      throw new Error('Worker unavailable in test host');
    };
  });
  await button('新对局').click();
  await page.getByRole('radio', { name: '人机对战', exact: true }).check();
  await page.getByLabel('你的阵营').selectOption('2');
  await page.getByLabel('AI难度').selectOption('easy');
  await button('开始正式对局').click();
  await waitForState((s) => s.present.active === 2);
  assert.match(await page.locator('.opponent-bar').innerText(), /分片/);
  scenario('unsupported Worker falls back to the same cooperatively scheduled planner');
  await page.setViewportSize({ width: 390, height: 844 });
  await button('新对局').click();
  assert.equal(await page.getByLabel('AI难度').count(), 1);
  await page.screenshot({ path: 'artifacts/ai-mobile.png', fullPage: true });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await button('关闭弹窗').click();
  assert.deepEqual(errors, []);
  assert.deepEqual(requests, []);
  scenario('390px setup remains usable; zero external requests and uncaught exceptions');
  await writeFile(
    'artifacts/ai-browser-report.json',
    JSON.stringify(
      {
        mode: renderOnly ? 'render-only' : 'offline-file',
        passed: checks,
        errors,
        externalRequests: requests,
      },
      null,
      2,
    ),
  );
} catch (e) {
  await page.screenshot({ path: 'artifacts/ai-failure.png', fullPage: true });
  console.log(await page.locator('.opponent-bar').innerText());
  console.log(await page.locator('.toast').allTextContents());
  throw e;
} finally {
  await browser.close();
}
