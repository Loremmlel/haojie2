import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { chromium } from 'playwright';
await mkdir('artifacts', { recursive: true });
const bundle = await build({
  entryPoints: ['tests/browser/fixtures/online-host.tsx'],
  bundle: true,
  format: 'iife',
  platform: 'browser',
  target: 'es2022',
  jsx: 'automatic',
  write: false,
  outfile: 'online-host.js',
  define: { 'process.env.NODE_ENV': '"development"' },
});
const js = bundle.outputFiles.find((f) => f.path.endsWith('.js')).text;
const css = bundle.outputFiles.find((f) => f.path.endsWith('.css'))?.text ?? '';
const html = `<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Two controlled clients</title><style>body{margin:0}#player-1,#player-2{--haojie-height:950px}${css}</style><div id="root"></div><script>${js.replace(/<\/script/gi, '<\\/script')}</script></html>`;
const file = resolve('artifacts/online-host.html');
await writeFile(file, html);
const renderOnly = process.env.HAOJIE_RENDER_ONLY === '1';
const browser = await chromium.launch({
  executablePath:
    process.env.BROWSER_PATH || (existsSync('/usr/bin/chromium') ? '/usr/bin/chromium' : undefined),
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});
const context = await browser.newContext({
  viewport: { width: 1440, height: 1080 },
  reducedMotion: 'reduce',
});
await context.setOffline(true);
await context.addInitScript(() => {
  window.localAccess = [];
  for (const method of ['getItem', 'setItem', 'removeItem', 'clear']) {
    const original = Storage.prototype[method];
    Storage.prototype[method] = function (...args) {
      window.localAccess.push(method);
      return original.apply(this, args);
    };
  }
  window.workerStarts = 0;
  window.Worker = class {
    constructor() {
      window.workerStarts++;
      throw new Error('Online client must not start an AI worker');
    }
  };
});
const page = await context.newPage();
page.setDefaultTimeout(12000);
const errors = [],
  network = [],
  checks = [];
page.on('pageerror', (error) => errors.push(error.message));
page.on('request', (r) => {
  if (/^https?:/.test(r.url())) network.push(r.url());
});
const client = (p = 1) => page.locator(`#player-${p}`);
const button = (p, name) => client(p).getByRole('button', { name, exact: true });
const cell = (p, x, y) => client(p).locator(`[data-cell="${x},${y}"]`);
const status = (p = 1) => client(p).locator('.online-status');
const check = (name) => {
  checks.push(name);
  console.log('✓ ' + name);
};
async function reset(name) {
  await page.evaluate((name) => window.onlineDemo.reset(name), name);
  await status(1).filter({ hasText: '已连接' }).waitFor();
}
async function idle(p = 1) {
  await status(p).filter({ hasText: '已连接' }).waitFor();
}
async function revision(n) {
  await page.waitForFunction((n) => window.onlineDemo.revision() === n, n);
}
async function summon(p) {
  const before = await page.evaluate(() => window.onlineDemo.revision());
  await button(p, '普通召唤').click();
  await revision(before + 1);
  await idle(p);
}
async function deployAll(p) {
  for (let count = 0; count < 20; count++) {
    const next = await page.evaluate(() => window.onlineDemo.nextDeployment());
    if (!next) return;
    await client(p).locator('.hand-card').nth(next.index).click();
    const normal = button(p, '正常部署 · 不扣血');
    if (await normal.count()) await normal.click();
    else {
      const deploy = button(p, '部署随从');
      if (await deploy.count()) await deploy.click();
    }
    const before = await page.evaluate(() => window.onlineDemo.revision());
    await cell(p, next.point.x, next.point.y).click();
    await revision(before + 1);
    await idle(p);
  }
  assert.fail('fixture deployment loop exceeded its bound');
}
try {
  if (renderOnly) await page.setContent(html);
  else await page.goto(pathToFileURL(file).href);
  await page.locator('.hj-game').nth(1).waitFor();
  assert.equal(await page.locator('.hj-game').count(), 2);
  for (const name of ['新对局', '悔棋', '重做', '导出存档'])
    assert.equal(await page.getByRole('button', { name, exact: true }).count(), 0);
  assert.equal(await page.locator('input[type=file]').count(), 0);
  assert.ok(await button(2, '普通召唤').isDisabled());
  await button(1, '开启音效').click();
  await button(1, '关闭音效').click();
  await button(1, '规则').click();
  await page.getByRole('dialog').waitFor();
  await page.keyboard.press('Escape');
  await page.getByRole('dialog').waitFor({ state: 'hidden' });
  await page.evaluate(() => {
    window.firstBoard = document.querySelector('#player-1 .hj-game');
  });
  await summon(1);
  await summon(1);
  await button(1, '完成召唤，开始行动').click();
  await idle(1);
  await deployAll(1);
  await button(1, '结束回合').click();
  await idle(1);
  await summon(2);
  await summon(2);
  await button(2, '完成召唤，开始行动').click();
  await idle(2);
  await deployAll(2);
  await button(2, '结束回合').click();
  await idle(2);
  assert.equal(await page.evaluate(() => window.onlineDemo.state().active), 1);
  assert.equal(
    await page.evaluate(() => window.firstBoard === document.querySelector('#player-1 .hj-game')),
    true,
  );
  const rev = await page.evaluate(() => window.onlineDemo.revision());
  await cell(1, 5, 5).focus();
  await page.keyboard.press('Control+z');
  await page.keyboard.press('Control+y');
  assert.equal(await page.evaluate(() => window.onlineDemo.revision()), rev);
  check(
    'two JSON-view clients summon, deploy and switch turns without remount or local history controls',
  );

  for (const name of ['攻击强化', '攻击削弱']) {
    await reset('keywords');
    await cell(1, 3, 4).click();
    const statuses = client(1).getByLabel('棋子当前状态');
    await statuses.locator('summary').click();
    await statuses.getByRole('link', { name, exact: true }).click();
    const detail = client(1).getByRole('dialog', { name: `${name} · 状态与特性`, exact: true });
    const instance = detail.getByRole('region', { name: '本次状态' });
    assert.equal(await page.evaluate(() => window.onlineDemo.revision()), 0);
    await page.evaluate(() => window.onlineDemo.server(1, { type: 'end' }));
    if (name === '攻击强化') {
      await instance.getByText(/余 1 次实际回合切换/).waitFor();
      assert.match(await instance.innerText(), /攻击 \+10/);
    } else {
      await instance.getByText(/该状态已结束/).waitFor();
      assert.doesNotMatch(await instance.innerText(), /攻击 \+10/);
    }
    await page.keyboard.press('Escape');
    assert.equal(await page.evaluate(() => window.onlineDemo.revision()), 1);
  }
  check('词条读取最新公开状态，回合更新刷新计时，已结束效果不串到后续状态');

  await reset('combat');
  await page.evaluate(() => window.onlineDemo.configure({ rejectNext: true }));
  await cell(1, 3, 4).click();
  await button(1, '攻击').click();
  await cell(1, 3, 5).click();
  await client(1).locator('.toast').filter({ hasText: '测试宿主拒绝' }).waitFor();
  await idle();
  assert.equal(await page.evaluate(() => window.onlineDemo.revision()), 0);
  await button(1, '攻击').click();
  await cell(1, 3, 5).click();
  await revision(1);
  await idle();
  assert.ok(
    (await page.evaluate(
      () => window.onlineDemo.state().units.find((u) => u.owner === 2 && u.kind === 1).hp,
    )) < 500,
  );
  check('server rejection restores control; a later accepted attack updates both clients');

  await reset('combat');
  await page.evaluate(() => window.onlineDemo.configure({ holdAck: true }));
  await cell(1, 3, 4).click();
  await button(1, '攻击').click();
  await cell(1, 3, 5).click();
  await revision(1);
  await status().filter({ hasText: '等待服务器确认' }).waitFor();
  await page.evaluate(() => window.onlineDemo.server(2, window.onlineDemo.expandCommand()));
  await revision(2);
  assert.match(await status().innerText(), /等待服务器确认/);
  const count = await page.evaluate(() => window.onlineDemo.attempts().length);
  await cell(1, 3, 4).click();
  assert.equal(await client(1).locator('.unit-actions button:not([disabled])').count(), 0);
  assert.equal(await page.evaluate(() => window.onlineDemo.attempts().length), count);
  await page.evaluate(() => window.onlineDemo.flushAcks());
  await idle();
  check('own update and unrelated opponent update cannot substitute for the pending command ack');

  await reset('combat');
  await page.evaluate(() => window.onlineDemo.configure({ holdUpdates: true }));
  await cell(1, 3, 4).click();
  await button(1, '攻击').click();
  await cell(1, 3, 5).click();
  await revision(1);
  await status().filter({ hasText: '等待服务器确认' }).waitFor();
  assert.equal(await page.evaluate(() => window.onlineDemo.views()[1].revision), 0);
  await page.evaluate(() => window.onlineDemo.flushUpdates());
  await idle();
  check('ack arriving first still waits for its authoritative snapshot');

  await reset('classic');
  await summon(1);
  await client(1).locator('.summon-reveal').waitFor({ state: 'hidden' });
  await page.evaluate(() => window.onlineDemo.duplicate(1));
  assert.equal(await client(1).locator('.summon-reveal').count(), 0);
  const cards = await client(1).locator('.hand-card').count();
  await page.evaluate(() => window.onlineDemo.stale(1));
  assert.equal(await client(1).locator('.hand-card').count(), cards);
  await page.evaluate(() => window.onlineDemo.connect(1, 'disconnected'));
  assert.ok(await button(1, '普通召唤').isDisabled());
  await page.evaluate(() => window.onlineDemo.server(1, { type: 'summon' }));
  await page.evaluate(() => window.onlineDemo.connect(1, 'connected'));
  await idle();
  assert.equal(await client(1).locator('.summon-reveal').count(), 0);
  check(
    'duplicate and older updates neither rewind nor replay; reconnect restores a snapshot without summon animation',
  );

  await reset('combat');
  await page.evaluate(() => window.onlineDemo.configure({ holdAck: true, holdUpdates: true }));
  await cell(1, 3, 4).click();
  await button(1, '攻击').click();
  await cell(1, 3, 5).click();
  await revision(1);
  await page.evaluate(() => window.onlineDemo.connect(1, 'disconnected'));
  await status().filter({ hasText: '连接中断' }).waitFor();
  await page.evaluate(() => {
    window.onlineDemo.flushAcks();
    window.onlineDemo.connect(1, 'connected');
  });
  await idle();
  assert.ok(await page.evaluate(() => window.onlineDemo.aborted() > 0));
  assert.equal(await page.evaluate(() => window.onlineDemo.revision()), 1);
  check(
    'disconnect aborts waiting, ignores late receipts and reconnects without duplicating a committed command',
  );

  await reset('shrine');
  await page.evaluate(() => window.onlineDemo.configure({ holdUpdates: true }));
  await client(2).locator('.shrine-choice').first().click();
  await button(2, '锁定神龛').click();
  await revision(1);
  await client(1).locator('.shrine-choice').first().click();
  await button(1, '锁定神龛').click();
  await revision(2);
  assert.deepEqual(
    await page.evaluate(() => window.onlineDemo.attempts().map((a) => a.baseRevision)),
    [0, 0],
  );
  assert.equal(await page.evaluate(() => window.onlineDemo.state().shrineDraft.revealed), true);
  await page.evaluate(() => {
    window.onlineDemo.configure({ holdUpdates: false });
    window.onlineDemo.flushUpdates();
  });
  await idle(1);
  await idle(2);
  check(
    'both seats choose from the same revision, including non-active side first, and reveal together',
  );

  await reset('reaction');
  assert.ok(await button(1, '放弃此效果').isDisabled());
  assert.ok(await button(2, '放弃此效果').isEnabled());
  await button(2, '放弃此效果').click();
  await revision(1);
  await idle(2);
  assert.equal(await page.evaluate(() => window.onlineDemo.state().pending.length), 0);
  check('a reaction is controlled by its owner rather than the active player');

  await reset('giant');
  await cell(2, 2, 8).click();
  const giant = client(2).locator('.unit-actions button').filter({ hasText: '巨大化' });
  assert.equal(await giant.count(), 1);
  await giant.click();
  await cell(2, 5, 8).click();
  const destination = await page.evaluate(() => window.onlineDemo.expandCommand());
  await cell(2, destination.x, destination.y).click();
  await revision(1);
  await idle(2);
  assert.equal(
    await page.evaluate(
      () => window.onlineDemo.state().units.find((u) => u.owner === 1 && u.x >= 4 && u.y >= 7).size,
    ),
    2,
  );
  check('out-of-turn BW can target and expand an enemy from the controlled board');

  await reset('path');
  await cell(1, 3, 4).click();
  await button(1, '自选穿透路径').click();
  await cell(1, 3, 4).click();
  await cell(1, 3, 5).click();
  await cell(1, 4, 5).click();
  await button(1, '确认攻击').click();
  await revision(1);
  await idle();
  assert.deepEqual(await page.evaluate(() => window.onlineDemo.attempts()[0].command.path), [
    { x: 3, y: 4 },
    { x: 3, y: 5 },
    { x: 4, y: 5 },
  ]);
  check('public-only queries support the existing multi-step piercing path selector');
  const payload = JSON.stringify(await page.evaluate(() => window.onlineDemo.views()));
  assert.doesNotMatch(payload, /"(?:seed|rng|past|future)":/);
  assert.deepEqual(await page.evaluate(() => window.localAccess), []);
  assert.equal(await page.evaluate(() => window.workerStarts), 0);
  assert.deepEqual(errors, []);
  assert.deepEqual(network, []);
  check(
    'controlled clients start no AI worker, touch no local storage and receive no random/history fields',
  );
  await page.screenshot({ path: 'artifacts/online-desktop.png', fullPage: false });
  await page.setViewportSize({ width: 390, height: 844 });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.screenshot({ path: 'artifacts/online-mobile.png', fullPage: false });
  check('controlled surface stays within a 390px host viewport');
} catch (error) {
  await page.screenshot({ path: 'artifacts/online-failure.png', fullPage: true });
  console.error('Browser errors:', errors);
  console.error(
    'Host state:',
    await page.evaluate(() => window.onlineDemo?.state()).catch(() => null),
  );
  throw error;
} finally {
  await writeFile(
    'artifacts/online-browser-report.json',
    JSON.stringify({ renderOnly, checks, errors, network }, null, 2),
  );
  await browser.close();
}
console.log(`${checks.length} controlled browser scenarios passed`);
