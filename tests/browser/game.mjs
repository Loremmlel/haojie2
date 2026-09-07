import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium } from 'playwright';

const renderOnly = process.env.HAOJIE_RENDER_ONLY === '1';
await mkdir('artifacts', { recursive: true });
execFileSync(process.execPath, ['--import', 'tsx', 'tests/browser/fixtures.ts'], {
  stdio: 'inherit',
});
assert.deepEqual(await readdir('dist'), ['index.html']);
const html = await readFile('dist/index.html', 'utf8');
assert.doesNotMatch(html, /<script[^>]+src\s*=|<link[^>]+rel=["']stylesheet/i);
const b = await chromium.launch({
  executablePath:
    process.env.BROWSER_PATH || (existsSync('/usr/bin/chromium') ? '/usr/bin/chromium' : undefined),
  headless: true,
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});
const checks = [],
  errors = [],
  network = [];
let page;
const button = (name) => page.getByRole('button', { name, exact: true });
const cell = (x, y) => page.locator(`[data-cell="${x},${y}"]`);
async function choose(x, y) {
  await cell(x, y).click();
}
async function load(name) {
  const expected = JSON.parse(await readFile(`artifacts/fixtures/${name}.json`, 'utf8')).present;
  await page.locator('input[type=file]').setInputFiles(`artifacts/fixtures/${name}.json`);
  for (let n = 0; n < 8; n++) {
    const actual = await readState();
    try {
      assert.deepEqual(actual, expected);
      return;
    } catch {
      if (n === 7) assert.deepEqual(actual, expected);
      await new Promise((r) => setTimeout(r, 80));
    }
  }
}
async function readState() {
  const waiting = page.waitForEvent('download');
  await button('导出存档').click();
  const d = await waiting;
  await d.saveAs('artifacts/browser-save.json');
  return JSON.parse(await readFile('artifacts/browser-save.json', 'utf8')).present;
}
async function cancel() {
  if (await button('取消当前操作').count()) await button('取消当前操作').click();
}
async function dismiss() {
  if (await button('关闭提示').count()) await button('关闭提示').click();
}
const scenario = (name) => {
  checks.push(name);
  console.log(`✓ ${name}`);
};
async function open(viewport, reducedMotion = 'no-preference') {
  const context = await b.newContext({ viewport, reducedMotion, acceptDownloads: true });
  await context.setOffline(true);
  const p = await context.newPage();
  p.setDefaultTimeout(7000);
  p.on('pageerror', (e) => errors.push(e.message));
  p.on('request', (r) => {
    if (/^https?:/.test(r.url())) network.push(r.url());
  });
  if (renderOnly) await p.setContent(html);
  else await p.goto(pathToFileURL(resolve('dist/index.html')).href);
  await p.getByRole('heading', { name: '浩劫2.0', exact: true }).waitFor();
  return { p, context };
}
try {
  const desktop = await open({ width: 1440, height: 1080 });
  page = desktop.p;
  assert.equal(await page.getByRole('gridcell').count(), 117);
  assert.equal(await button('普通召唤').count(), 1);
  scenario(
    renderOnly
      ? 'in-memory rendering of the production HTML (not a file:// acceptance claim)'
      : 'file:// production HTML launches with network offline',
  );
  await button('新对局').click();
  await page.getByPlaceholder('例如 20260907').fill('7');
  await button('开始正式对局').click();
  await button('普通召唤').click();
  await button('普通召唤').click();
  await button('完成召唤，开始行动').click();
  await button('选择冲锋怪随从').click();
  await page.getByRole('checkbox').check();
  await choose(2, 3);
  assert.match(await cell(2, 3).getAttribute('aria-label'), /40生命/);
  await button('移动').click();
  await choose(2, 4);
  assert.match(await cell(2, 4).getAttribute('aria-label'), /冲锋怪/);
  await page.keyboard.press('Control+z');
  assert.match(await cell(2, 3).getAttribute('aria-label'), /冲锋怪/);
  await button('重做').click();
  assert.match(await cell(2, 4).getAttribute('aria-label'), /冲锋怪/);
  scenario('formal staged summons, charge deployment, movement, keyboard undo and redo');
  await load('summon');
  await page.getByRole('button', { name: /^终极召唤/ }).click();
  let state = await readState();
  assert.equal(state.heads[1], 2);
  assert.equal(state.summonSlots, 1);
  assert.ok(state.hands[1].every((c) => String(c.kind).startsWith('u')));
  await button('悔棋').click();
  assert.equal((await readState()).heads[1], 4);
  await button('重做').click();
  assert.deepEqual((await readState()).hands, state.hands);
  scenario('ultimate summon pays before reveal and undo restores its currency and RNG result');
  await load('archer');
  await choose(3, 4);
  await button('攻击').click();
  await choose(3, 6);
  assert.equal(await button('移动').isDisabled(), true);
  assert.match(await cell(3, 6).getAttribute('aria-label'), /60生命/);
  await choose(5, 4);
  assert.match(await cell(5, 4).getAttribute('aria-label'), /60生命/);
  assert.equal(await button('攻击').isDisabled(), true);
  scenario('attack mode locks out movement while preserving the remaining attack');
  await load('guardian');
  await choose(3, 4);
  assert.equal(await button('移动').isDisabled(), true);
  await button('蓄力 · 移动').click();
  state = await readState();
  assert.equal(state.units[0].charge, 1);
  assert.equal(state.units[0].operations, 1);
  scenario('zero-attack guardian has a real charge-move operation');
  await load('staff');
  await button('选择寒冰法杖武器').click();
  await choose(4, 4);
  await choose(4, 4);
  await button('攻击').click();
  await choose(4, 6);
  state = await readState();
  assert.deepEqual(state.units.find((u) => u.kind === 'u6').equipment, ['u5']);
  assert.equal(state.units.find((u) => u.kind === 26).hp, 35);
  assert.ok(state.units.find((u) => u.kind === 26).effects.some((e) => e.type === 'freeze'));
  assert.match(await cell(4, 6).getAttribute('aria-label'), /中立/);
  scenario('mage-only weapon equip, on-hit frost and neutral battlefield presentation');
  await load('storm-clones');
  await button('选择烈焰风暴法术').click();
  await button('烈焰风暴 · 横排').click();
  await choose(5, 7);
  state = await readState();
  assert.equal(state.units.length, 0);
  assert.equal(state.heads[1], 7);
  assert.equal(state.hazards.length, 1);
  assert.equal(await page.locator('.hazard-row').count(), 1);
  scenario('firestorm affects every clone in a stack and awards one head for the entire batch');
  await load('revive');
  await choose(3, 4);
  await button('复活').click();
  await page.getByRole('button', { name: /杀手.*第2轮/ }).click();
  await choose(4, 4);
  state = await readState();
  assert.equal(state.units.find((u) => u.kind === 26).hp, 45);
  assert.equal(state.units.find((u) => u.kind === 'u19').maxHp, 5);
  assert.ok(state.deaths[0].revived);
  scenario('revival picker, two-step targeting, pristine revival and maximum-health payment');
  await load('siphon');
  await choose(3, 5);
  await button('虹吸 · 免费').click();
  await choose(5, 5);
  await choose(2, 5);
  assert.equal(await page.locator('.siphon-thread').count(), 1);
  await button('结束回合').click();
  state = await readState();
  assert.equal(state.units.find((u) => u.kind === 'grave').hp, 50);
  assert.equal(state.units.find((u) => u.kind === 1).hp, 30);
  scenario('two-target free siphon, persistent link and end-turn transfer');
  await load('runner');
  await choose(3, 4);
  await button('移动').click();
  await choose(3, 5);
  assert.ok((await readState()).pending.some((r) => r.kind === 'bounce'));
  await choose(4, 5);
  state = await readState();
  assert.equal(state.units.find((u) => u.kind === 'u12').moves, 4);
  assert.equal(state.pending.length, 0);
  await button('悔棋').click();
  assert.equal((await readState()).pending[0].kind, 'bounce');
  await choose(4, 5);
  scenario('mandatory SZF bounce uses no movement point and remains undoable');
  await load('horn');
  await button('选择冲锋号令法术').click();
  await choose(3, 4);
  await choose(3, 4);
  await button('攻击').click();
  await choose(3, 6);
  state = await readState();
  assert.equal(
    state.units.some((u) => u.kind === 5),
    false,
  );
  assert.equal(state.units[0].offset, 2);
  scenario('haste advances only one unit and activates its scheduled execution effect');
  await load('counter');
  await button('选择金身法术').click();
  await choose(3, 4);
  state = await readState();
  assert.equal(state.hands[1].length, 0);
  assert.equal(state.units.find((u) => u.kind === 1).effects.length, 0);
  assert.ok(state.events.some((e) => e.text === '法术反制'));
  scenario('enemy counterspell consumes the spell without applying its effect');
  await load('cross');
  await choose(3, 4);
  await button('十字浩劫 · 一生一次').click();
  await choose(5, 5);
  state = await readState();
  assert.equal(state.units.find((u) => u.x === 5 && u.y === 5).hp, 30);
  assert.equal(state.units.find((u) => u.x === 5 && u.y === 6).hp, 50);
  assert.ok(state.units.find((u) => u.kind === 'u6').onceUsed);
  scenario('charged cross attack distinguishes center damage from arm damage');
  await load('craft');
  await page.getByRole('button', { name: /3枚炎魔之心/ }).click();
  await button('选择炎魔之王随从').click();
  await choose(2, 4);
  state = await readState();
  assert.equal(state.hands[1].length, 0);
  assert.equal(state.units[0].kind, 'firelord');
  scenario('three stored hearts synthesize a deployable firelord');
  await load('reroll');
  await button('选择冲锋怪随从').click();
  await page.getByRole('button', { name: '改判 · (3,4)', exact: true }).click();
  state = await readState();
  assert.equal(state.units[0].freeUsed, state.ply);
  assert.equal(state.hands[1][0].rerolled, true);
  assert.equal(state.summonSlots, 1);
  scenario('reroll replaces a revealed summon without consuming an additional summon slot');
  await load('clone-control');
  await choose(3, 4);
  await button('移动').click();
  await choose(3, 5);
  await choose(3, 4);
  await button('移动').click();
  await choose(4, 4);
  state = await readState();
  assert.equal(state.units.filter((u) => u.operations === 1).length, 2);
  scenario('individual allied clones remain separately selectable and movable');
  await button('棋子图鉴').click();
  assert.equal(await page.locator('.codex-card').count(), 59);
  await page.getByRole('textbox', { name: '搜索图鉴' }).fill('免疫塔');
  assert.ok(await page.getByRole('heading', { name: '免疫塔', exact: true }).isVisible());
  await page.getByRole('textbox', { name: '搜索图鉴' }).fill('');
  await button('终极').click();
  assert.equal(await page.locator('.codex-card').count(), 28);
  await button('武器').click();
  assert.equal(await page.locator('.codex-card').count(), 4);
  await page.screenshot({ path: 'artifacts/codex.png', fullPage: true });
  await button('关闭弹窗').click();
  await button('规则').click();
  await page.locator('.interpretations summary').click();
  assert.ok(
    await page
      .getByText(/善铁.*3.0神龛模式/)
      .first()
      .isVisible(),
  );
  await button('关闭弹窗').click();
  scenario(
    '59-entry searchable codex, 28 ultimate entries, four weapons and confirmed/deferred rules',
  );
  const beforeInvalid = await readState();
  await page.locator('input[type=file]').setInputFiles({
    name: 'bad.json',
    mimeType: 'application/json',
    buffer: Buffer.from('{oops'),
  });
  assert.deepEqual(await readState(), beforeInvalid);
  if (!renderOnly) {
    await page.reload();
    await page.getByRole('heading', { name: '浩劫2.0', exact: true }).waitFor();
    assert.deepEqual(await readState(), beforeInvalid);
    scenario('file:// local save reloads and malformed imports never replace the current match');
  } else
    scenario(
      'JSON export and malformed import rejection (origin persistence deferred to file:// CI)',
    );
  await load('victory');
  await choose(5, 10);
  await button('攻击').click();
  await choose(5, 13);
  assert.ok(await page.getByRole('heading', { name: '苍穹方获胜' }).isVisible());
  await button('悔棋').click();
  assert.equal((await readState()).winner, undefined);
  scenario('victory presentation and undo back into the ongoing match');
  await button('新对局').click();
  await button('载入演示棋局').click();
  await choose(6, 6);
  await button('攻击').click();
  await dismiss();
  await page.screenshot({ path: 'artifacts/desktop.png', fullPage: true });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await desktop.context.close();
  const mobile = await open({ width: 390, height: 844 }, 'reduce');
  page = mobile.p;
  await button('新对局').click();
  await button('载入演示棋局').click();
  await choose(7, 3);
  await dismiss();
  await page.screenshot({ path: 'artifacts/mobile.png', fullPage: true });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await load('archer');
  await choose(3, 4);
  await button('攻击').click();
  await choose(3, 6);
  assert.match(await cell(3, 6).getAttribute('aria-label'), /60生命/);
  await button('棋子图鉴').click();
  await button('武器').click();
  assert.equal(await page.locator('.codex-card').count(), 4);
  await button('关闭弹窗').click();
  await mobile.context.close();
  scenario('390px mobile layout, reduced motion, attack controls and weapon codex');
  assert.deepEqual(network, []);
  assert.deepEqual(errors, []);
  scenario('no external network requests or uncaught browser errors');
  await writeFile(
    'artifacts/browser-report.json',
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
  console.log(
    `Browser acceptance: ${checks.length} scenarios passed (${renderOnly ? 'render-only' : 'offline file://'})`,
  );
} catch (e) {
  if (page && !page.isClosed())
    await page.screenshot({ path: 'artifacts/browser-failure.png', fullPage: true });
  throw e;
} finally {
  await b.close();
}
