import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium } from 'playwright';

// 从生产 HTML 验证公开交互；不读取 React 状态或向页面注入规则实现。
const renderOnly = process.env.HAOJIE_RENDER_ONLY === '1';
const browser = await chromium.launch({
  executablePath:
    process.env.BROWSER_PATH || (existsSync('/usr/bin/chromium') ? '/usr/bin/chromium' : undefined),
});
const errors = [],
  network = [],
  checks = [];
await mkdir('artifacts', { recursive: true });
try {
  for (const width of [1440, 390]) {
    const context = await browser.newContext({
      viewport: { width, height: 1000 },
      reducedMotion: 'reduce',
    });
    await context.setOffline(true);
    const page = await context.newPage();
    page.on('pageerror', (error) => errors.push(error.message));
    page.on('request', (request) => {
      if (/^https?:/.test(request.url())) network.push(request.url());
    });
    if (renderOnly) await page.setContent(await readFile('dist/index.html', 'utf8'));
    else await page.goto(pathToFileURL(resolve('dist/index.html')).href);
    const button = (name) => page.getByRole('button', { name, exact: true });
    await button('棋子图鉴').click();
    const codex = page.getByRole('dialog', { name: '浩劫图鉴', exact: true });
    await codex.getByRole('textbox').fill('跑得快小屋');
    const hut = codex
      .locator('.codex-card')
      .filter({ has: page.getByRole('heading', { name: '跑得快小屋', exact: true }) });
    assert.equal(await hut.count(), 1);
    assert.doesNotMatch(await codex.innerText(), /(?:普通|终极)\s*\d/);
    const runner = hut.getByRole('link', { name: '超级跑得快', exact: true });
    await runner.focus();
    await page.keyboard.press('Enter');
    const detail = page.getByRole('dialog', { name: '超级跑得快', exact: true });
    await detail.waitFor();
    assert.match(await detail.innerText(), /攻击|移动/);
    assert.equal(await detail.locator('.codex-card').count(), 1);
    assert.equal(await page.locator('dialog[open]').count(), 2);
    assert.equal(new URL(page.url()).hash, '');
    await page.screenshot({ path: `artifacts/unit-reference-${width}.png`, fullPage: true });
    for (let n = 0; n < 8; n++) {
      await page.keyboard.press('Tab');
      assert.equal(
        await detail.evaluate((dialog) => dialog.contains(document.activeElement)),
        true,
      );
    }
    await page.keyboard.press('Escape');
    await detail.waitFor({ state: 'detached' });
    assert.equal(await runner.evaluate((link) => link === document.activeElement), true);
    assert.equal(await codex.getByRole('textbox').inputValue(), '跑得快小屋');
    await codex.getByRole('textbox').fill('寒冰法杖');
    await codex.getByRole('heading', { name: '寒冰法杖', exact: true }).getByRole('link').click();
    const wand = page.getByRole('dialog', { name: '寒冰法杖', exact: true });
    await wand.getByRole('link', { name: '大法师', exact: true }).click();
    const mage = page.getByRole('dialog', { name: '大法师', exact: true });
    await mage.getByRole('button', { name: '返回上一介绍' }).click();
    await wand.getByRole('button', { name: '关闭弹窗' }).click();
    await codex.getByRole('button', { name: '清空图鉴搜索' }).click();
    assert.equal(await codex.locator('.codex-card').count(), 82);
    await codex.getByRole('textbox').fill('不存在的单位名称');
    assert.match(await codex.innerText(), /没有匹配条目/);
    await codex.getByRole('button', { name: '关闭弹窗' }).click();
    await button('规则').click();
    const rules = page.getByRole('dialog', { name: '浩劫 · 规则手册' });
    assert.doesNotMatch(await rules.innerText(), /(?:普通|终极)\s*\d|抽到17/);
    await rules.getByRole('link', { name: '改判小法师', exact: true }).first().click();
    await page.getByRole('dialog', { name: '改判小法师', exact: true }).waitFor();
    await page.keyboard.press('Escape');
    await rules.getByRole('button', { name: '关闭弹窗' }).click();
    await button('普通召唤').click();
    const card = page.locator('.hand-card').first();
    const before = await page.locator('.instruction-bar').innerText();
    const name = await card.getByRole('link').innerText();
    const saved = renderOnly ? null : await page.evaluate(() => JSON.stringify(localStorage));
    await card.getByRole('link').click();
    await page
      .getByRole('dialog', { name, exact: true })
      .getByRole('button', { name: '关闭弹窗' })
      .click();
    assert.equal(await page.locator('.instruction-bar').innerText(), before);
    if (!renderOnly) assert.equal(await page.evaluate(() => JSON.stringify(localStorage)), saved);
    assert.equal(await page.locator('button a, a a').count(), 0);
    assert.equal(
      await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
      true,
    );
    checks.push(
      `${width}px：名称跳转、嵌套介绍、键盘焦点、返回、搜索空态、规则入口、手牌只读、无嵌套控件`,
    );
    await context.close();
  }
  assert.deepEqual(errors, []);
  assert.deepEqual(network, []);
  await writeFile(
    'artifacts/unit-reference-report.json',
    JSON.stringify(
      { mode: renderOnly ? 'in-memory' : 'file:// offline', checks, errors, network },
      null,
      2,
    ),
  );
  console.log(checks.join('\n'));
} finally {
  await browser.close();
}
