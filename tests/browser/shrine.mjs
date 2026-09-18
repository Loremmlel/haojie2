import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readFile, writeFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium } from 'playwright';
const renderOnly = process.env.HAOJIE_RENDER_ONLY === '1';
execFileSync(process.execPath, ['--import', 'tsx', 'tests/browser/shrine-fixtures.ts'], {
  stdio: 'inherit',
});
const html = await readFile('dist/index.html', 'utf8');
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
const check = (label) => {
  checks.push(label);
  console.log('✓ ' + label);
};
async function state() {
  await page.evaluate(() => (window.shrineExport = null));
  await button('导出存档').click();
  await page.waitForFunction(() => window.shrineExport !== null);
  return JSON.parse(await page.evaluate(() => window.shrineExport));
}
async function load(name) {
  const path = `artifacts/shrine-fixtures/${name}.json`,
    expected = JSON.parse(await readFile(path, 'utf8'));
  await page.locator('input[type=file]').setInputFiles(path);
  assert.deepEqual((await state()).present, expected.present);
}
async function wait(predicate) {
  let value;
  for (let n = 0; n < 300; n++) {
    value = await state();
    if (predicate(value)) return value;
    await page.waitForTimeout(40);
  }
  assert.fail(JSON.stringify(value?.present).slice(0, 700));
}
try {
  if (renderOnly) await page.setContent(html);
  else await page.goto(pathToFileURL(resolve('dist/index.html')).href);
  await page.getByRole('heading', { name: '浩劫3.0', exact: true }).waitFor();
  await page.evaluate(() => {
    const original = URL.createObjectURL.bind(URL);
    URL.createObjectURL = (blob) => {
      if (blob instanceof Blob && blob.type === 'application/json')
        blob.text().then((t) => (window.shrineExport = t));
      return original(blob);
    };
  });
  await button('新对局').click();
  await page.getByLabel('规则模式').selectOption('shrine');
  await page.getByPlaceholder('例如 20260907').fill('90');
  await button('开始正式对局').click();
  assert.equal((await state()).present.phase, 'shrine-draft');
  assert.equal(await page.locator('.shrine-choice').count(), 3);
  check(
    'new match selects independent shrine rules with three local and three visible opponent candidates',
  );
  await page.locator('.shrine-choice').filter({ hasText: '举旗' }).click();
  await button('锁定神龛').click();
  assert.equal((await state()).present.shrineDraft.revealed, false);
  await page.locator('.shrine-choice').filter({ hasText: '老千K' }).click();
  await button('锁定神龛').click();
  assert.equal((await state()).present.phase, 'shrine-setup');
  await button('储存神龛，完成入场').click();
  await page.locator('.hand-card').filter({ hasText: '老千K' }).click();
  await button('启用永久光环').click();
  await button('完成神龛入场').click();
  assert.equal((await state()).present.ply, 1);
  check('both commitments reveal together, sequential setup can store a unit or activate an aura');
  await load('flag-card');
  await page.locator('.hand-card').filter({ hasText: '举旗' }).click();
  for (const y of [8, 9, 10]) assert.match(await cell(7, y).getAttribute('class'), /legal/);
  assert.doesNotMatch(await cell(8, 7).getAttribute('class'), /legal/);
  await cell(7, 9).click();
  assert.equal((await state()).present.landmarks[0].y, 9);
  await button('悔棋').click();
  assert.equal((await state()).present.landmarks.length, 0);
  await button('重做').click();
  check('flag highlights only 78/79/710 and supports atomic deployment undo/redo');
  await load('flag-stack');
  await cell(7, 9).click();
  assert.ok(await page.getByRole('heading', { name: '冲锋怪', exact: true }).isVisible());
  await cell(7, 9).click();
  assert.ok(await page.getByRole('heading', { name: '举旗', exact: true }).isVisible());
  check('clicking an occupied landmark cycles resident and landmark inspector');
  await load('layer');
  await cell(3, 4).click();
  await button('攻击').click();
  await button('目标：地标').click();
  assert.doesNotMatch(await cell(3, 6).getAttribute('class'), /legal/);
  await button('取消当前操作').click();
  await page.locator('.hand-card').filter({ hasText: '爆弹' }).click();
  await cell(3, 6).click();
  let s = (await state()).present;
  assert.equal(s.landmarks[0].hp, 0);
  assert.equal(s.units.find((u) => u.owner === 2).hp, 30);
  check('target-layer switch respects occupant priority and blast damages both layers');
  await load('charge');
  await page.locator('.hand-card').filter({ hasText: '冲锋怪' }).click();
  await button('扣10血 · 冲锋部署').click();
  await cell(3, 6).click();
  s = (await state()).present;
  assert.equal(s.units[0].extraOperations, 1);
  assert.equal(s.units[0].hp, 40);
  check('Jinye charge deployment retains the health cost and grants two complete operations');
  await load('heal');
  await cell(3, 4).click();
  assert.ok(await button('攻击 · 造成伤害').isVisible());
  await button('攻击 · 治疗生命').click();
  await cell(3, 5).click();
  assert.equal((await state()).present.units[1].hp, 30);
  check('signed attack exposes distinct damage and healing choices');
  await load('offer');
  await button('免费终极召唤').click();
  s = (await state()).present;
  assert.ok(s.summonOffer.groups.length >= 3);
  await page.locator('.summon-offer input').nth(0).check();
  await page.locator('.summon-offer input').nth(1).check();
  await button('保留两个召唤结果').click();
  assert.equal((await state()).present.summonOffer, undefined);
  check('Old K candidate groups are selected atomically in pairs');
  await load('synthesis');
  for (let i = 0; i < 3; i++)
    await page.locator('.synthesis-controls input[type=checkbox]').nth(i).check();
  await button('启用光环 · 合成牢千K').click();
  assert.equal((await state()).present.auras[1][0].kind, 'laoqian');
  await button('免费终极召唤').click();
  await page.getByRole('dialog').getByRole('combobox').selectOption('u25');
  await button('确定自选 · 消耗本回合能力').click();
  assert.equal((await state()).present.hands[1].length, 8);
  check(
    'three U13 fuse into Laoqian and one selected ultimate outcome produces its complete clone batch',
  );
  await load('clock');
  await button('时钟 · 选择回溯目标').click();
  await cell(4, 4).click();
  s = (await state()).present;
  assert.equal(s.units[0].x, 3);
  assert.equal(s.units[0].hp, 45);
  assert.equal(s.units[0].onceUsed, false);
  assert.ok(await button('时钟 · 选择回溯目标').isDisabled());
  check('Clock restores prior own-start state and disables its used allowance');
  await load('jade');
  await cell(3, 4).click();
  await button('玉碎 · 自杀伤敌').click();
  await cell(3, 5).click();
  s = (await state()).present;
  assert.equal(s.units.length, 1);
  assert.equal(s.units[0].hp, 35);
  assert.equal(s.heads[2], 0);
  check('Jade suicide targets an enemy without awarding a suicide head');
  await load('weapon');
  await page.locator('.hand-card').filter({ hasText: '铁吊' }).click();
  await cell(3, 4).click();
  s = (await state()).present;
  assert.equal(s.units[0].maxHp, 70);
  check('shrine equipment uses existing targeting and updates carrier health');
  await load('dormant');
  await cell(7, 9).click();
  assert.ok(await page.getByText(/地标休眠 · 重建还需4个己方回合/).isVisible());
  await page.screenshot({ path: 'artifacts/shrine-desktop.png', fullPage: true });
  check('dormant landmarks remain inspectable with visible rebuild countdown');
  await page.setViewportSize({ width: 390, height: 844 });
  await load('flag-card');
  await page.locator('.hand-card').filter({ hasText: '举旗' }).focus();
  await page.keyboard.press('Enter');
  await cell(7, 10).focus();
  await page.keyboard.press('Enter');
  assert.equal((await state()).present.landmarks[0].y, 10);
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.screenshot({ path: 'artifacts/shrine-mobile.png', fullPage: true });
  check('390px keyboard deployment reaches corrected flag coordinates without horizontal overflow');
  await page.setViewportSize({ width: 1440, height: 1080 });
  for (const human of ['1', '2']) {
    await button('新对局').click();
    await page.getByLabel('规则模式').selectOption('shrine');
    await page.getByRole('radio', { name: '人机对战', exact: true }).check();
    await page.getByLabel('AI难度').selectOption('easy');
    await page.getByLabel('你的阵营').selectOption(human);
    await page.getByPlaceholder('例如 20260907').fill('90');
    await button('开始正式对局').click();
    await wait((v) => v.present.phase === 'shrine-draft' && v.present.active === Number(human));
    await page.locator('.shrine-choice').first().click();
    await button('锁定神龛').click();
    await wait((v) => v.present.phase === 'shrine-setup' && v.present.active === Number(human));
    await button('储存神龛，完成入场').click();
    await wait((v) => v.present.ply >= 1 && v.present.active === Number(human));
  }
  check('AI as either side completes hidden draft, setup and summons without stalling');
  assert.deepEqual(errors, []);
  assert.deepEqual(network, []);
  await writeFile(
    'artifacts/shrine-browser-report.json',
    JSON.stringify({ renderOnly, checks, errors, network }, null, 2),
  );
  console.log(
    `${checks.length} shrine browser scenarios passed${renderOnly ? ' (in-memory rendering; file:// remains CI acceptance)' : ''}`,
  );
} finally {
  await browser.close();
}
