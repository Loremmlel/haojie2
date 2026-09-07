import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';

/** Test the user's contract, not a minimum width that forces a portrait board to overflow. */
export async function verifyWorkbench(page, { renderOnly = false } = {}) {
  const measurements = [];
  async function measure(label) {
    await page.waitForTimeout(100); // Let import/persistence feedback settle before measuring.
    const dismiss = page.getByRole('button', { name: '关闭提示', exact: true });
    if (await dismiss.count()) await dismiss.click();
    await page.evaluate(
      () => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done))),
    );
    const result = await page.evaluate(() => {
      const box = (selector) => {
        const el = document.querySelector(selector),
          r = el.getBoundingClientRect();
        return {
          x: r.x,
          y: r.y,
          width: r.width,
          height: r.height,
          right: r.right,
          bottom: r.bottom,
        };
      };
      const board = document.querySelector('.board');
      const cells = [...board.querySelectorAll('[data-cell]')].map((el) => {
        const r = el.getBoundingClientRect();
        // Exclude the possibility of a clipped board disguised as "no scrollbar".
        const center = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
        return {
          visible: r.x >= 0 && r.y >= 0 && r.right <= innerWidth + 1 && r.bottom <= innerHeight + 1,
          hit: center === el || el.contains(center),
          width: r.width,
          height: r.height,
        };
      });
      return {
        viewport: [innerWidth, innerHeight],
        scroll: [
          document.documentElement.scrollWidth,
          document.documentElement.scrollHeight,
          scrollX,
          scrollY,
        ],
        board: box('.board'),
        shell: box('.board-shell'),
        stage: box('.board-viewport'),
        app: box('.game-container'),
        commands: box('.command-bar'),
        hint: box('.instruction-bar'),
        save: box('.save-tools'),
        cells,
      };
    });
    const [w, h] = result.viewport;
    assert.ok(
      result.scroll[0] <= w + 1 && result.scroll[1] <= h + 1,
      `${label}: page overflow ${JSON.stringify(result.scroll)}`,
    );
    assert.ok(result.scroll[2] === 0 && result.scroll[3] === 0, `${label}: page was scrolled`);
    assert.equal(result.cells.length, 117);
    assert.ok(
      result.cells.every((c) => c.visible && c.hit),
      `${label}: every board square must be visible and hit-testable`,
    );
    assert.ok(
      result.cells.every((c) => Math.abs(c.width - c.height) < 1),
      `${label}: cells must stay square`,
    );
    for (const key of ['shell', 'commands', 'hint', 'save']) {
      const b = result[key];
      assert.ok(
        b.y >= 0 && b.bottom <= h + 1 && b.x >= 0 && b.right <= w + 1,
        `${label}: ${key} clipped`,
      );
    }
    assert.ok(result.app.width >= w - 1, `${label}: workbench must use the full horizontal space`);
    const maxGridHeight = Math.min(result.stage.height - 74, ((result.stage.width - 36) * 13) / 9);
    assert.ok(
      Math.abs(result.board.height - maxGridHeight) < 2,
      `${label}: unnecessary space reserved around board`,
    );
    measurements.push({
      label,
      viewport: result.viewport,
      board: result.board,
      scroll: result.scroll,
    });
    return result;
  }
  const button = (name) => page.getByRole('button', { name, exact: true });
  await button('新对局').click();
  await page.getByRole('radio', { name: '同屏双人', exact: true }).check();
  await button('载入演示棋局').click();
  if (await button('关闭提示').count()) await button('关闭提示').click();
  for (const [w, h] of [
    [1100, 600],
    [1100, 768],
    [1101, 900],
    [1279, 900],
    [1280, 900],
    [1366, 768],
    [1440, 900],
    [1440, 1080],
    [1536, 864],
    [1920, 1080],
    [2560, 1440],
    [3440, 1440],
  ]) {
    await page.setViewportSize({ width: w, height: h });
    await measure('populated-local');
  }
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.locator('[data-cell="6,6"]').click();
  await page.locator('.ability-details summary').click();
  await button('攻击').click();
  await measure('expanded-inspector-and-targeting');
  await page.locator('.hand-scroll').hover();
  await page.mouse.wheel(0, 900);
  await page.waitForTimeout(160);
  await measure('sidebar-scroll-leaves-board-fixed');
  const scrollTop = await page.locator('.hand-scroll').evaluate((el) => el.scrollTop);
  assert.ok(scrollTop > 0, 'Long hand and log should scroll locally');
  await page.locator('.hand-scroll').evaluate((el) => (el.scrollTop = 0));
  await page.screenshot({ path: 'artifacts/workbench-desktop.png', fullPage: false });
  // A real saved terminal state must not grow the page or cover the board.
  const victory = JSON.parse(await readFile('artifacts/ai-fixtures/response.json', 'utf8'));
  victory.match.mode = 'local';
  victory.present.winner = 1;
  victory.present.bases[2] = 0;
  await page.locator('input[type=file]').setInputFiles({
    name: 'victory.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(victory)),
  });
  await page.locator('.victory-banner').waitFor();
  if (await button('关闭提示').count()) await button('关闭提示').click();
  await page.setViewportSize({ width: 1100, height: 600 });
  await measure('victory');
  await button('新对局').click();
  await page.getByRole('radio', { name: '人机对战', exact: true }).check();
  await page.getByLabel('你的阵营').selectOption('1');
  await button('开始正式对局').click();
  if (await button('关闭提示').count()) await button('关闭提示').click();
  await measure('ai-summon');
  await page.locator('input[type=file]').setInputFiles('artifacts/ai-fixtures/human-reaction.json');
  if (await button('关闭提示').count()) await button('关闭提示').click();
  await measure('human-reaction');
  await button('新对局').click();
  await page.getByRole('radio', { name: '同屏双人', exact: true }).check();
  await button('载入演示棋局').click();
  if (await button('关闭提示').count()) await button('关闭提示').click();
  await page.setViewportSize({ width: 1366, height: 768 });
  await page.locator('[data-cell="6,6"]').click();
  await page.screenshot({ path: 'artifacts/workbench-laptop.png', fullPage: false });
  // Compact widths still use natural document flow; no application/global overflow locks.
  for (const width of [1099, 900, 820, 390]) {
    await page.setViewportSize({ width, height: 844 });
    assert.ok(
      await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
      `compact overflow ${width}`,
    );
    await button('新对局').click();
    await button('关闭弹窗').click();
  }
  // The public component responds to its own width, not a wide host's viewport.
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.locator('.hj-game').evaluate((el) => (el.style.width = '780px'));
  await page.waitForTimeout(100);
  assert.ok(await page.locator('.board').evaluate((el) => el.getBoundingClientRect().right <= 780));
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
  await page.locator('.hj-game').evaluate((el) => {
    el.style.width = '';
    el.style.setProperty('--haojie-height', '700px');
  });
  await measure('host-provided-height');
  assert.ok(
    (await page.locator('.board-shell').boundingBox()).y +
      (await page.locator('.board-shell').boundingBox()).height <=
      700,
  );
  await page.locator('.hj-game').evaluate((el) => el.style.removeProperty('--haojie-height'));
  await writeFile(
    'artifacts/layout-report.json',
    JSON.stringify(
      {
        mode: renderOnly ? 'render-only' : 'offline-file',
        contract:
          'Whole square 9×13 board and essential controls visible without page/board scrolling on wide screens; sidebars may scroll.',
        measurements,
      },
      null,
      2,
    ),
  );
}
