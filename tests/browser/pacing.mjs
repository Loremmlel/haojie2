import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';

/** Observe public cell labels and exported sessions, not React timers or planner internals. */
export async function verifyPacing({ page, load, exported, waitForState, scenario }) {
  const button = (name) => page.getByRole('button', { name, exact: true });
  const report = { firstActionMs: 0, attackIntervalsMs: [], cancellation: [] };
  const ready = () => page.getByText('AI准备行动…', { exact: true }).waitFor();
  const initial = JSON.parse(await readFile('artifacts/ai-fixtures/pacing.json', 'utf8'));

  await page.evaluate(() => {
    window.pacingHits = [];
    window.pacingStarted = performance.now();
    let last = '70';
    window.pacingObserver = new MutationObserver(() => {
      const text = document.querySelector('[data-cell="4,8"]')?.getAttribute('aria-label') ?? '';
      const hp = text.includes('墓地') ? text.match(/(\d+)生命/)?.[1] : null;
      if (hp && hp !== last) {
        window.pacingHits.push({ time: performance.now(), hp: Number(hp) });
        last = hp;
      }
    });
    window.pacingObserver.observe(document.querySelector('.board'), {
      attributes: true,
      attributeFilter: ['aria-label'],
      subtree: true,
    });
  });
  try {
    await load('pacing');
    // Even a ready/cached decision cannot execute immediately.
    await ready();
    assert.deepEqual((await exported()).present, initial.present);
    await page.waitForFunction(() => window.pacingHits.length >= 3, null, { timeout: 15000 });
    const { hits, start } = await page.evaluate(() => ({
      hits: window.pacingHits,
      start: window.pacingStarted,
    }));
    report.firstActionMs = Math.round(hits[0].time - start);
    report.attackIntervalsMs = hits.slice(1).map((hit, n) => Math.round(hit.time - hits[n].time));
    assert.ok(report.firstActionMs >= 1400, 'AI must leave time to notice its first decision');
    assert.ok(
      report.attackIntervalsMs.every((ms) => ms >= 1250),
      'Repeated/cached attacks must leave hit effects readable instead of flashing through',
    );

    // Stop an already planned move, not only an in-flight Worker calculation.
    await ready();
    await button('暂停AI').click();
    const paused = await exported();
    await page.waitForTimeout(1800);
    assert.deepEqual(await exported(), paused);
    report.cancellation.push('pause during ready-to-act wait');
    await button('继续AI').click();
    await waitForState((s) => s.past.length > paused.past.length);
    await button('暂停AI').click();
    scenario('AI opening and cached attacks are paced; a ready action is cancellable and resumes');

    await load('pacing');
    await ready();
    const dialogSnapshot = await exported();
    await button('新对局').click();
    await page.waitForTimeout(1800);
    await button('关闭弹窗').click();
    assert.deepEqual(await exported(), dialogSnapshot);
    await button('新对局').click();
    await page.getByRole('radio', { name: '同屏双人', exact: true }).check();
    await button('开始正式对局').click();
    const replaced = await exported();
    await page.waitForTimeout(1800);
    assert.deepEqual(await exported(), replaced);
    report.cancellation.push('dialog and new match during ready-to-act wait');

    await load('pacing');
    await ready();
    await load('human-reaction');
    const imported = await exported();
    await page.waitForTimeout(1800);
    assert.deepEqual(await exported(), imported);
    report.cancellation.push('import while a previous match has a pending action');

    await load('response');
    const humanDecision = await exported();
    await button('结束回合').click();
    await ready();
    await button('悔棋').click();
    assert.deepEqual((await exported()).present, humanDecision.present);
    await page.waitForTimeout(1800);
    assert.deepEqual((await exported()).present, humanDecision.present);
    report.cancellation.push('undo during ready-to-act wait');
    scenario('presentation waits are invalidated by dialogs, new games, import and human undo');
    await writeFile('artifacts/ai-pacing-report.json', JSON.stringify(report, null, 2));
  } finally {
    await page.evaluate(() => window.pacingObserver.disconnect());
  }
}
