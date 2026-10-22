import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium } from 'playwright';

const output = resolve('artifacts/colab-browser');
await mkdir(output, { recursive: true });
const source = `
import json
from haojie_training.cloud.notebook import Dashboard
from haojie_training.console.opponents import schedule, summarize
d = Dashboard()
s = {"experiment":"browser", "state":"sampling", "device":"cuda", "updates":128,
     "saved_step":120, "counts":{"games":32,"truncated":2,"retained_samples":3800},
     "concurrency":{"actual":16,"requested":32,"pending":True},
     "backup":{"state":"备份完成"}, "performance":{"phase":"sampling"},
     "resources":{"cpu_percent":65,"gpu_utilization":42,"rss":1073741824,
                  "gpu_allocated":536870912,"gpu_reserved":805306368},
     "losses":[{"step":32,"policy":0.6,"value":0.4},
               {"step":64,"policy":0.3,"value":0.2},
               {"step":128,"policy":0.2,"value":0.1}],
     "config":{"eval_games":64},"next_eval_games":12,"evaluation_ready":True}
games = schedule(1, 1)
games += [{**g,"difficulty":"baseline"} for g in games if g["difficulty"] == "easy"]
for game in games:
    game["result"] = {("classic",1):"win",("classic",2):"loss",
                      ("shrine",1):"draw",("shrine",2):"unfinished"}[
                          (game["start"]["rules"],game["side"])]
s["evaluations"] = [{"model":"a"*64,"updates":96,"games":games,"results":summarize(games)}]
empty = Dashboard().render({"state":"paused"}, now=0)
for i in range(6):
    s["counts"]["retained_samples"] += i * 50
    s["resources"]["cpu_percent"] = 65 + i * 3
    running = d.render(s, now=i * 5)
s["error"] = "测试错误 <script>bad()</script>"
error = d.render(s, now=35)
live_games = json.loads(json.dumps(games))
for game in live_games[4:]:
    game["result"] = "pending"
s["evaluation_job"] = {"model":"b"*64,"updates":128,"completed_tasks":4,"planned":16,
                       "results":summarize(live_games, breakdown=True)}
live = d.render(s, now=40)
print(json.dumps([empty, running, error, live]))
`;
const views = JSON.parse(
  execFileSync(process.env.HAOJIE_PYTHON || 'python', ['-X', 'utf8', '-c', source], {
    encoding: 'utf8',
  }),
);
const files = [];
for (const [index, view] of views.entries()) {
  const path = resolve(output, `${index}.html`);
  await writeFile(
    path,
    `<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Colab 训练观测</title>${view}</html>`,
  );
  files.push(pathToFileURL(path).href);
}
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1200, height: 850 } });
  const requests = [],
    errors = [];
  page.on('request', (r) => {
    if (/^https?:/.test(r.url())) requests.push(r.url());
  });
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(files[0]);
  assert.equal(await page.getByText('等待实际数据', { exact: true }).count(), 3);
  await page.goto(files[1]);
  assert.equal(await page.getByRole('img').count(), 3);
  await page.getByRole('img', { name: '训练损失', exact: true }).waitFor();
  await page.getByText('16 / 32 · 待生效', { exact: true }).waitFor();
  const table = page.getByRole('table', { name: '最近完成批次', exact: true });
  await table
    .getByRole('row')
    .filter({ hasText: '简单' })
    .getByText('33.33%（3局）', { exact: true })
    .waitFor();
  assert.equal(await table.getByText('50%（2局）', { exact: true }).count(), 4);
  await page.getByText('按经典 / 神龛与先后手细分', { exact: true }).click();
  await page
    .getByRole('table', { name: '规则与座位细分', exact: true })
    .getByRole('row')
    .filter({ hasText: '简单 · 经典 · 模型先手' })
    .getByText('100%（1局）', { exact: true })
    .waitFor();
  await page.getByText('按经典 / 神龛与先后手细分', { exact: true }).click();
  await page.getByText('详细状态与排错信息', { exact: true }).focus();
  await page.keyboard.press('Enter');
  assert.ok(
    (await page
      .locator('details')
      .filter({ has: page.getByText('详细状态与排错信息', { exact: true }) })
      .getAttribute('open')) !== null,
  );
  await page.keyboard.press('Enter');
  await page.screenshot({ path: resolve(output, 'desktop.png'), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.screenshot({ path: resolve(output, 'mobile.png'), fullPage: true });
  await page.goto(files[2]);
  assert.ok((await page.getByRole('alert').innerText()).includes('<script>bad()</script>'));
  assert.equal(await page.locator('script').count(), 0);
  await page.goto(files[3]);
  await page
    .getByText('本轮评测进度：4 / 16 局（包含未完成或错误任务）', { exact: true })
    .waitFor();
  await page.getByRole('table', { name: '本轮进行中', exact: true }).waitFor();
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.screenshot({ path: resolve(output, 'evaluation.png'), fullPage: true });
  assert.deepEqual(requests, []);
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify({
      output,
      checks: [
        '空态',
        '三张图表',
        '并发待生效',
        '先后手胜率',
        '规则细分',
        '实时评测进度',
        '键盘折叠',
        '390px',
        '错误转义',
        '零网络请求',
      ],
    }),
  );
} finally {
  await browser.close();
}
