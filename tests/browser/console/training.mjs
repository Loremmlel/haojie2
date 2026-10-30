import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { chromium } from "playwright";

const directory = await mkdtemp(join(tmpdir(), "haojie-console-"));
const python =
  process.platform === "win32"
    ? "training/.venv/Scripts/python.exe"
    : "training/.venv/bin/python";
const child = spawn(
  python,
  [
    "-u",
    "-B",
    "-m",
    "training.haojie",
    "web",
    "--root",
    directory,
    "--port",
    "0",
  ],
  { windowsHide: true, env: { ...process.env, PYTHONIOENCODING: "utf-8" } },
);
let output = "";
child.stdout.on("data", (data) => (output += data.toString()));
child.stderr.on("data", (data) => (output += data.toString()));
const wait = async (condition, label, timeout = 60000) => {
  const until = Date.now() + timeout;
  while (Date.now() < until) {
    if (await condition()) return;
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(`等待超时：${label}\n${output}`);
};
let browser;
let url;
try {
  await wait(() => /http:\/\/127\.0\.0\.1:\d+/.test(output), "服务启动");
  url = output.match(/http:\/\/127\.0\.0\.1:\d+/)[0];
  const state = () => fetch(`${url}/api/state`).then((r) => r.json());
  browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1080 },
    reducedMotion: "reduce",
  });
  const page = await context.newPage();
  const errors = [],
    external = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("request", (r) => {
    if (!r.url().startsWith(url)) external.push(r.url());
  });
  await page.goto(url);
  await page.getByRole("button", { name: "开始训练" }).waitFor();
  await wait(
    async () =>
      await page.getByRole("button", { name: "开始训练" }).isEnabled(),
    "初始连接",
  );
  await mkdir("artifacts/training-web", { recursive: true });
  await page.screenshot({
    path: "artifacts/training-web/empty-desktop.png",
    fullPage: true,
  });
  await page.locator("#workers").fill("2");
  await page.locator("#games_per_worker").fill("2");
  await page.locator("#batch").fill("2");
  await page.locator("summary").filter({ hasText: "新训练初始化" }).click();
  await page.locator("#warmup_games").fill("0");
  await page.locator("summary").filter({ hasText: /^搜索/ }).click();
  await page.locator("#simulations").fill("4");
  await page.locator("#fast_simulations").fill("1");
  await page.locator("summary").filter({ hasText: "保存与停止" }).click();
  await page.locator("#chunk_commands").fill("1");
  await page.locator("#seconds").fill("90");
  await page.locator("#workers").fill("0");
  await page.getByRole("button", { name: "开始训练" }).click();
  assert.match(await page.getByRole("alert").innerText(), /搜索线程/);
  await page.locator("#workers").fill("2");
  await page.waitForTimeout(2400);
  assert.equal(await page.locator("#workers").inputValue(), "2");
  await page.getByRole("button", { name: "开始训练" }).click();
  await wait(async () => (await state()).stats.updates >= 2, "真实 GPU 学习");
  const before = await state();
  await page.reload();
  await wait(
    async () =>
      await page.getByRole("button", { name: "保存并停止" }).isEnabled(),
    "刷新保留训练",
  );
  assert.equal((await state()).run, before.run);
  await page.locator("#workers").fill("1");
  await page.locator("#games_per_worker").fill("1");
  await page.getByRole("button", { name: "保存并应用参数" }).click();
  await wait(async () => {
    const s = await state();
    return (
      s.state === "running" &&
      s.config.workers === 1 &&
      s.stats.queued_games === 3
    );
  }, "缩容后保留三局待恢复");
  await page.getByRole("button", { name: "保存并停止" }).click();
  await wait(async () => (await state()).state === "saved", "保存停止");
  await wait(
    async () =>
      await page.getByRole("button", { name: "继续训练" }).isEnabled(),
    "继续按钮",
  );
  await page.locator("#workers").fill("2");
  await page.locator("#games_per_worker").fill("3");
  await page.getByRole("button", { name: "继续训练" }).click();
  await wait(async () => {
    const s = await state();
    return (
      s.state === "running" &&
      s.stats.active_games === 6 &&
      s.stats.queued_games === 0
    );
  }, "扩大至六局");
  await page.getByLabel("图表时间范围").focus();
  await page.keyboard.press("Alt+ArrowDown");
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("Enter");
  await page.getByLabel("曲线平滑时长").selectOption("120");
  await page.getByLabel("原始线", { exact: true }).check();
  await wait(
    async () =>
      (await page
        .locator(".chart-panel")
        .first()
        .locator(".trend-line")
        .count()) > 0,
    "真实损失观测进入曲线",
  );
  await page.getByRole("group", { name: /学习损失，左右/ }).focus();
  await page.keyboard.press("ArrowLeft");
  await wait(
    async () => (await page.locator(".chart-tooltip").count()) > 0,
    "键盘曲线读数",
    5000,
  );
  await page.keyboard.press("Escape");
  await page.screenshot({
    path: "artifacts/training-web/running-desktop.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({
    path: "artifacts/training-web/running-mobile.png",
    fullPage: true,
  });
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
    true,
  );
  await context.setOffline(true);
  await wait(
    async () =>
      (await page.locator(".state-pill").innerText()).includes("连接中断"),
    "断线提示",
    15000,
  );
  assert.equal(
    await page.getByRole("button", { name: "保存并停止" }).isEnabled(),
    false,
  );
  await context.setOffline(false);
  await wait(
    async () =>
      await page.getByRole("button", { name: "保存并停止" }).isEnabled(),
    "连接恢复",
  );
  await page.getByRole("button", { name: "保存并停止" }).click();
  await wait(async () => (await state()).state === "saved", "最终保存");
  const final = await state();
  assert.ok(final.stats.updates > before.stats.updates);
  assert.deepEqual(errors, []);
  assert.deepEqual(external, []);
  console.log(
    JSON.stringify({
      checks:
        "真实 GPU 开始、刷新、缩容排队、保存停止、扩容继续、平滑与原始线、键盘、390px、断线恢复、无外部请求",
      firstUpdates: before.stats.updates,
      finalUpdates: final.stats.updates,
      active: final.stats.active_games,
      queued: final.stats.queued_games,
      historyPoints: final.history.length,
    }),
  );
} finally {
  if (browser) await browser.close();
  if (url) {
    const current = await fetch(`${url}/api/state`)
      .then((r) => r.json())
      .catch(() => null);
    if (
      current?.running &&
      !["stopping", "restarting"].includes(current.state)
    ) {
      await fetch(`${url}/api/control`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Haojie-Control": "1",
        },
        body: JSON.stringify({
          action: "stop",
          revision: current.revision,
          session: current.session,
        }),
      });
    }
    await wait(
      async () =>
        !(await fetch(`${url}/api/state`).then((r) => r.json())).running,
      "清理前等待训练退出",
    ).catch(() => {});
  }
  child.kill();
  await new Promise((r) => child.once("exit", r));
  const target = resolve(directory);
  assert.ok(
    target.startsWith(
      resolve(tmpdir()) + (process.platform === "win32" ? "\\" : "/"),
    ),
  );
  await rm(target, { recursive: true });
}
