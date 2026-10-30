import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFile, mkdir } from "node:fs/promises";
import { chromium } from "playwright";

// 图表视觉与草稿回执验收使用明确标注的合成数据，不作为训练记录。
const python =
  process.platform === "win32"
    ? "training/.venv/Scripts/python.exe"
    : "training/.venv/bin/python";
const schema = JSON.parse(
  execFileSync(
    python,
    [
      "-B",
      "-c",
      "import json; from training.haojie.web.settings import defaults,schema; print(json.dumps(dict(defaults=defaults(),fields=schema())))",
    ],
    { encoding: "utf8", windowsHide: true },
  ),
);
const html = await readFile("training/haojie/web/index.html", "utf8");
const now = Date.now() / 1000;
const history = Array.from({ length: 120 }, (_, i) => ({
  id: i + 1,
  t: now - (119 - i) * 5,
  policy: 3.4 - i * 0.012 + Math.sin(i * 2) * 0.22,
  value: 0.9 - i * 0.003 + Math.sin(i * 1.7) * 0.08,
  aux: 0.3 - i * 0.001,
  cps: 25 + Math.sin(i * 0.13) * 3 + Math.sin(i * 3) * 5,
  sps: 12 + Math.sin(i * 0.13) * 2 + Math.sin(i * 2) * 3,
  cpu: 48 + Math.sin(i * 0.2) * 10,
  gpu: 65 + Math.sin(i * 0.15) * 18,
  ram: 2.3 + i * 0.003,
  vram: 2.6 + Math.sin(i * 0.05) * 0.1,
  replay: i * 0.002,
}));
let state = {
  run: "界面验收样例",
  state: "running",
  message: "合成数据，仅用于界面验收。",
  revision: 0,
  session: "visual-fixture",
  config: schema.defaults,
  stats: {
    updates: 1840,
    completed: 128,
    truncated: 16,
    trained_samples: 29440,
    phase: "self",
    demonstrations: 8,
    imitation_updates: 512,
    active_games: 64,
    queued_games: 0,
  },
  running: true,
  checkpoint: true,
  sequence: 120,
  logs: [],
  runs: [{ name: "界面验收样例", updated: now }],
  cpu_count: 16,
  memory_gib: 32,
};
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1080 },
    reducedMotion: "reduce",
  });
  await page.route("http://127.0.0.1:48765/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/")
      return route.fulfill({ contentType: "text/html", body: html });
    if (url.pathname === "/api/schema") return route.fulfill({ json: schema });
    if (url.pathname === "/api/control") {
      const body = route.request().postDataJSON();
      assert.equal(body.action, "apply");
      state = { ...state, state: "restarting", revision: 1 };
      await route.fulfill({ json: { ...state, history } });
      setTimeout(() => {
        state = {
          ...state,
          state: "running",
          config: body.config,
          revision: 2,
        };
      }, 250);
      return;
    }
    return route.fulfill({
      json: {
        ...state,
        history:
          state.session === "restarted-service"
            ? []
            : history.filter(
                (p) => p.id > Number(url.searchParams.get("since") || 0),
              ),
      },
    });
  });
  await page.goto("http://127.0.0.1:48765/");
  await page.locator(".trend-line").first().waitFor();
  assert.ok(
    (await page.locator(".trend-line").first().getAttribute("d")).includes("C"),
  );
  await page.getByLabel("原始线", { exact: true }).check();
  assert.ok(await page.locator(".raw-line").count());
  await mkdir("artifacts/training-web", { recursive: true });
  await page.screenshot({
    path: "artifacts/training-web/fixture-desktop.png",
    fullPage: true,
  });
  await page.locator("#workers").fill("4");
  await page.getByRole("button", { name: "保存并应用参数" }).click();
  await page.waitForTimeout(3000);
  assert.equal(await page.locator("#workers").inputValue(), "4");
  assert.equal(
    await page.getByRole("button", { name: "保存并应用参数" }).isDisabled(),
    true,
  );
  await page.setViewportSize({ width: 390, height: 844 });
  assert.ok(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  );
  await page.screenshot({
    path: "artifacts/training-web/fixture-mobile.png",
    fullPage: true,
  });
  state = {
    ...state,
    session: "restarted-service",
    revision: 0,
    run: null,
    state: "idle",
    running: false,
    stats: {},
    sequence: 0,
  };
  await page.locator("#run-name").waitFor();
  assert.equal(
    await page
      .getByRole("button", { name: "开始训练", exact: true })
      .isEnabled(),
    true,
  );
  assert.equal(await page.locator(".trend-line").count(), 0);
  console.log(
    "图表真实曲线路径、原始线、桌面/手机布局、参数草稿和服务重启后重新连接：通过（合成数据）。",
  );
} finally {
  await browser.close();
}
