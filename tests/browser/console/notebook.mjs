import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { chromium } from 'playwright';

const output = resolve(`artifacts/notebook-browser-${Date.now()}`);
await mkdir(output, { recursive: true });
// 仅替换 Colab 的消息传输；面板回调、回环鉴权、控制器与 Rust 采样都使用正式实现。
const driver = `
import gzip, json, sys, threading, urllib.request
from pathlib import Path
from haojie_training.cloud.panel import Panel
from haojie_training.cloud.session import Session
from haojie_training.console.server import http_server
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
starts = json.loads(gzip.decompress(Path("tests/fixtures/native/late-starts.json.gz").read_bytes()))[:2]
session = Session(sys.argv[1], Path(sys.argv[2]), device="cpu", tiny=True, starts=starts,
    settings={"environments":2,"games_per_round":2,"updates_per_round":1,
              "max_commands":32,"pool_mib":16,"history_percent":0,"eval_games":100000})
server = http_server(session, 0)
threading.Thread(target=server.serve_forever, daemon=True).start()
base = "http://127.0.0.1:" + str(server.server_port)
def api(action=None, values=None):
    headers, data, path = {}, None, "/api/status"
    if action is not None:
        with urllib.request.urlopen(base + "/api/session") as response:
            token = json.load(response)["token"]
        headers = {"Origin":base,"X-Session-Token":token,"Content-Type":"application/json"}
        data, path = json.dumps({"action":action,"values":values}).encode(), "/api/command"
    try:
        with urllib.request.urlopen(urllib.request.Request(base + path, data=data, headers=headers), timeout=10) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(json.load(error)["error"])
panel = Panel(api)
try:
    print(json.dumps({"html":panel.render(),"identity":panel.identity}), flush=True)
    for line in sys.stdin:
        message = json.loads(line)
        print(json.dumps({"id":message["id"],"value":panel.call(*message["args"])}), flush=True)
finally:
    panel.close()
    session.close()
    server.shutdown()
    server.server_close()
`;
const child = spawn(
  process.env.HAOJIE_PYTHON || 'python',
  [
    '-X',
    'utf8',
    '-u',
    '-c',
    driver,
    process.env.HAOJIE_NATIVE || resolve('artifacts/native-target/release/haojie-engine.exe'),
    resolve(output, 'training'),
  ],
  { stdio: ['pipe', 'pipe', 'pipe'] },
);
let serial = 0,
  stderr = '',
  failStatus = false;
const pending = new Map(),
  calls = [];
let readyResolve, readyReject;
const ready = new Promise((yes, no) => {
  readyResolve = yes;
  readyReject = no;
});
child.stderr.on('data', (chunk) => {
  stderr += chunk.toString();
});
createInterface({ input: child.stdout }).on('line', (line) => {
  const reply = JSON.parse(line);
  if (!reply.id) readyResolve(reply);
  else {
    pending.get(reply.id)?.(reply.value);
    pending.delete(reply.id);
  }
});
child.on('exit', (code) => {
  if (code) readyReject(new Error(stderr));
});
function rpc(args) {
  return new Promise((resolveReply) => {
    const id = ++serial;
    pending.set(id, resolveReply);
    child.stdin.write(JSON.stringify({ id, args }) + '\n');
  });
}
const browser = await chromium.launch({ executablePath: process.env.BROWSER_PATH, headless: true });
try {
  const initial = await ready;
  const page = await browser.newPage({ viewport: { width: 1200, height: 850 } });
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.exposeFunction('notebookInvoke', async (name, args) => {
    assert.equal(name, 'haojie.panel');
    calls.push(args[1]);
    if (args[1] === 'status' && failStatus) throw new Error('模拟连接中断');
    return { data: { 'application/json': await rpc(args) } };
  });
  await page.evaluate(() => {
    window.rpcCompleted = 0;
    window.google = {
      colab: {
        kernel: {
          invokeFunction: async (...args) => {
            const reply = await window.notebookInvoke(...args);
            window.rpcCompleted++;
            return reply;
          },
        },
      },
    };
  });
  await page.setContent(
    `<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Notebook 回调验收</title>${initial.html}</html>`,
  );
  await page.getByRole('button', { name: '开始', exact: true }).click();
  await page.getByText('开始请求已接收', { exact: true }).waitFor();
  await page.getByLabel('并发环境（1–128）', { exact: true }).fill('3');
  await page.getByLabel('并发环境（1–128）', { exact: true }).press('Enter');
  await page
    .getByText('服务已确认并发目标 3；实际进程在下一采样窗口切换', { exact: true })
    .waitFor();
  assert.equal(calls.filter((action) => action === 'concurrency').length, 1);
  await page.waitForFunction(() =>
    document.querySelector('.metrics').textContent.includes('3 / 3'),
  );

  await page.getByText('其他设置（暂停后应用）', { exact: true }).click();
  await page.getByLabel('设置覆盖（JSON）', { exact: true }).fill('{"pool_mib":32}');
  await page.getByText('详细状态与排错信息', { exact: true }).click();
  const before = await page.evaluate(() => {
    const raw = document.querySelector('[data-key=raw]'),
      pre = raw.querySelector('pre');
    window.savedRaw = raw;
    pre.scrollTop = 100;
    raw.querySelector('summary').focus({ preventScroll: true });
    const range = document.createRange();
    range.setStart(pre.firstChild, 0);
    range.setEnd(pre.firstChild, 10);
    getSelection().removeAllRanges();
    getSelection().addRange(range);
    window.scrollTo(0, Math.max(0, document.documentElement.scrollHeight - innerHeight - 50));
    return {
      y: scrollY,
      top: pre.scrollTop,
      text: pre.textContent,
      selected: getSelection().toString(),
      rpc: window.rpcCompleted,
    };
  });
  // 等待真实自动轮询，核对展开内容、选区、焦点、输入草稿和两层滚动都未重置。
  await page.waitForFunction((n) => window.rpcCompleted > n, before.rpc, { timeout: 15000 });
  const after = await page.evaluate(() => ({
    same: window.savedRaw === document.querySelector('[data-key=raw]'),
    open: window.savedRaw.open,
    y: scrollY,
    top: window.savedRaw.querySelector('pre').scrollTop,
    text: window.savedRaw.querySelector('pre').textContent,
    selected: getSelection().toString(),
    focused: document.activeElement === window.savedRaw.querySelector('summary'),
    draft: document.querySelector('textarea').value,
  }));
  assert.equal(after.same, true);
  assert.equal(after.open, true);
  assert.equal(after.focused, true);
  assert.equal(after.text, before.text);
  assert.equal(after.selected, before.selected);
  assert.equal(after.top, before.top);
  assert.ok(Math.abs(after.y - before.y) <= 2);
  assert.equal(after.draft, '{"pool_mib":32}');
  await page.screenshot({ path: resolve(output, 'desktop.png'), fullPage: true });
  await page.getByRole('button', { name: '暂停刷新', exact: true }).click();
  failStatus = true;
  await page.getByRole('button', { name: '继续刷新', exact: true }).click();
  await page.getByRole('alert').filter({ hasText: '模拟连接中断' }).waitFor();
  await page.getByRole('button', { name: '暂停刷新', exact: true }).click();
  failStatus = false;
  await page.getByRole('button', { name: '继续刷新', exact: true }).click();
  await page.getByText('连接已恢复，已读取最新状态。', { exact: true }).waitFor();
  await page.getByLabel('并发环境（1–128）', { exact: true }).fill('129');
  await page.getByRole('button', { name: '应用并发量', exact: true }).click();
  await page.getByRole('alert').filter({ hasText: '须为 1–128 的整数' }).waitFor();
  assert.equal(calls.filter((action) => action === 'concurrency').length, 1);
  await page.setViewportSize({ width: 390, height: 844 });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.screenshot({ path: resolve(output, 'mobile.png'), fullPage: true });
  await page.getByRole('button', { name: '暂停', exact: true }).click();
  await page.getByText('暂停请求已接收', { exact: true }).waitFor();
  assert.deepEqual(errors, []);
  const checks = [
    '真实回调与鉴权 API',
    '回车提交并发 2→3',
    '采样进程实际切换',
    '自动刷新保留滚动/展开/选区/焦点/草稿',
    '连接恢复',
    '非法输入未提交',
    '390px',
  ];
  await writeFile(resolve(output, 'report.json'), JSON.stringify({ checks, calls }, null, 2));
  console.log(JSON.stringify({ output, checks }));
} finally {
  await browser.close();
  child.stdin.end();
  if (child.exitCode === null)
    await new Promise((done) => {
      const timeout = setTimeout(() => {
        child.kill();
        done();
      }, 20000);
      child.once('exit', () => {
        clearTimeout(timeout);
        done();
      });
    });
}
