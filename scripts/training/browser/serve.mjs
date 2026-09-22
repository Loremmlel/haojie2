import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const { values } = parseArgs({
  options: { data: { type: 'string' }, port: { type: 'string', default: '4318' } },
});
if (!values.data) throw new Error('请指定 --data 导出产物目录');
const port = Number(values.port);
if (!Number.isInteger(port) || port < 1024 || port > 65535) throw new Error('无效端口');
const here = fileURLToPath(new URL('.', import.meta.url));
const data = resolve(values.data);
const runtime = resolve('node_modules/onnxruntime-web/dist');
const files = new Map([
  ['/', [resolve(here, 'index.html'), 'text/html; charset=utf-8']],
  ['/client.mjs', [resolve(here, 'client.mjs'), 'text/javascript; charset=utf-8']],
  ['/ort.mjs', [resolve(runtime, 'ort.webgpu.bundle.min.mjs'), 'text/javascript']],
  ['/runtime.wasm', [resolve(runtime, 'ort-wasm-simd-threaded.asyncify.wasm'), 'application/wasm']],
  ['/manifest.json', [resolve(data, 'manifest.json'), 'application/json']],
  ['/cases.json', [resolve(data, 'cases.json'), 'application/json']],
  ...['fp32', 'mixed-fp16'].map((precision) => [
    `/${precision}.onnx`,
    [resolve(data, `${precision}.onnx`), 'application/octet-stream'],
  ]),
]);

// 仅开放明确列出的测试资产，不提供文件系统浏览、任意路径或转发功能。
createServer(async (request, response) => {
  if (request.headers.host !== `127.0.0.1:${port}` || request.method !== 'GET') {
    response.writeHead(403).end();
    return;
  }
  const asset = files.get(request.url);
  if (!asset) {
    response.writeHead(404).end();
    return;
  }
  try {
    const body = await readFile(asset[0]);
    response
      .writeHead(200, {
        'Content-Type': asset[1],
        'Cache-Control': 'no-store',
        'X-Content-Type-Options': 'nosniff',
        'Content-Security-Policy':
          "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'unsafe-inline'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'",
      })
      .end(body);
  } catch (error) {
    console.error(error.message);
    response.writeHead(500).end('测试资产不可用；先运行ONNX导出');
  }
}).listen(port, '127.0.0.1', () => console.log(`模型推理实验：http://127.0.0.1:${port}`));
