import { build } from 'esbuild';
import { readFile, writeFile } from 'node:fs/promises';

const result = await build({
  entryPoints: ['training/ui/main.tsx'],
  bundle: true,
  write: false,
  outdir: 'console',
  format: 'iife',
  platform: 'browser',
  target: ['es2022'],
  minify: true,
  define: { 'process.env.NODE_ENV': '"production"' },
  legalComments: 'inline',
});
const js = result.outputFiles.find((f) => f.path.endsWith('.js')).text;
const css = result.outputFiles.find((f) => f.path.endsWith('.css')).text;
const html = `<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>浩劫 · 训练台</title><style>${css}</style></head><body class="hj-game training-console"><div id="root"></div><script>${js.replace(/<\/script/gi, '<\\/script')}</script></body></html>\n`;
const path = 'training/haojie/web/index.html';
if (process.argv.includes('--check')) {
  if ((await readFile(path, 'utf8')) !== html)
    throw new Error('训练页面产物过期，请运行 npm run training:web:build');
} else await writeFile(path, html, 'utf8');
console.log(`Training console: ${Buffer.byteLength(html).toLocaleString()} bytes, single HTML`);
