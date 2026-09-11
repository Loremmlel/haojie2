import { build } from 'esbuild';
import { mkdir, writeFile } from 'node:fs/promises';
const result = await build({
  entryPoints: ['scripts/vfx/preview.tsx'],
  bundle: true,
  write: false,
  outdir: 'artifacts/vfx',
  format: 'iife',
  platform: 'browser',
  minify: true,
  define: { 'process.env.NODE_ENV': '"production"' },
});
const css = result.outputFiles.find((f) => f.path.endsWith('.css')).text;
const js = result.outputFiles.find((f) => f.path.endsWith('.js')).text;
await mkdir('artifacts', { recursive: true });
await writeFile(
  'artifacts/vfx-preview.html',
  `<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>浩劫 · 特效工坊</title><style>body{margin:0}${css}</style><div id="root"></div><script>${js.replace(/<\/script/gi, '<\\/script')}</script></html>`,
);
console.log('Developer-only preview: artifacts/vfx-preview.html');
