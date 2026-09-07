import { build } from 'esbuild';
import { mkdir, writeFile, readdir, readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

/** Bundle both JavaScript and CSS into one HTML. No CDN, chunks or runtime fetches. */
export async function buildGame() {
  const result = await build({
    entryPoints: ['src/main.tsx'],
    bundle: true,
    write: false,
    outdir: 'dist',
    format: 'iife',
    platform: 'browser',
    target: ['es2022'],
    minify: true,
    sourcemap: false,
    legalComments: 'inline',
    define: { 'process.env.NODE_ENV': '"production"' },
    metafile: true,
    logLevel: 'warning',
  });
  const script = result.outputFiles.find((f) => f.path.endsWith('.js'))?.text;
  const styles = result.outputFiles.find((f) => f.path.endsWith('.css'))?.text ?? '';
  if (!script || result.outputFiles.some((f) => !/\.(js|css)$/.test(f.path)))
    throw new Error('Build must contain only one script and inline styles.');
  if (Object.values(result.metafile.outputs).some((o) => o.imports.some((i) => i.external)))
    throw new Error('External runtime imports are forbidden.');
  const notices = await Promise.all(
    ['react', 'react-dom', 'scheduler'].map(
      async (name) => `${name}\n${await readFile(`node_modules/${name}/LICENSE`, 'utf8')}`,
    ),
  );
  const html = `<!doctype html>
<html lang="zh-CN"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#f5f4ef">
<meta name="description" content="浩劫：26种普通召唤、28种终极召唤、117格战场，同屏双人回合制战棋。离线可玩，支持悔棋与存档。">
<title>浩劫 2.0 · 双人回合制战棋</title>
<!-- Bundled third-party notices\n${notices.join('\n\n').replace(/-->/g, '-- >')} -->
<style>html,body{margin:0;min-height:100%;background:#f5f4ef}#root{min-height:100vh}${styles.replace(/<\/style/gi, '<\\/style')}</style>
</head><body><div id="root"></div><noscript>浩劫需要启用JavaScript，但不需要联网。</noscript>
<script>${script.replace(/<\/script/gi, '<\\/script')}</script>
</body></html>`;
  await mkdir('dist', { recursive: true });
  const existing = await readdir('dist');
  if (existing.some((name) => name !== 'index.html'))
    throw new Error('dist contains unrelated files; use an empty output directory.');
  await writeFile('dist/index.html', html);
  console.log(
    `Single-file build: dist/index.html (${Buffer.byteLength(html).toLocaleString()} bytes)`,
  );
  return html;
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  await buildGame();
}
