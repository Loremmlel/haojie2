// 冻结完整 TS 依赖图和已有原生程序；旧版本只读 Git 对象，不导入工作树中的新实现。
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { cpus } from 'node:os';
import { join, relative, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { build } from 'esbuild';

const { values } = parseArgs({
  options: {
    output: { type: 'string' },
    ref: { type: 'string' },
    executable: { type: 'string' },
  },
});
assert.ok(values.output);
const git = (...args: string[]) => execFileSync('git', args, { encoding: 'utf8' });
const ref = values.ref ? git('rev-parse', '--verify', `${values.ref}^{commit}`).trim() : undefined;
const output = resolve(values.output);
mkdirSync(output);
const hash = (value: string | Uint8Array) => createHash('sha256').update(value).digest('hex');
const sources: Record<string, string> = {};
const source = (path: string) => {
  const content = ref ? git('show', `${ref}:${path}`) : readFileSync(path, 'utf8');
  sources[path] = hash(content);
  return content;
};
const result = await build({
  entryPoints: ['scripts/training/performance/api.ts'],
  bundle: true,
  platform: 'node',
  format: 'esm',
  target: 'node22',
  write: false,
  plugins: [
    {
      name: 'frozen-sources',
      setup(b) {
        b.onLoad({ filter: /\.ts$/ }, ({ path }) => ({
          contents: source(relative(process.cwd(), path).replaceAll('\\', '/')),
          loader: 'ts',
        }));
      },
    },
  ],
});
for (const path of git('ls-tree', '-r', '--name-only', ref ?? 'HEAD', 'native/engine-prototype')
  .trim()
  .split('\n'))
  if (/\.(rs|toml|lock|json)$/.test(path)) source(path);
// 未提交的新 Rust 模块同样进入当前源码指纹。
if (!ref)
  for (const path of git(
    'ls-files',
    '--others',
    '--exclude-standard',
    'native/engine-prototype/src',
  )
    .trim()
    .split('\n'))
    if (path.endsWith('.rs')) source(path);
writeFileSync(join(output, 'api.mjs'), result.outputFiles[0].contents, { flag: 'wx' });
if (values.executable) copyFileSync(values.executable, join(output, 'engine.exe'));
const manifest = {
  ref,
  head: git('rev-parse', 'HEAD').trim(),
  node: process.version,
  cpu: cpus()[0].model,
  bundleSha256: hash(result.outputFiles[0].contents),
  executableSha256: values.executable ? hash(readFileSync(values.executable)) : undefined,
  sources,
};
writeFileSync(join(output, 'manifest.json'), JSON.stringify(manifest, null, 2), { flag: 'wx' });
console.log(JSON.stringify({ ...manifest, sources: Object.keys(sources).length }));
