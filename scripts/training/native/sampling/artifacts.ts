import { createHash } from 'node:crypto';
import { copyFileSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { cpus } from 'node:os';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import { build } from 'esbuild';

/** 与 build.rs 一样收集实际源码和规则包，模块增删时无需同步手写清单。 */
export function nativeSourcePaths(native = 'native/engine') {
  return [
    ...['Cargo.toml', 'Cargo.lock', 'build.rs'].map((name) => join(native, name)),
    ...['src', 'data'].flatMap((directory) =>
      readdirSync(join(native, directory), { recursive: true, withFileTypes: true })
        .filter((entry) => entry.isFile())
        .map((entry) => join(entry.parentPath, entry.name)),
    ),
  ]
    .map((path) => path.replaceAll('\\', '/'))
    .sort();
}

/** 新目录冻结源文件指纹、驱动和可执行文件；基准只运行复制后的二进制。 */
export async function freeze(
  entry: string,
  output: string,
  executable: string,
  native = 'native/engine',
) {
  mkdirSync(output, { recursive: false });
  const built = await build({
    entryPoints: [entry],
    bundle: true,
    platform: 'node',
    format: 'esm',
    target: 'node22',
    packages: 'external',
    write: false,
    metafile: true,
  });
  writeFileSync(join(output, 'runner.mjs'), built.outputFiles[0].contents, { flag: 'wx' });
  const paths = [...Object.keys(built.metafile.inputs), ...nativeSourcePaths(native)];
  const hash = (v: Buffer) => createHash('sha256').update(v).digest('hex');
  const target = join(output, 'engine.exe');
  copyFileSync(executable, target);
  const sources = Object.fromEntries(paths.map((p) => [p, hash(readFileSync(p))]));
  const save = (name: string, value: unknown) =>
    writeFileSync(join(output, name), JSON.stringify(value, null, 2), { flag: 'wx' });
  save('manifest.json', {
    head: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
    cpu: cpus()[0].model,
    node: process.version,
    nativeSource: native,
    sources,
    executableSha256: hash(readFileSync(target)),
  });
  return { executable: target, save };
}
