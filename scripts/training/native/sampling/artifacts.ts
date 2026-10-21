import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { copyFileSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { cpus } from 'node:os';
import { join, relative } from 'node:path';
import { execFileSync } from 'node:child_process';
import { build } from 'esbuild';

/** 与 build.rs 一样收集实际源码和规则包，模块增删时无需同步手写清单。 */
export function nativeSourcePaths(native = 'native/engine') {
  return [
    ...['Cargo.toml', 'Cargo.lock', 'build.rs'].map((name) => join(native, name)),
    ...['src', 'data'].flatMap((directory) =>
      readdirSync(join(native, directory), {
        recursive: true,
        withFileTypes: true,
      })
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
  writeFileSync(join(output, 'runner.mjs'), built.outputFiles[0].contents, {
    flag: 'wx',
  });
  const nativePaths = nativeSourcePaths(native);
  const nativeBytes = new Map(nativePaths.map((path) => [path, readFileSync(path)]));
  const sourceHash = createHash('sha256');
  const field = (bytes: Buffer) => {
    const length = Buffer.alloc(8);
    length.writeBigUInt64LE(BigInt(bytes.length));
    sourceHash.update(length).update(bytes);
  };
  // 与 build.rs 一致，名称和原始文件字节分别带 u64 小端长度；换行也属于身份。
  for (const path of nativePaths) {
    field(Buffer.from(relative(native, path).replaceAll('\\', '/'), 'utf8'));
    field(nativeBytes.get(path)!);
  }
  const nativeSourceSha256 = sourceHash.digest('hex');
  const paths = [...Object.keys(built.metafile.inputs), ...nativePaths];
  const hash = (v: Buffer) => createHash('sha256').update(v).digest('hex');
  const target = join(output, 'engine.exe');
  copyFileSync(executable, target);
  const identity = JSON.parse(
    execFileSync(target, ['--version'], {
      encoding: 'utf8',
      windowsHide: true,
    }),
  );
  assert.equal(
    identity.build?.sourceSha256,
    nativeSourceSha256,
    '冻结拒绝：可执行文件的 sourceSha256 与当前原始源码不一致，请先重新构建。',
  );
  const sources = Object.fromEntries(
    paths.map((p) => [p, hash(nativeBytes.get(p) ?? readFileSync(p))]),
  );
  const save = (name: string, value: unknown) =>
    writeFileSync(join(output, name), JSON.stringify(value, null, 2), {
      flag: 'wx',
    });
  save('manifest.json', {
    head: execFileSync('git', ['rev-parse', 'HEAD'], {
      encoding: 'utf8',
    }).trim(),
    cpu: cpus()[0].model,
    node: process.version,
    nativeSource: native,
    nativeSourceSha256,
    sources,
    executableSha256: hash(readFileSync(target)),
  });
  return { executable: target, save };
}
