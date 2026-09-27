import { createHash } from 'node:crypto';
import { copyFileSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { cpus } from 'node:os';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import { build } from 'esbuild';

/** 新目录冻结源文件指纹、驱动和可执行文件；基准只运行复制后的二进制。 */
export async function freeze(entry: string, output: string, executable: string) {
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
  const native = 'native/engine-prototype';
  const paths = [
    ...Object.keys(built.metafile.inputs),
    ...readdirSync(join(native, 'src'), { recursive: true })
      .filter((p) => String(p).endsWith('.rs'))
      .map((p) => join(native, 'src', String(p))),
    join(native, 'Cargo.toml'),
    join(native, 'Cargo.lock'),
  ];
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
    sources,
    executableSha256: hash(readFileSync(target)),
  });
  return { executable: target, save };
}
