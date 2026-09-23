import { createReadStream, createWriteStream, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { createHash } from 'node:crypto';
import { once } from 'node:events';
import { PassThrough } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { createInterface } from 'node:readline';
import { createGzip, createGunzip } from 'node:zlib';

/** 独占创建记录；背压覆盖压缩和磁盘，失败保留可诊断的部分文件，不覆盖旧实验。 */
export async function withRecordOutput<T>(
  path: string | undefined,
  run: (emit: (row: unknown) => Promise<void>) => Promise<T>,
): Promise<T> {
  if (!path) return run(async () => {});
  mkdirSync(dirname(path), { recursive: true });
  const output = createWriteStream(path, { flags: 'wx' });
  await once(output, 'open');
  const input = new PassThrough();
  const done = path.endsWith('.gz')
    ? pipeline(input, createGzip(), output)
    : pipeline(input, output);
  // 写入者可能仍在计算；立即接管异步错误，最终由 await done 抛出。
  void done.catch(() => {});
  try {
    const result = await run(async (row) => {
      if (input.destroyed) await done;
      if (!input.write(JSON.stringify(row) + '\n')) await once(input, 'drain');
    });
    input.end();
    await done;
    return result;
  } catch (error) {
    // 生成器失败也封好压缩尾部，已有完整命令仍可作为中断轨迹读取。
    input.end();
    await done.catch(() => {});
    throw error;
  }
}

/** 压缩损坏和源读取错误同样失败；消费者提前退出时关闭整条流。 */
export async function* readRecordLines(path: string) {
  const input = new PassThrough();
  const source = createReadStream(path);
  const done = path.endsWith('.gz')
    ? pipeline(source, createGunzip(), input)
    : pipeline(source, input);
  void done.catch(() => {});
  const lines = createInterface({ input, crlfDelay: Infinity });
  try {
    for await (const line of lines) if (line.trim()) yield JSON.parse(line);
    await done;
  } finally {
    lines.close();
    input.destroy();
    await done.catch(() => {});
  }
}

export async function hashRecordFile(path: string) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}
