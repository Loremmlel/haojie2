import { createInterface } from 'node:readline';
import { once } from 'node:events';
import { TrainingService } from './protocol';

/** 常驻JSONL接口；stdout只写协议，遵守背压，EOF退出后全部内存环境释放。 */
const service = new TrainingService();
const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of lines) {
  if (!line.trim()) continue;
  let reply: unknown;
  try {
    if (Buffer.byteLength(line) > 4 * 1024 * 1024) throw new Error('单行请求超过4 MiB。');
    reply = service.handle(JSON.parse(line));
  } catch (error) {
    reply = { id: null, ok: false, error: error instanceof Error ? error.message : String(error) };
  }
  if (!process.stdout.write(JSON.stringify(reply) + '\n')) await once(process.stdout, 'drain');
}
