import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { CATALOG, COMBAT_RULES, RULESET_ID } from '../../../src/engine/catalog';

export const protocol = 'haojie-native-engine-v2';

/** 实验用常驻进程，一次一个请求；超时或异常直接失败，禁止隐式回退到 TS。 */
export async function nativeClient(executable: string) {
  const child = spawn(executable, [], { stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true });
  let stderr = '';
  let failure: Error | undefined;
  child.stderr.on('data', (chunk) => {
    stderr = (stderr + chunk).slice(-4096);
  });
  child.on('error', (error) => {
    failure = error;
  });
  const lines = createInterface({ input: child.stdout });
  const iterator = lines[Symbol.asyncIterator]();
  async function request(value: unknown): Promise<any> {
    if (failure) throw failure;
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      return await Promise.race([
        (async () => {
          await new Promise<void>((resolve, reject) =>
            child.stdin.write(JSON.stringify(value) + '\n', (error) =>
              error ? reject(error) : resolve(),
            ),
          );
          const line = await iterator.next();
          assert.ok(!line.done, `Rust 进程提前退出：${failure ?? stderr}`);
          const result = JSON.parse(line.value);
          assert.ok(!result.error, `Rust 协议失败：${result.error}`);
          return result;
        })(),
        new Promise<never>((_, reject) => {
          timer = setTimeout(() => {
            child.kill();
            reject(new Error('Rust 请求超过60秒'));
          }, 60_000);
        }),
      ]);
    } finally {
      clearTimeout(timer);
    }
  }
  const close = () => {
    lines.close();
    child.stdin.end();
    child.kill();
  };
  try {
    const hello = await request({
      op: 'init',
      protocol,
      ruleset: RULESET_ID,
      catalog: CATALOG,
      combat: COMBAT_RULES,
    });
    assert.equal(hello.protocol, protocol);
    assert.equal(hello.ruleset, RULESET_ID);
    assert.equal(hello.completeEngine, false);
  } catch (error) {
    close();
    throw error;
  }
  return { request, close };
}
