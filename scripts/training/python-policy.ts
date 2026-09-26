import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { createInterface } from 'node:readline';
import { isDeepStrictEqual } from 'node:util';
import type { PolicyEvaluator } from '../../src/ai/training/decoder';
import { ENCODING_SCHEMA } from '../../src/ai/training/encoding/schema';
import { HAOJIE_RULESET } from '../../src/engine/online/player-view';

export interface PolicyProcessOptions {
  python: string;
  checkpoint: string;
  device: string;
  precision: string;
  threads: number;
  timeoutMs: number;
  signal?: AbortSignal;
  /** 显式研究入口；默认仍是拒绝实验检查点的正式推理模块。 */
  module?: string;
}
type Reply = Record<string, any>;

/**
 * 一个本地Python进程、一个在途请求；只传编码张量，不发送观察、种子或教师决策。
 * 握手核对规则/完整编码schema，stdout严格JSONL，stderr仅保留短诊断尾部。
 * 取消/管道错误/超时使进程失效，拒绝迟到回包；超时是故障保护，不用于改选动作。
 */
export class PythonPolicy {
  readonly #child: ChildProcessWithoutNullStreams;
  #pending?: { resolve: (reply: Reply) => void; reject: (error: Error) => void };
  #failure?: Error;
  #stderr = '';
  #id = 0;
  ready!: Reply;
  startupMs = 0;
  totals = { calls: 0, modelMs: 0, validationMs: 0, roundTripMs: 0, bytesSent: 0 };

  private constructor(readonly options: PolicyProcessOptions) {
    this.#child = spawn(
      options.python,
      [
        '-X',
        'utf8',
        '-u',
        '-m',
        options.module ?? 'haojie_training.inference',
        '--checkpoint',
        options.checkpoint,
        '--device',
        options.device,
        '--precision',
        options.precision,
        '--threads',
        String(options.threads),
      ],
      { stdio: 'pipe', windowsHide: true },
    );
    this.#child.stderr.setEncoding('utf8').on('data', (chunk: string) => {
      this.#stderr = (this.#stderr + chunk).slice(-8192);
    });
    this.#child.on('error', (error) => this.#fail(error));
    this.#child.stdin.on('error', (error) => this.#fail(error));
    this.#child.on('exit', (code, signal) => {
      this.#fail(new Error(`推理进程退出（${code ?? signal}）：${this.#stderr}`));
    });
    const lines = createInterface({ input: this.#child.stdout, crlfDelay: Infinity });
    lines.on('line', (line) => {
      try {
        if (Buffer.byteLength(line) > 4 * 1024 * 1024) throw new Error('推理回包超过4MiB');
        const reply = JSON.parse(line);
        if (!reply || typeof reply !== 'object' || Array.isArray(reply) || !this.#pending)
          throw new Error('推理stdout协议或请求顺序错误');
        const waiting = this.#pending;
        this.#pending = undefined;
        waiting.resolve(reply);
      } catch (error) {
        this.#fail(error instanceof Error ? error : new Error(String(error)));
      }
    });
  }

  #fail(error: Error) {
    this.#failure ??= error;
    const waiting = this.#pending;
    this.#pending = undefined;
    waiting?.reject(this.#failure);
    this.#child.kill();
  }

  #receive(signal?: AbortSignal): Promise<Reply> {
    if (this.#failure) throw this.#failure;
    if (this.#pending) throw new Error('同一推理进程不能并发请求');
    return new Promise((resolve, reject) => {
      const cancel = () => this.#fail(new Error('推理已取消'));
      const timer = setTimeout(
        () => this.#fail(new Error('推理进程响应超时')),
        this.options.timeoutMs,
      );
      const finish = () => {
        clearTimeout(timer);
        signal?.removeEventListener('abort', cancel);
      };
      this.#pending = {
        resolve: (reply) => {
          finish();
          resolve(reply);
        },
        reject: (error) => {
          finish();
          reject(error);
        },
      };
      signal?.addEventListener('abort', cancel, { once: true });
      if (signal?.aborted) cancel();
    });
  }

  static async start(options: PolicyProcessOptions) {
    const start = performance.now();
    const policy = new PythonPolicy(options);
    try {
      const ready = await policy.#receive(options.signal);
      if (
        ready.type !== 'ready' ||
        ready.format !== 'haojie-policy-jsonl-v1' ||
        ready.ruleset !== HAOJIE_RULESET ||
        !isDeepStrictEqual(ready.schema, ENCODING_SCHEMA)
      )
        throw new Error('推理进程的协议/规则/编码schema与当前游戏不兼容');
      policy.ready = ready;
      policy.startupMs = performance.now() - start;
      return policy;
    } catch (error) {
      policy.close();
      throw error;
    }
  }

  evaluate: PolicyEvaluator = async (input, signal) => {
    const start = performance.now(),
      id = ++this.#id;
    const line = JSON.stringify({ id, inputs: input }) + '\n';
    const bytes = Buffer.byteLength(line);
    if (bytes > 4 * 1024 * 1024) throw new Error('推理请求超过4MiB，未截断实体');
    const response = this.#receive(signal);
    if (!this.#failure) this.#child.stdin.write(line);
    const reply = await response;
    if (reply.id !== id || reply.ok !== true) throw new Error(reply.error ?? '推理回包id不匹配');
    for (const key of ['model_ms', 'validation_ms'])
      if (!Number.isFinite(reply.timing?.[key]) || reply.timing[key] < 0)
        throw new Error('推理耗时回包无效');
    this.totals.calls++;
    this.totals.bytesSent += bytes;
    this.totals.modelMs += reply.timing.model_ms;
    this.totals.validationMs += reply.timing.validation_ms;
    this.totals.roundTripMs += performance.now() - start;
    return { logits: reply.logits, value: reply.value };
  };

  close() {
    this.#fail(new Error('推理进程已关闭'));
  }
}
