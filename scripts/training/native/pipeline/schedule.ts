/** Python就绪批次调度的TS对应：每源一个请求，不等待尚未就绪的其他源，不依赖墙钟。 */
export async function readyBatches<T>(
  sources: { read(): Promise<T | undefined>; close(): void }[],
  evaluate: (batch: { index: number; value: T }[]) => Promise<void>,
) {
  const waiting = new Map<number, Promise<void>>();
  const ready: { index: number; value?: T; error?: unknown; failed: boolean }[] = [];
  function request(index: number) {
    waiting.set(
      index,
      sources[index].read().then(
        (value) => {
          ready.push({ index, value, failed: false });
        },
        (error) => {
          ready.push({ index, error, failed: true });
        },
      ),
    );
  }
  try {
    sources.forEach((_, index) => request(index));
    while (waiting.size) {
      await Promise.race(waiting.values());
      const completed = ready.splice(0).sort((a, b) => a.index - b.index);
      const batch: { index: number; value: T }[] = [];
      for (const result of completed) {
        waiting.delete(result.index);
        if (result.failed) throw result.error;
        if (result.value !== undefined) batch.push({ index: result.index, value: result.value });
      }
      if (batch.length) await evaluate(batch);
      for (const { index } of batch) request(index);
    }
  } finally {
    // 关闭自己的源，唤醒在途读取；已关闭的源不得再提交策略结果。
    sources.forEach((source) => source.close());
  }
}
