import assert from 'node:assert/strict';
import test from 'node:test';
import { readyBatches } from '../../scripts/training/native/pipeline/schedule';

test('就绪批次不等待慢源，单源保持顺序并在失败时关闭全部输入', async () => {
  let release!: (value: number) => void;
  const delayed = new Promise<number>((resolve) => {
    release = resolve;
  });
  const seen: number[][] = [],
    closed: number[] = [];
  const queues = [[Promise.resolve(1), Promise.resolve(3)], [delayed]];
  const active = [false, false];
  await readyBatches(
    queues.map((values, index) => ({
      async read() {
        assert.equal(active[index], false);
        active[index] = true;
        const value = await values.shift();
        active[index] = false;
        return value;
      },
      close() {
        closed.push(index);
      },
    })),
    async (batch) => {
      seen.push(batch.map((item) => item.value));
      if (seen.length === 1) {
        assert.deepEqual(seen[0], [1]);
        release(2);
      }
    },
  );
  assert.deepEqual(seen.flat().sort(), [1, 2, 3]);
  assert.deepEqual(closed, [0, 1]);
  let stopped = 0;
  let cancel!: () => void;
  const waiting = new Promise<undefined>((resolve) => {
    cancel = () => resolve(undefined);
  });
  await assert.rejects(
    readyBatches(
      [
        {
          read: async () => {
            throw new Error('model-version');
          },
          close() {
            stopped++;
          },
        },
        {
          read: () => waiting,
          close() {
            stopped++;
            cancel();
          },
        },
      ],
      async () => {
        assert.fail('无效输入不得进入前向');
      },
    ),
    /model-version/,
  );
  assert.equal(stopped, 2);
});
