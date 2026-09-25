import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import type { Observation } from '../../../src/ai/types';
import { reference } from './reference';
import { search } from './puct';

interface Protocol {
  horizon: number;
  referenceMaxActions: number;
  maxActionNodes: number;
  searchSeeds: number[];
}

function recheck(observation: Observation, protocol: Protocol) {
  let oracle: ReturnType<typeof reference> | undefined;
  let failure: string | undefined;
  try {
    oracle = reference(observation, protocol.horizon, {
      maxActions: protocol.referenceMaxActions,
      maxActionNodes: protocol.maxActionNodes,
    });
  } catch (error) {
    failure = error instanceof Error ? error.message : String(error);
  }
  return {
    oracle,
    failure,
    repeated: search(observation, {
      simulations: 128,
      horizon: protocol.horizon,
      sampleSeed: protocol.searchSeeds[0],
      maxActionNodes: protocol.maxActionNodes,
    }),
  };
}

/**
 * 按独立公开局面动态分发复核，结果仍按输入索引返回；不读正式随机状态或写实验产物。
 * 每个孩子一次只做一个位置，固定种子/预算不受完成顺序影响；任一孩子异常时停止全部孩子。
 */
export async function parallelRechecks(
  observations: Observation[],
  protocol: Protocol,
  workers = 4,
) {
  assert.ok(Number.isSafeInteger(workers) && workers > 0 && workers <= 16);
  const results: ReturnType<typeof recheck>[] = Array(observations.length);
  const children: ReturnType<typeof spawn>[] = [];
  let next = 0;
  let completed = 0;
  let active = 0;
  let maxActive = 0;
  const started = performance.now();
  try {
    await Promise.all(
      Array.from(
        { length: Math.min(workers, observations.length) },
        () =>
          new Promise<void>((resolve, reject) => {
            const child = spawn(
              process.execPath,
              ['--import', 'tsx', fileURLToPath(import.meta.url), '--worker'],
              {
                stdio: ['ignore', 'ignore', 'inherit', 'ipc'],
                windowsHide: true,
              },
            );
            children.push(child);
            let assigned: number | undefined;
            let done = false;
            const dispatch = () => {
              if (next === observations.length) {
                done = true;
                child.disconnect();
                resolve();
                return;
              }
              assigned = next++;
              active++;
              maxActive = Math.max(maxActive, active);
              child.send!({ index: assigned, observation: observations[assigned], protocol });
            };
            child.on('error', reject);
            child.on('exit', (code) => {
              if (!done) reject(new Error(`审计子进程提前退出：${code}`));
            });
            child.on('message', (message: any) => {
              try {
                assert.equal(message.index, assigned);
                assert.ok(!message.error, message.error);
                assert.ok(message.result);
                results[assigned!] = message.result;
                active--;
                completed++;
                console.log(
                  JSON.stringify({
                    type: 'recheck-complete',
                    index: assigned,
                    completed,
                    total: observations.length,
                  }),
                );
                dispatch();
              } catch (error) {
                reject(error);
              }
            });
            dispatch();
          }),
      ),
    );
    assert.equal(completed, observations.length);
    return { results, workers: children.length, maxActive, elapsedMs: performance.now() - started };
  } finally {
    for (const child of children) child.kill();
  }
}

if (process.argv.includes('--worker')) {
  process.on('disconnect', () => process.exit(0));
  process.on('message', ({ index, observation, protocol }: any) => {
    try {
      process.send!({ index, result: recheck(observation, protocol) });
    } catch (error) {
      process.send!({ index, error: error instanceof Error ? error.message : String(error) });
    }
  });
}
