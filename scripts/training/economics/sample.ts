import { TrainingActionTree } from '../../../src/ai/training/action-tree';
import { encodeDecision } from '../../../src/ai/training/encoding/decision';
import type { Observation } from '../../../src/ai/types';
import type { Command, Player } from '../../../src/engine/types';
import { TinyPolicy } from './policy';

export const emptyMetrics = () => ({
  treeMs: 0,
  encodingMs: 0,
  inferenceMs: 0,
  samplingMs: 0,
  observationMs: 0,
  stepMs: 0,
  recordMs: 0,
  nodes: 0,
  evaluations: 0,
  forced: 0,
  backtracks: 0,
  maxEntities: 0,
  maxCandidates: 0,
  offTurnCommands: 0,
  offTurnPasses: 0,
});
export type Metrics = ReturnType<typeof emptyMetrics>;

/**
 * 只在一个公开局面内采样动作参数；空参数分支回溯，不推演未来局面、不做棋力搜索。
 * 回合外给合法巨大化显式的“不介入”候选（特征63），不伪造引擎pass命令。
 * 固定节点上限耗尽即中断并报告；墙钟只由外层在完整命令边界停止。
 */
export function sampleCommand(
  observation: Observation,
  actor: Player,
  policy: TinyPolicy | null,
  random: () => number,
  metrics: Metrics,
  optional = false,
): { command?: Command; passed?: boolean; error?: string } {
  let start = performance.now();
  const tree = new TrainingActionTree(observation, actor);
  metrics.treeMs += performance.now() - start;
  let nodes = 0;
  const visit = (cursor: number[]): Command | 'pass' | undefined => {
    if (++nodes > 4096 || cursor.length > 256) throw new Error('参数解码预算耗尽');
    metrics.nodes++;
    start = performance.now();
    const node = tree.node(cursor);
    metrics.treeMs += performance.now() - start;
    const canPass = optional && cursor.length === 0;
    if (!node.choices.length) return canPass ? 'pass' : undefined;
    metrics.maxCandidates = Math.max(metrics.maxCandidates, node.choices.length + Number(canPass));
    let order = node.choices.map((_, i) => i);
    if (canPass) order.push(node.choices.length);
    if (order.length === 1) metrics.forced++;
    else {
      start = performance.now();
      const input = encodeDecision(observation, actor, node);
      if (canPass) {
        const pass = Array<number>(64).fill(0);
        pass[63] = 1;
        input.candidates.push(pass);
        input.candidate_mask.push(true);
        input.sources.push(-1);
        input.targets.push(-1);
      }
      metrics.encodingMs += performance.now() - start;
      metrics.maxEntities = Math.max(metrics.maxEntities, input.entities.length);
      metrics.evaluations++;
      start = performance.now();
      const logits = policy ? policy.logits(input) : input.candidates.map(() => 0);
      metrics.inferenceMs += performance.now() - start;
      if (logits.length !== order.length || !logits.every(Number.isFinite))
        throw new Error('网络输出无效');
      start = performance.now();
      const scores = logits.map((value) => value - Math.log(-Math.log(random())));
      order = order.sort((a, b) => scores[b] - scores[a]);
      metrics.samplingMs += performance.now() - start;
    }
    for (const i of order) {
      if (canPass && i === node.choices.length) return 'pass';
      const choice = node.choices[i];
      if (!choice.next) return choice.command;
      const result = visit([...cursor, i]);
      if (result) return result;
      metrics.backtracks++;
    }
    return undefined;
  };
  const result = visit([]);
  return result === 'pass'
    ? { passed: true }
    : result
      ? { command: result }
      : { error: '没有完整合法命令' };
}
