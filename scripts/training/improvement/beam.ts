import { ensure } from '../../../src/engine/core/state';
import type { Command, Player } from '../../../src/engine/types';
import type { Observation } from '../../../src/ai/types';
import { TrainingActionTree } from '../../../src/ai/training/action-tree';
import { encodeDecision } from '../../../src/ai/training/encoding/decision';
import type {
  DecodeOptions,
  DecodeResult,
  PolicyEvaluator,
} from '../../../src/ai/training/decoder';

export interface CompleteCandidate {
  command: Command;
  logProbability: number;
  path: DecodeResult['path'];
  leafStatus: 'available' | 'uncertain';
}
export interface BeamResult extends DecodeResult {
  candidates: CompleteCandidate[];
  beam: {
    width: number;
    pruned: number;
    stopped?: string;
    retainedMass: number;
  };
}

/**
 * 按条件概率乘积扩展完整命令，有限束宽只保留公开动作树中的合法补全。
 * 每条前缀的累计对数概率是其后代的上界；叶子与未完成参数不直接比较成命令。
 * 工作量固定计数，记录裁剪与已保留质量，不声称覆盖全域；模型价值头不参与排序。
 * 输入不可变，所有推理仅接收编码白名单；取消/推理错误不能退回已完成的旧结果。
 */
export async function beamDecode(
  observation: Observation,
  actor: Player,
  evaluate: PolicyEvaluator,
  options: DecodeOptions & { width?: number; candidates?: number } = {},
): Promise<BeamResult> {
  const width = options.width ?? 16,
    count = options.candidates ?? 8,
    maxNodes = options.maxNodes ?? 256,
    maxEvaluations = options.maxEvaluations ?? 32;
  ensure(
    [width, count, maxNodes].every((n) => Number.isSafeInteger(n) && n > 0),
    '束预算无效。',
  );
  ensure(Number.isSafeInteger(maxEvaluations) && maxEvaluations >= 0, '推理预算无效。');
  const stats: DecodeResult['stats'] = {
    nodes: 0,
    evaluations: 0,
    forcedNodes: 0,
    emptyBranches: 0,
    backtracks: 0,
    maxDepth: 0,
    maxEntities: 0,
    maxCandidates: 0,
    treeMs: 0,
    encodingMs: 0,
    inferenceMs: 0,
  };
  const result: BeamResult = {
    status: 'paused',
    value: null,
    path: [],
    stats,
    candidates: [],
    beam: { width, pruned: 0, retainedMass: 0 },
  };
  if (options.signal?.aborted) return { ...result, reason: 'cancelled' };
  const started = performance.now();
  const tree = new TrainingActionTree(observation, actor);
  stats.treeMs += performance.now() - started;
  const frontier: {
    cursor: number[];
    path: DecodeResult['path'];
    logProbability: number;
    order: number;
  }[] = [{ cursor: [], path: [], logProbability: 0, order: 0 }];
  const complete = new Map<string, CompleteCandidate>();
  let serial = 1;
  while (frontier.length) {
    if (options.signal?.aborted) return { ...result, reason: 'cancelled' };
    frontier.sort((a, b) => b.logProbability - a.logProbability || a.order - b.order);
    const ranked = [...complete.values()].sort((a, b) => b.logProbability - a.logProbability);
    if (ranked.length >= count && ranked[count - 1].logProbability >= frontier[0].logProbability)
      break;
    if (stats.nodes >= maxNodes) {
      result.beam.stopped = 'node-budget';
      break;
    }
    const prefix = frontier.shift()!;
    if (prefix.cursor.length > 256) return { ...result, reason: 'depth-limit' };
    const nodeStart = performance.now();
    const node = tree.node(prefix.cursor);
    stats.treeMs += performance.now() - nodeStart;
    stats.nodes++;
    stats.maxDepth = Math.max(stats.maxDepth, prefix.cursor.length);
    stats.maxCandidates = Math.max(stats.maxCandidates, node.choices.length);
    if (!node.choices.length) {
      stats.emptyBranches++;
      continue;
    }
    let logp = [0];
    if (node.choices.length === 1) stats.forcedNodes++;
    else {
      if (stats.evaluations >= maxEvaluations) {
        result.beam.stopped = 'inference-budget';
        break;
      }
      const encodingStart = performance.now();
      const input = encodeDecision(observation, actor, node);
      stats.encodingMs += performance.now() - encodingStart;
      stats.maxEntities = Math.max(stats.maxEntities, input.entities.length);
      stats.evaluations++;
      const inferenceStart = performance.now();
      let output;
      try {
        output = await evaluate(input, options.signal);
      } catch (error) {
        return {
          ...result,
          reason: options.signal?.aborted ? 'cancelled' : 'inference-error',
          error: String(error),
        };
      } finally {
        stats.inferenceMs += performance.now() - inferenceStart;
      }
      if (options.signal?.aborted) return { ...result, reason: 'cancelled' };
      if (
        !Array.isArray(output.logits) ||
        output.logits.length !== node.choices.length ||
        !output.logits.every(Number.isFinite)
      )
        return { ...result, reason: 'invalid-output' };
      const high = Math.max(...output.logits);
      const normalizer =
        high + Math.log(output.logits.reduce((sum, v) => sum + Math.exp(v - high), 0));
      logp = output.logits.map((v) => v - normalizer);
    }
    for (const [selected, choice] of node.choices.entries()) {
      const logProbability = prefix.logProbability + logp[selected];
      const path = [...prefix.path, { stage: node.stage, selected, key: choice.key }];
      if (choice.next)
        frontier.push({
          cursor: [...prefix.cursor, selected],
          path,
          logProbability,
          order: serial++,
        });
      else {
        const key = JSON.stringify(choice.command);
        const previous = complete.get(key);
        // 同一规范命令若有多条树路径，显式累加质量，不能重复占据候选槽。
        const combined = previous
          ? Math.max(previous.logProbability, logProbability) +
            Math.log1p(Math.exp(-Math.abs(previous.logProbability - logProbability)))
          : logProbability;
        complete.set(key, {
          command: choice.command,
          path,
          logProbability: combined,
          leafStatus: choice.status as 'available' | 'uncertain',
        });
      }
    }
    frontier.sort((a, b) => b.logProbability - a.logProbability || a.order - b.order);
    result.beam.pruned += Math.max(0, frontier.length - width);
    frontier.length = Math.min(width, frontier.length);
  }
  if (options.signal?.aborted) return { ...result, reason: 'cancelled' };
  result.candidates = [...complete.values()]
    .sort((a, b) => b.logProbability - a.logProbability)
    .slice(0, count);
  result.beam.retainedMass = result.candidates.reduce(
    (sum, c) => sum + Math.exp(c.logProbability),
    0,
  );
  const best = result.candidates[0];
  if (!best)
    return {
      ...result,
      reason:
        result.beam.stopped === 'inference-budget'
          ? 'inference-budget'
          : result.beam.stopped === 'node-budget'
            ? 'node-budget'
            : 'no-command',
    };
  return {
    ...result,
    status: 'command',
    command: best.command,
    path: best.path,
    leafStatus: best.leafStatus,
  };
}
