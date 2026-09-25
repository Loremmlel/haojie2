import { ensure } from '../../../src/engine/core/state';
import { allPieces, hasTrait } from '../../../src/engine/core/traits';
import type { Command, Player } from '../../../src/engine/types';
import { decisionOwner, hash } from '../../../src/ai/observation';
import { TrainingActionTree } from '../../../src/ai/training/action-tree';
import { trainingPosition } from '../../../src/ai/training/queries';
import {
  sampleTrainingTransition,
  simulationRandomSource,
} from '../../../src/ai/training/simulation';
import type { Observation } from '../../../src/ai/types';

/** 实验搜索只接受经典公开局面；尚不建模回合外巨大化的抢占时机。 */
export function checkPosition(observation: Observation) {
  const position = trainingPosition(observation);
  ensure(!observation.mode, '搜索探针仅支持经典模式。');
  ensure(observation.phase !== 'shrine-draft', '搜索探针不支持暗选。');
  ensure(!allPieces(position).some((u) => hasTrait(u, 'u7')), '搜索探针不支持回合外巨大化。');
}

/** 完整遍历分步动作树后才返回；上限不足时抛错，不把截断前缀当合法动作子集。 */
export function commands(observation: Observation, maxNodes = 4096) {
  checkPosition(observation);
  const tree = new TrainingActionTree(observation, decisionOwner(observation));
  const result = new Map<string, Command>();
  let nodes = 0;
  const visit = (cursor: number[]) => {
    ensure(++nodes <= maxNodes && cursor.length <= 256, '完整命令枚举超出预算。');
    const node = tree.node(cursor);
    node.choices.forEach((choice, index) => {
      if (choice.next) visit([...cursor, index]);
      else result.set(JSON.stringify(choice.command), choice.command);
    });
  };
  if (!observation.winner) visit([]);
  return { commands: [...result.values()], nodes };
}

export function terminalValue(observation: Observation, player: Player): number | null {
  return observation.winner === undefined
    ? null
    : observation.winner === 'draw'
      ? 0
      : observation.winner === player
        ? 1
        : -1;
}

/** 首轮只跨play及其强制反应；进入下一召唤/合成窗口时估值截断，不模拟新一轮抽牌。 */
export const windowBoundary = (observation: Observation) =>
  !observation.pending.length && observation.phase !== 'play';

interface Edge {
  command: Command;
  visits: number;
  total: number;
  children: Map<string, Node>;
}
interface Node {
  observation: Observation;
  actor: Player;
  visits: number;
  edges?: Edge[];
  evaluated?: boolean;
  initialValue?: number;
}
export interface ProbeProfile {
  enumerationMs: number;
  transitionMs: number;
  leafMs: number;
  leafCalls: number;
}
export interface ProbeOptions {
  simulations: number;
  horizon: number;
  sampleSeed: number;
  maxActionNodes?: number;
  signal?: AbortSignal;
  deferExpansion?: boolean;
  leafValue?: (observation: Observation, rootActor: Player, remainingCommands: number) => number;
  firstPlayValue?: 'zero' | 'parent';
  profile?: ProbeProfile;
}

/**
 * 仅供CLI研究的均匀先验PUCT；完整命令为一条决策边，规则随机结果为抽样机会子节点。
 * 回报始终存根操作者视角，选择时由当前实际操作者决定最大/最小，不逐命令机械翻转。
 * 默认非终局估值0；可注入根操作者视角的有界估计作受控实验，均不产生训练价值标签。
 * 可延迟首次新叶的枚举至第二次访问；不裁剪根命令域。profile只累计耗时，不影响选招预算。
 * firstPlayValue可让未访问边继承父节点估计；避免绝对优势局面中用0低估所有未试动作。
 * 不连接实局、不写文件、不修改输入；非法转移、未支持局面、取消或枚举超限时整个探针暂停。
 * ponytail: 首轮同步CPU、无网络/重用跨决策树；测得收益后再接批量网络与异步取消。
 */
export function search(observation: Observation, options: ProbeOptions) {
  ensure(
    Number.isSafeInteger(options.simulations) && options.simulations >= 0,
    '模拟数须为非负整数。',
  );
  for (const count of [options.horizon, options.maxActionNodes ?? 4096])
    ensure(Number.isSafeInteger(count) && count > 0, '搜索预算须为正整数。');
  const random = simulationRandomSource(options.sampleSeed);
  const rootActor = decisionOwner(observation);
  const root: Node = { observation, actor: rootActor, visits: 0 };
  const measure = <T>(key: 'enumerationMs' | 'transitionMs' | 'leafMs', action: () => T): T => {
    if (!options.profile) return action();
    const started = performance.now();
    try {
      return action();
    } finally {
      options.profile[key] += performance.now() - started;
    }
  };
  const estimate = (o: Observation, depth: number) => {
    if (options.profile) options.profile.leafCalls++;
    return measure('leafMs', () => {
      const value = options.leafValue?.(o, rootActor, Math.max(0, options.horizon - depth)) ?? 0;
      ensure(Number.isFinite(value) && Math.abs(value) <= 1, '叶端估计须为[-1,1]有限数。');
      return value;
    });
  };
  const stats = {
    simulations: 0,
    transitions: 0,
    actionNodes: 0,
    expanded: 0,
    terminalLeaves: 0,
    cutoffLeaves: 0,
    expansionLeaves: 0,
    sameActorEdges: 0,
    changedActorEdges: 0,
    networkCalls: 0,
  };
  const expand = (node: Node) => {
    const result = measure('enumerationMs', () =>
      commands(node.observation, options.maxActionNodes),
    );
    stats.actionNodes += result.nodes;
    stats.expanded++;
    ensure(result.commands.length > 0, '非终局没有完整命令。');
    // 同一个实验编号固定候选顺序；排序不查询结果，也不消费机会采样序列。
    const ordering = simulationRandomSource(
      hash(`${options.sampleSeed}:${JSON.stringify(node.observation)}`),
    );
    const ordered = [...result.commands];
    for (let i = ordered.length - 1; i > 0; i--) {
      const j = Math.floor(ordering() * (i + 1));
      [ordered[i], ordered[j]] = [ordered[j], ordered[i]];
    }
    node.edges = ordered.map((command) => ({ command, visits: 0, total: 0, children: new Map() }));
  };
  const visit = (node: Node, depth: number): number => {
    ensure(!options.signal?.aborted, '搜索已取消。');
    checkPosition(node.observation);
    const ended = terminalValue(node.observation, rootActor);
    if (ended !== null) {
      stats.terminalLeaves++;
      return ended;
    }
    if (depth >= options.horizon || windowBoundary(node.observation)) {
      stats.cutoffLeaves++;
      return estimate(node.observation, depth);
    }
    if (!node.edges) {
      if (!node.evaluated) {
        node.evaluated = true;
        if (!options.deferExpansion) expand(node);
        stats.expansionLeaves++;
        node.initialValue = estimate(node.observation, depth);
        return node.initialValue;
      }
      expand(node);
    }
    const sign = node.actor === rootActor ? 1 : -1;
    const score = (edge: Edge) =>
      sign *
        (edge.visits
          ? edge.total / edge.visits
          : options.firstPlayValue === 'parent'
            ? (node.initialValue ?? 0)
            : 0) +
      ((1.5 / node.edges!.length) * Math.sqrt(node.visits + 1)) / (edge.visits + 1);
    const edge = node.edges!.reduce((best, item) => (score(item) > score(best) ? item : best));
    stats.transitions++;
    const next = measure('transitionMs', () =>
      sampleTrainingTransition(
        node.observation,
        node.actor,
        edge.command,
        Math.floor(random() * 4294967296),
      ),
    );
    const actor = decisionOwner(next);
    if (actor === node.actor) stats.sameActorEdges++;
    else stats.changedActorEdges++;
    const key = JSON.stringify(next);
    let child = edge.children.get(key);
    if (!child) {
      child = { observation: next, actor, visits: 0 };
      edge.children.set(key, child);
    }
    const value = visit(child, depth + 1);
    edge.visits++;
    edge.total += value;
    node.visits++;
    return value;
  };
  try {
    ensure(!options.signal?.aborted, '搜索已取消。');
    checkPosition(observation);
    ensure(!observation.winner, '终局不搜索。');
    ensure(!windowBoundary(observation), '搜索根须位于play或强制反应窗口。');
    if (options.firstPlayValue === 'parent') root.initialValue = estimate(observation, 0);
    expand(root);
    const baseline = root.edges![0].command;
    for (let i = 0; i < options.simulations; i++) {
      visit(root, 0);
      stats.simulations++;
    }
    const edges = root.edges!.map((edge) => ({
      command: edge.command,
      visits: edge.visits,
      value: edge.visits ? edge.total / edge.visits : null,
      chanceOutcomesSeen: edge.children.size,
    }));
    const selected = edges.reduce((best, edge) =>
      edge.visits > best.visits ||
      (edge.visits === best.visits && (edge.value ?? -Infinity) > (best.value ?? -Infinity))
        ? edge
        : best,
    );
    return { status: 'command' as const, command: selected.command, baseline, edges, stats };
  } catch (error) {
    return {
      status: 'paused' as const,
      reason: error instanceof Error ? error.message : String(error),
      stats,
    };
  }
}
