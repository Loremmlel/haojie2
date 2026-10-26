/** 完整命令级随机 MCTS；环境负责公开观察与独立随机转移，模型只提供先验和价值。 */
export interface SearchPosition<S> {
  state: S;
  key: string;
  actor: 1 | 2;
  value: number;
  terminal: boolean;
  cutoff?: boolean;
}
export interface SearchAction<A> {
  action: A;
  key: string;
  prior: number;
}
export interface SearchDomain<S, A> {
  sample(position: SearchPosition<S>, seed: number): Promise<SearchAction<A>>;
  step(position: SearchPosition<S>, action: A, seed: number): Promise<SearchPosition<S>>;
}

/** 完整命令访问数按已选前缀条件化；未访问前缀返回 null，由调用者重新搜索。 */
export function conditionSearchVisits(
  actions: { action: number[]; visits: number }[],
  cursor: number[],
  prior: number[],
  mask: boolean[],
) {
  const counts = prior.map(() => 0);
  let total = 0;
  for (const { action, visits } of actions) {
    if (action.length <= cursor.length || !cursor.every((index, i) => index === action[i]))
      continue;
    const next = action[cursor.length];
    if (next < counts.length && mask[next]) {
      counts[next] += visits;
      total += visits;
    }
  }
  if (!total) return null;
  const sum = prior.reduce((a, b) => a + b, 0);
  const target = counts.map((count) => count / total);
  return { target, mass: target.map((value, i) => 0.95 * value + (0.05 * prior[i]) / sum) };
}
interface Edge<S, A> extends SearchAction<A> {
  visits: number;
  total: number;
  outcomes: Map<string, Node<S, A>>;
}
interface Node<S, A> {
  position: SearchPosition<S>;
  visits: number;
  edges: Edge<S, A>[];
}

/** 和原生宿主使用相同的独立 uint32 序列；不接受真实局面的 seed/rng。 */
export function searchRandom(seed: number) {
  let value = seed >>> 0 || 1;
  return () => {
    value ^= value << 13;
    value ^= value >>> 17;
    value ^= value << 5;
    return value >>> 0;
  };
}

/** 只从公开全局特征和参数前缀派生场景；与原生宿主逐个 uint32 混合一致。 */
export function publicSearchSeed(
  row: { ids: ArrayLike<bigint>; present: bigint; integers: bigint; continuous: ArrayLike<number> },
  cursor: number[],
) {
  let seed = 2166136261;
  const mix = (value: number) => {
    seed = Math.imul(seed ^ value, 16777619) >>> 0;
  };
  for (const value of [...Array.from(row.ids), row.present, row.integers]) {
    const bits = BigInt.asUintN(64, value);
    mix(Number(bits & 0xffffffffn));
    mix(Number(bits >> 32n));
  }
  for (const value of Array.from(row.continuous)) mix(value);
  for (const index of cursor) mix(index);
  return seed || 1;
}

export async function searchCommands<S, A>(
  position: SearchPosition<S>,
  domain: SearchDomain<S, A>,
  simulations = 64,
  depth = 64,
  seed = 1,
  cancelled = () => false,
) {
  if (
    !Number.isSafeInteger(simulations) ||
    simulations < 1 ||
    !Number.isSafeInteger(depth) ||
    depth < 1
  )
    throw new Error('搜索预算须为正整数。');
  const make = (p: SearchPosition<S>): Node<S, A> => ({ position: p, visits: 0, edges: [] });
  const root = make(position);
  const random = searchRandom(seed);
  let expanded = 1;
  let deepest = 0;
  let opponentNodes = 0;
  const run = async (node: Node<S, A>, level: number): Promise<number> => {
    deepest = Math.max(deepest, level);
    if (cancelled()) throw new Error('搜索已取消。');
    if (node.position.terminal || node.position.cutoff || level === depth)
      return node.position.value;
    // 渐进扩宽从完整动作域采样，不用教师分数或固定候选上限裁剪合法动作。
    if (node.edges.length < Math.floor(Math.sqrt(node.visits)) + 1) {
      const choice = await domain.sample(node.position, random());
      if (!node.edges.some((edge) => edge.key === choice.key))
        node.edges.push({ ...choice, visits: 0, total: 0, outcomes: new Map() });
    }
    const prior = node.edges.reduce((sum, edge) => sum + edge.prior, 0);
    const sign = node.position.actor === position.actor ? 1 : -1;
    const score = (edge: Edge<S, A>) =>
      sign * (edge.visits ? edge.total / edge.visits : node.position.value) +
      (1.5 * edge.prior * Math.sqrt(node.visits + 1)) / (prior * (edge.visits + 1));
    let selected = node.edges[0];
    for (const edge of node.edges) if (score(edge) > score(selected)) selected = edge;
    const next = await domain.step(node.position, selected.action, random());
    let child = selected.outcomes.get(next.key);
    let value: number;
    if (child) value = await run(child, level + 1);
    else {
      child = make(next);
      selected.outcomes.set(next.key, child);
      ++expanded;
      if (!next.terminal && !next.cutoff && next.actor !== position.actor) ++opponentNodes;
      deepest = Math.max(deepest, level + 1);
      value = next.value;
    }
    ++selected.visits;
    selected.total += value;
    ++node.visits;
    return value;
  };
  if (!position.terminal) for (let i = 0; i < simulations; ++i) await run(root, 0);
  return {
    actions: root.edges.map(({ action, visits, prior }) => ({ action, visits, prior })),
    expanded,
    deepest,
    opponentNodes,
    simulations: root.visits,
  };
}
