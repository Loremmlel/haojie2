import { commandDomain } from './parameters';
import type { Step } from './parameters';
import type { Command } from '../../engine/types';
import type { Observation } from '../types';

/** Rust 分层解码的同算法参考；各参数选择不改变实际局面。 */
export function decoderFrontier(observation: Observation, path: number[]) {
  const { actions, inspect, expand } = commandDomain(observation);
  const steps = actions.map(
    (a) =>
      [
        ...(a.materialIds?.length
          ? Array.from({ length: 3 }, () => ({ kind: 'material' as const }))
          : []),
        ...a.steps,
        ...(a.chosenKinds?.length ? [{ kind: 'chosen' as const }] : []),
        ...(a.command.type === 'attack' && !a.command.path
          ? [{ kind: 'direction' as const, label: '' }]
          : []),
      ] satisfies Step[],
  );
  type Node = {
    command: Command;
    stage: number;
    group?: number[];
    action?: number;
    offset?: number;
    depth?: number;
  };
  function possible(a: number, c: Command, offset: number, depth: number): boolean {
    if (depth > 256) throw new Error('命令参数深度超出规则表示');
    if (offset === steps[a].length) return inspect(c).status !== 'invalid';
    return expand(c, actions[a], steps[a][offset], offset + 1 === steps[a].length).some(
      ([next, repeat]) => possible(a, next, offset + Number(!repeat), depth + 1),
    );
  }
  function children(node: Node): Node[] {
    if (node.stage === 0) {
      const groups = new Map<string, Node>();
      actions.forEach((a, i) => {
        if (!possible(i, a.command, 0, 0)) return;
        const c = a.command,
          key = c.unitId ? `unit:${c.unitId}` : c.cardId ? `card:${c.cardId}` : 'global';
        if (groups.has(key)) groups.get(key)!.group!.push(i);
        else
          groups.set(key, {
            command: {
              type: '',
              // 与来源分组键一致，单位来源不能绑定组内第一张手牌。
              ...(c.unitId ? { unitId: c.unitId } : c.cardId ? { cardId: c.cardId } : {}),
            } as unknown as Command,
            stage: 1,
            group: [i],
          });
      });
      return [...groups.values()];
    }
    if (node.stage === 1)
      return node.group!.map((a) => ({
        command: actions[a].command,
        stage: steps[a].length ? 2 : 3,
        action: a,
        offset: 0,
        depth: 0,
      }));
    if (node.stage === 3) return [];
    const a = node.action!,
      offset = node.offset!,
      depth = node.depth!;
    return expand(node.command, actions[a], steps[a][offset], offset + 1 === steps[a].length)
      .map(([c, repeat]) => ({
        command: c,
        stage: offset + Number(!repeat) === steps[a].length ? 3 : 2,
        action: a,
        offset: offset + Number(!repeat),
        depth: depth + 1,
      }))
      .filter((n) => possible(a, n.command, n.offset, n.depth));
  }
  let node: Node = { command: { type: '' } as unknown as Command, stage: 0 };
  for (const index of path) {
    const next = children(node)[index];
    if (!next) throw new Error('无效解码前缀');
    node = next;
  }
  const step = node.stage === 2 ? steps[node.action!][node.offset!] : undefined;
  const focus =
    node.stage === 0
      ? 'source'
      : step
        ? step.kind === 'target'
          ? (step.field ?? 'targetId')
          : step.kind
        : 'action';
  return node.stage === 3
    ? { leaf: true, command: node.command }
    : { leaf: false, stage: node.stage, focus, choices: children(node).map((n) => n.command) };
}
