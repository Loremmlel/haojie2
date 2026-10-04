import type { Command, Player } from '../../../engine/types';
import type { Observation } from '../../types';
import { TrainingActionTree } from '../action-tree';
import { createSamplingEncoder, type EncodedDecision } from '../encoding/decision';

export interface McNode {
  stage: string;
  input: EncodedDecision;
  choices: { command: Command; next: boolean; status: string }[];
}
export interface McDecision {
  input: EncodedDecision;
  selected: number;
  depth: number;
  stage: string;
  pass: boolean;
}

/** 与 Rust mc_search 对应：重新采样剩余候选，完整公开尝试日志固定每次输入的条件。
 * 回调只借用本次输入；持久保留须复制。所有选择含失败尝试共享随后实际终局收益。
 */
export function mcSearch(
  nodeAt: (path: number[]) => McNode,
  logits: (input: EncodedDecision) => number[],
  random: () => number,
  optional: boolean,
  accept: (command: Command, path: number[], status: string) => boolean,
  decision: (decision: McDecision) => void,
) {
  const journal: number[][] = [];
  let nodes = 0,
    rejected = 0;
  function event(depth: number, index: number, result: number) {
    const row = Array<number>(64).fill(0);
    row[0] = journal.length / 4096;
    row[1] = depth / 256;
    row[2] = index;
    row[3] = result;
    journal.push(row);
  }
  function visit(
    path: number[],
  ): { command?: Command; pass?: boolean; path: number[] } | undefined {
    if (path.length > 256) throw Error('参数解码预算耗尽');
    const node = nodeAt(path),
      pass = optional && path.length === 0;
    if (!node.choices.length) return pass ? { pass: true, path: [] } : undefined;
    const count = node.choices.length + Number(pass),
      remaining = Array<boolean>(count).fill(true);
    while (remaining.some(Boolean)) {
      if (++nodes > 4096) throw Error('参数解码预算耗尽');
      const input = structuredClone(node.input);
      if (pass) {
        const row = Array<number>(64).fill(0);
        row[63] = 1;
        input.candidates.push(row);
        input.sources.push(-1);
        input.targets.push(-1);
      }
      input.candidate_mask = [...remaining];
      for (const row of journal) {
        input.entities.push([...row]);
        input.kinds.push(250);
        input.entity_mask.push(true);
      }
      let selected = remaining.indexOf(true);
      if (remaining.filter(Boolean).length > 1) {
        const values = logits(input);
        if (values.length !== count || !values.every(Number.isFinite))
          throw Error('invalid policy logits');
        let best = -Infinity;
        values.forEach((value, index) => {
          if (!remaining[index]) return;
          const score = value - Math.log(-Math.log(random()));
          if (score > best) {
            selected = index;
            best = score;
          }
        });
      }
      decision({
        input,
        selected,
        depth: path.length,
        stage: node.stage,
        pass: pass && selected === node.choices.length,
      });
      event(path.length, selected, 1);
      const nextPath = [...path, selected];
      if (pass && selected === node.choices.length) return { pass: true, path: nextPath };
      const choice = node.choices[selected];
      if (!choice.next) {
        if (accept(choice.command, nextPath, choice.status))
          return { command: choice.command, path: nextPath };
        if (++rejected >= 64) throw Error('参数解码预算耗尽');
        event(path.length, selected, -2);
      } else {
        const result = visit(nextPath);
        if (result) return result;
        event(path.length, selected, -1);
      }
      remaining[selected] = false;
    }
    return undefined;
  }
  return visit([]);
}

export function sampleMc(
  observation: Observation,
  actor: Player,
  logits: (input: EncodedDecision) => number[],
  random: () => number,
  optional: boolean,
  accept: (command: Command, path: number[], status: string) => boolean,
  decision: (decision: McDecision) => void,
) {
  const tree = new TrainingActionTree(observation, actor),
    encode = createSamplingEncoder(observation, actor);
  return mcSearch(
    (path) => {
      const node = tree.node(path);
      return {
        stage: node.stage,
        choices: node.choices.map((c) => ({ ...c, next: !!c.next })),
        input: node.choices.length ? encode(node) : ({} as EncodedDecision),
      };
    },
    logits,
    random,
    optional,
    accept,
    decision,
  );
}
