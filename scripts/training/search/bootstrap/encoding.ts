import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { once } from 'node:events';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Observation } from '../../../../src/ai/types';
import type { Command, Player } from '../../../../src/engine/types';
import { TrainingActionTree } from '../../../../src/ai/training/action-tree';
import { encodeDecision } from '../../../../src/ai/training/encoding/decision';
import { ENCODING_SCHEMA } from '../../../../src/ai/training/encoding/schema';
import { HAOJIE_RULESET } from '../../../../src/engine/online/player-view';
import { encodingSourceHash } from '../../encode';
import { readTrainingRecords } from '../../records/replay';

interface Visit {
  command: Command;
  visits: number;
  probability: number;
}

/**
 * 将真实根访问分布投影到实际执行命令的解码路径，每个前缀按到达质量条件归一。
 * 不把未经过的前缀伪造成决策；价值仍仅在实际根step=0监督一次。
 * 所有候选经共享动作树验证，不读取正式随机状态；分布是教师候选子集上的条件目标。
 */
export function searchExamples(o: Observation, actor: Player, command: Command, policy: Visit[]) {
  assert.ok(policy.length > 0 && policy.length <= 8);
  assert.equal(new Set(policy.map((p) => JSON.stringify(p.command))).size, policy.length);
  assert.equal(
    policy.reduce((n, p) => n + p.visits, 0),
    16,
  );
  assert.ok(
    policy.every(
      (p) => Number.isInteger(p.visits) && p.visits > 0 && p.probability === p.visits / 16,
    ),
  );
  assert.ok(policy.some((p) => JSON.stringify(p.command) === JSON.stringify(command)));
  const tree = new TrainingActionTree(o, actor);
  const traces = policy.map((p) => ({ trace: tree.trace(p.command), mass: p.probability }));
  return tree.trace(command).map(({ node, selected }, step) => {
    const target = Array(node.choices.length).fill(0) as number[];
    for (const { trace, mass } of traces) {
      const same = trace.find((t) => JSON.stringify(t.node.cursor) === JSON.stringify(node.cursor));
      if (same) target[same.selected] += mass;
    }
    const reach = target.reduce((a, b) => a + b, 0);
    assert.ok(reach > 0 && target[selected] > 0);
    return {
      step,
      stage: node.stage,
      selected,
      policy: target.map((p) => p / reach),
      prefixMass: reach,
      input: encodeDecision(o, actor, node),
    };
  });
}

/** 只接受自对弈分组，拒绝评测局；回退命令不伪造访问标签，结束状态仍由共享重放边界确定。 */
export async function* encodeSearchFile(path: string) {
  yield {
    type: 'encoding',
    format: 'haojie-encoded-jsonl-v1',
    schema: ENCODING_SCHEMA,
    source_sha256: createHash('sha256')
      .update(encodingSourceHash())
      .update(readFileSync(fileURLToPath(import.meta.url), 'utf8').replaceAll('\r\n', '\n'))
      .digest('hex'),
    ruleset: HAOJIE_RULESET,
    policy_source: 'teacher-assisted-conditional-visits-v1',
  };
  for await (const r of readTrainingRecords(path)) {
    if (r.type === 'game') {
      assert.equal(r.experimentKind, 'selfplay', '评测种子不能进入搜索训练编码');
      assert.ok(
        ['teacher-assisted-restricted-puct-v1', 'teacher-assisted-restricted-puct-v2'].includes(
          r.policyKind,
        ),
      );
      yield {
        type: 'game',
        game: r.game,
        group: `${r.ruleset}:${r.rules}:${r.seed}`,
        game_id: r.gameId,
        rules: r.rules,
        seed: r.seed,
        source: 'search-selfplay',
        policy_source: 'teacher-assisted-conditional-visits-v1',
        ...(r.valueModelSha256
          ? { value_model_sha256: r.valueModelSha256, neural_leaf_scale: r.neuralLeafScale }
          : {}),
      };
    } else if (r.type === 'sample') {
      if (r.policyMode !== 'search') {
        assert.equal(r.searchPolicy, null);
        continue;
      }
      for (const row of searchExamples(r.observation, r.actor, r.command, r.searchPolicy))
        yield {
          type: 'example',
          game: r.game,
          index: r.index,
          actor: r.actor,
          command: r.command.type,
          ...row,
        };
    } else if (r.type === 'outcome') {
      yield {
        type: 'outcome',
        game: r.game,
        commands: r.commands,
        terminated: r.terminated,
        truncated: r.truncated,
        interrupted: !!r.interrupted,
        winner: r.winner,
        returns: r.returns,
      };
    }
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  for await (const row of encodeSearchFile(process.argv[2]))
    if (!process.stdout.write(JSON.stringify(row) + '\n')) await once(process.stdout, 'drain');
}
