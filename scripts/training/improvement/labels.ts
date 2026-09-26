import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { once } from 'node:events';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { readCorrections } from '../corrections/records';
import { hashRecordFile, readRecordLines, withRecordOutput } from '../records/io';
import { encodingSourceHash } from '../encode';
import { decide } from '../../../src/ai/planning/search';
import { canonicalTrainingCommand, TrainingActionTree } from '../../../src/ai/training/action-tree';
import { encodeDecision } from '../../../src/ai/training/encoding/decision';
import { ENCODING_SCHEMA } from '../../../src/ai/training/encoding/schema';
import { HAOJIE_RULESET } from '../../../src/engine/online/player-view';
import type { Command, Player } from '../../../src/engine/types';
import type { Observation } from '../../../src/ai/types';

const FORMAT = 'haojie-ranked-corrections-v2';
const scriptHash = () =>
  createHash('sha256')
    .update(readFileSync(fileURLToPath(import.meta.url)))
    .digest('hex');

/** 只把同评估边界的完整教师候选排序当作软策略，不把分数解释为胜率或反事实终局。 */
export function rankedPolicy(o: Observation, actor: Player, command: Command, nodes = 800) {
  const canonical = (c: Command) => canonicalTrainingCommand(o, actor, c);
  const key = (c: Command) => JSON.stringify(canonical(c));
  const chosen = canonical(command);
  const decision = decide(o, actor, 'hard', {
    simulations: nodes,
    mode: 'work',
    trace: true,
  });
  assert.ok(decision.command);
  assert.equal(key(decision.command), key(chosen), '重算教师与纠错标签不一致');
  const alternatives = decision.trace?.alternatives ?? [];
  const reference = alternatives.find((a) => key(a.command) === key(chosen));
  const candidates =
    reference && reference.stage !== 'static'
      ? alternatives
          .filter((a) => a.stage === reference.stage && Number.isFinite(a.score))
          .sort((a, b) => b.score - a.score)
          .filter((a, i, all) => all.findIndex((b) => key(b.command) === key(a.command)) === i)
          .slice(0, 8)
      : [];
  if (candidates.length < 2)
    return {
      stage: reference?.stage ?? null,
      policy: [{ command: chosen, probability: 1 }],
    };
  candidates.sort((a, b) => b.score - a.score);
  const weight = candidates.map((a) =>
    Math.exp(-candidates.filter((b) => b.score > a.score).length / 2),
  );
  const total = weight.reduce((a, b) => a + b, 0);
  const policy = candidates.map((a, i) => ({
    command: canonical(a.command),
    probability: (0.2 * weight[i]) / total + (key(a.command) === key(chosen) ? 0.8 : 0),
  }));
  assert.ok(policy.some((p) => key(p.command) === key(chosen)));
  assert.ok(Math.abs(policy.reduce((sum, p) => sum + p.probability, 0) - 1) < 1e-9);
  return { stage: reference!.stage, policy };
}

/** 沿真实教师命令的路径按到达质量条件归一；其它候选只贡献软质量，不伪造额外访问轨迹。 */
export function softExamples(
  o: Observation,
  actor: Player,
  command: Command,
  policy: { command: Command; probability: number }[],
) {
  assert.ok(
    policy.length > 0 && policy.every((p) => p.probability > 0 && Number.isFinite(p.probability)),
  );
  assert.ok(Math.abs(policy.reduce((n, p) => n + p.probability, 0) - 1) < 1e-9);
  assert.equal(new Set(policy.map((p) => JSON.stringify(p.command))).size, policy.length);
  const tree = new TrainingActionTree(o, actor);
  const traces = policy.map((p) => ({
    trace: tree.trace(p.command),
    mass: p.probability,
  }));
  return tree.trace(command).map(({ node, selected }, step) => {
    const target = Array(node.choices.length).fill(0) as number[];
    for (const { trace, mass } of traces) {
      const at = trace.find((t) => JSON.stringify(t.node.cursor) === JSON.stringify(node.cursor));
      if (at) target[at.selected] += mass;
    }
    const reach = target.reduce((a, b) => a + b, 0);
    assert.ok(reach > 0 && target[selected] > 0);
    return {
      step,
      stage: node.stage,
      selected,
      policy: target.map((p) => p / reach),
      input: encodeDecision(o, actor, node),
    };
  });
}

async function generate(input: string, output: string) {
  const sha256 = await hashRecordFile(input);
  let labels = 0,
    soft = 0;
  await withRecordOutput(output, async (emit) => {
    await emit({
      type: 'ranked',
      format: FORMAT,
      source: {
        path: relative(dirname(resolve(output)), resolve(input)),
        sha256,
      },
      sourceSha256: encodingSourceHash(),
      scriptSha256: scriptHash(),
      chosenMass: 0.8,
      rankTemperature: 2,
    });
    for await (const row of readCorrections(input)) {
      if (row.type !== 'sample') continue;
      const target = rankedPolicy(row.observation, row.actor, row.command);
      await emit({
        type: 'label',
        game: row.game,
        index: row.index,
        before: row.before,
        actor: row.actor,
        command: row.command,
        ...target,
      });
      labels++;
      soft += Number(target.policy.length > 1);
      if (labels % 32 === 0) console.error(JSON.stringify({ labels, soft }));
    }
    assert.equal(await hashRecordFile(input), sha256);
    await emit({ type: 'complete', labels, soft });
  });
  console.log(JSON.stringify({ output, labels, soft }));
}

/** 引用的纠错与原始实局均须通过共享重放验证；软标签不改变编码语义或价值遮罩。 */
async function* encode(path: string) {
  let header: any, footer: any;
  const labels = new Map<string, any>();
  for await (const row of readRecordLines(path)) {
    assert.ok(!footer);
    if (!header) {
      assert.equal(row.format, FORMAT);
      header = row;
    } else if (row.type === 'label') {
      const key = `${row.game}:${row.index}`;
      assert.ok(!labels.has(key));
      labels.set(key, row);
    } else {
      assert.equal(row.type, 'complete');
      footer = row;
    }
  }
  assert.ok(header && footer && labels.size > 0);
  assert.equal(footer.labels, labels.size);
  assert.equal(header.sourceSha256, encodingSourceHash());
  assert.equal(header.scriptSha256, scriptHash());
  const source = resolve(dirname(path), header.source.path);
  assert.equal(await hashRecordFile(source), header.source.sha256);
  yield {
    type: 'encoding',
    format: 'haojie-encoded-jsonl-v1',
    schema: ENCODING_SCHEMA,
    source_sha256: encodingSourceHash(),
    ruleset: HAOJIE_RULESET,
  };
  let count = 0;
  for await (const row of readCorrections(source)) {
    if (row.type === 'game') {
      count = 0;
      yield {
        type: 'game',
        game: row.game,
        game_id: row.gameId,
        group: `${row.ruleset}:${row.rules}:${row.seed}`,
        difficulty: row.difficulty,
        source: 'ranked-correction',
        rules: row.rules,
        seed: row.seed,
      };
    } else if (row.type === 'sample') {
      const key = `${row.game}:${row.index}`,
        label = labels.get(key);
      assert.ok(label);
      assert.equal(label.before, row.before);
      assert.equal(label.actor, row.actor);
      assert.deepEqual(label.command, row.command);
      for (const example of softExamples(row.observation, row.actor, row.command, label.policy))
        yield {
          type: 'example',
          game: row.game,
          index: count,
          actor: row.actor,
          command: row.command.type,
          ...example,
        };
      count++;
      labels.delete(key);
    } else if (row.type === 'outcome') yield { ...row, commands: count };
  }
  assert.equal(labels.size, 0);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { values } = parseArgs({
    options: {
      input: { type: 'string' },
      output: { type: 'string' },
      encode: { type: 'string' },
    },
  });
  if (values.encode) {
    for await (const row of encode(values.encode))
      if (!process.stdout.write(JSON.stringify(row) + '\n')) await once(process.stdout, 'drain');
  } else {
    assert.ok(values.input && values.output);
    await generate(values.input, values.output);
  }
}
