import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { applyCommand, parseSession, type Command } from '../../src/engine';
import { observe, fingerprint } from '../../src/ai/observation';
import { TrainingActionTree } from '../../src/ai/training/action-tree';
import { encodeDecision } from '../../src/ai/training/encoding/decision';
import { encodePosition } from '../../src/ai/training/encoding/state';
import { ROLES } from '../../src/ai/training/encoding/schema';
import { trainingGeometry } from '../../src/ai/training/queries';
import { TrainingEnvironment } from '../../src/match/training';
import { add, card, fixture } from '../helpers';

test('79条当前正式回放命令均能分解并恢复，实际随机结果保持一致', () => {
  const [header, ...rows] = readFileSync('docs/playtests/cli-counter-freeze-20260921.jsonl', 'utf8')
    .trim()
    .split('\n')
    .map((line) => JSON.parse(line));
  let state = parseSession(JSON.stringify(header.initial)).present;
  for (const [i, row] of rows.entries()) {
    const o = observe(state, row.owner);
    const before = structuredClone(o);
    const tree = new TrainingActionTree(o, row.owner);
    const root = encodeDecision(o, row.owner, tree.node());
    let trace;
    try {
      trace = tree.trace(row.command);
    } catch (e) {
      throw new Error(`回放命令${i}: ${JSON.stringify(row.command)}`, { cause: e });
    }
    for (const { node, selected } of trace) {
      const encoded = encodeDecision(o, row.owner, node);
      assert.ok(encoded.candidate_mask[selected]);
      for (const id of [...encoded.sources, ...encoded.targets])
        assert.ok(id === -1 || encoded.entity_mask[id]);
      assert.equal(encoded.entities.length, encoded.kinds.length);
      assert.ok(encoded.entities.flat().every(Number.isFinite));
    }
    assert.deepEqual(encodeDecision(o, row.owner, tree.node()), root, '标签不能改变输入或候选');
    const last = trace.at(-1)!;
    const decoded = last.node.choices[last.selected].command;
    const next = applyCommand(state, decoded);
    assert.deepEqual(next, applyCommand(state, row.command), `命令${i}结算漂移`);
    assert.equal(fingerprint(observe(next)), row.after);
    assert.deepEqual(o, before);
    state = next;
  }
});

test('ID重命名和正式随机数不改变特征；未知嵌套字段不能静默进入训练', () => {
  const s = fixture();
  const a = add(s, 1, 1, 3, 4),
    b = add(s, 2, 2, 4, 4);
  a.attacked = [b.id];
  a.effects.push({ type: 'mark', from: 1, until: 9, owner: 2, sourceId: b.id, amount: 3 });
  const o = observe(s);
  const expected = encodeDecision(o, 1, new TrainingActionTree(o, 1).node());
  const renamed = JSON.parse(
    JSON.stringify(o).replaceAll(a.id, 'renamed-A').replaceAll(b.id, 'renamed-B'),
  );
  renamed.serial = 99999;
  assert.deepEqual(encodeDecision(renamed, 1, new TrainingActionTree(renamed, 1).node()), expected);
  s.seed = 98765;
  s.rng = 23456;
  assert.deepEqual(
    encodeDecision(observe(s), 1, new TrainingActionTree(observe(s), 1).node()),
    expected,
  );
  assert.throws(() => encodePosition(s, 1), /禁止携带/);
  Object.assign(o.units[0].effects[0], { rng: 123 });
  assert.throws(() => encodePosition(o, 1), /未编码字段 rng/);
});

test('时钟快照的附属效果保持独立，较大公开局面不截断', () => {
  const s = fixture();
  const u = add(s, 1, 1, 3, 4);
  u.effects = [{ type: 'mark', from: 1, until: 9, owner: 2, amount: 2 }];
  s.clockFrames = {
    1: { previous: { ply: 3, turns: { 1: 2, 2: 1 }, units: [structuredClone(u)] } },
    2: {},
  };
  for (let i = 0; i < 80; i++)
    s.deaths.push({ id: `dead-${i}`, kind: 1, owner: 2, ply: i, revived: false });
  const first = encodePosition(observe(s), 1);
  assert.ok(first.entities.length > 80);
  const effects = first.entities.filter((r) => r[0] === (ROLES.indexOf('effect') + 1) / 32);
  assert.equal(effects.length, 2);
  assert.notEqual(effects[0][3], effects[1][3], '当前实例与旧快照必须有不同的附属状态归属');
  s.clockFrames[1].previous!.units[0].effects[0].amount = 5;
  assert.notDeepEqual(encodePosition(observe(s), 1).entities, first.entities);
  s.clockFrames[1].previous!.units[0].effects[0].amount = 2;
  s.deaths.at(-1)!.revived = true;
  assert.notDeepEqual(encodePosition(observe(s), 1).entities, first.entities);
});

test('完整材料池、手绘路径和对方死亡反应都可编码，无教师裁剪', () => {
  const s = fixture();
  s.phase = 'synthesis';
  const ids = [2, 3, 4, 6, 7].map((x) => add(s, 'u21', 1, x, 4).id);
  const o = observe(s);
  const command: Command = { type: 'synthesize', recipeId: 'sage', materialIds: ids.slice(2) };
  Object.assign(command, trainingGeometry(o, command).points![0]);
  const trace = new TrainingActionTree(o, 1).trace(command);
  assert.equal(trace[1].node.choices.length, 5);
  trace.forEach(({ node }) => encodeDecision(o, 1, node));
  assert.ok(
    applyCommand(s, trace.at(-1)!.node.choices[trace.at(-1)!.selected].command).units.some(
      (u) => u.kind === 'sage',
    ),
  );

  const h = fixture();
  const carrier = add(h, 26, 1, 2, 5);
  add(h, 1, 2, 3, 5);
  add(h, 1, 2, 4, 6);
  const equipped = applyCommand(h, { type: 'equip', cardId: card(h, 'u28'), targetId: carrier.id });
  const path: Command = {
    type: 'attack',
    unitId: carrier.id,
    path: [
      { x: 2, y: 5 },
      { x: 3, y: 5 },
      { x: 3, y: 6 },
      { x: 4, y: 6 },
    ],
  };
  const heart = observe(equipped);
  const steps = new TrainingActionTree(heart, 1).trace(path);
  assert.ok(steps.length >= 6);
  steps.forEach(({ node }) => encodeDecision(heart, 1, node));
  assert.equal(steps.at(-1)!.node.choices[steps.at(-1)!.selected].key, 'commit-path');

  const r = fixture();
  const attacker = add(r, 26, 1, 3, 4),
    victim = add(r, 2, 2, 3, 5);
  victim.hp = 1;
  const reaction = observe(
    applyCommand(r, { type: 'attack', unitId: attacker.id, targetId: victim.id }),
    2,
  );
  const reactions = new TrainingActionTree(reaction, 2).trace({ type: 'react' });
  const encoded = encodeDecision(reaction, 2, reactions[0].node);
  assert.ok(encoded.sources[reactions[0].selected] >= 0, '已死亡反应源仍有公开快照');
});

test('神龛暗选不会编码对方尚未揭示的选择', () => {
  const env = new TrainingEnvironment({ seed: 90, rules: 'shrine' });
  env.step(1, { type: 'choose-shrine', shrineKind: 's9', parity: 'even' });
  const o = env.observation(2);
  const tree = new TrainingActionTree(o, 2);
  encodeDecision(o, 2, tree.node());
  o.shrineDraft!.choices[1] = { kind: 's9', parity: 'even' };
  assert.throws(() => encodePosition(o, 2), /泄露/);
});
