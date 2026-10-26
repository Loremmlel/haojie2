import assert from 'node:assert/strict';
import test from 'node:test';
import {
  publicSearchSeed,
  searchCommands,
  type SearchDomain,
  type SearchPosition,
} from '../../src/ai/training/mcts';
import { FeatureRow } from '../../src/ai/training/encoding/schema';
import { searchTrainingCommand, type SearchModel } from '../../src/ai/training/command-search';
import { TrainingActionTree } from '../../src/ai/training/action-tree';
import { observe } from '../../src/ai/observation';
import { add, fixture } from '../helpers';

test('公开搜索场景保留完整 int64 特征，与原生 uint32 混合结果一致', () => {
  const row = new FeatureRow();
  row.ids[3] = 9007199254740993n;
  row.present = 4097n;
  row.integers = 7n;
  row.continuous[0] = 0x3f80;
  assert.equal(publicSearchSeed(row, [2, 1]), 32189891);
});

test('完整命令 MCTS 保留同方连续行动，并识别对手回应后的陷阱', async () => {
  const position = (state: string): SearchPosition<string> => ({
    state,
    key: state,
    actor: state === 'reply' ? 2 : 1,
    value: state === 'loss' ? -1 : state === 'win' ? 1 : state === 'reply' ? 0.9 : 0,
    terminal: ['loss', 'win', 'safe'].includes(state),
  });
  const domain: SearchDomain<string, string> = {
    async sample(p, seed) {
      const action =
        p.state === 'root'
          ? seed & 1
            ? 'trap'
            : 'safe'
          : p.state === 'prep'
            ? 'next'
            : seed & 1
              ? 'punish'
              : 'blunder';
      return { action, key: action, prior: 0.5 };
    },
    async step(p, action) {
      return position(
        p.state === 'root'
          ? action === 'trap'
            ? 'prep'
            : 'safe'
          : p.state === 'prep'
            ? 'reply'
            : action === 'punish'
              ? 'loss'
              : 'win',
      );
    },
  };
  const result = await searchCommands(position('root'), domain, 512, 16, 73127);
  const visits = Object.fromEntries(result.actions.map((a) => [a.action, a.visits]));
  assert.ok(visits.safe > visits.trap);
  assert.ok(result.opponentNodes > 0);
  assert.ok(result.deepest >= 3);
  assert.equal(result.simulations, 512);
  assert.deepEqual(visits, { trap: 21, safe: 491 });
  assert.equal(result.expanded, 6);
});

test('随机后继按实际观察分开，搜索不把最幸运分支当作确定结果', async () => {
  const position = (state: string): SearchPosition<string> => ({
    state,
    key: state,
    actor: 1,
    value: state === 'win' ? 1 : state === 'loss' ? -1 : 0,
    terminal: state !== 'root',
  });
  const domain: SearchDomain<string, string> = {
    async sample(_, seed) {
      const action = seed & 1 ? 'risky' : 'safe';
      return { action, key: action, prior: 0.5 };
    },
    async step(_, action, seed) {
      return position(action === 'safe' ? 'safe' : seed % 8 === 0 ? 'win' : 'loss');
    },
  };
  const a = await searchCommands(position('root'), domain, 512, 8, 123);
  const b = await searchCommands(position('root'), domain, 512, 8, 123);
  assert.deepEqual(a, b);
  assert.deepEqual(Object.fromEntries(a.actions.map((x) => [x.action, x.visits])), {
    safe: 490,
    risky: 22,
  });
  assert.ok(
    a.actions.find((x) => x.action === 'safe')!.visits >
      a.actions.find((x) => x.action === 'risky')!.visits,
  );
});

test('公开完整命令搜索从零价值先验找到真实基地终结，保持实局与随机状态不变', async () => {
  const state = fixture();
  const unit = add(state, 9, 1, 5, 10);
  unit.mode = 'attack';
  state.bases[2] = 10;
  const before = JSON.stringify(state);
  const observation = observe(state, 1);
  const root = new TrainingActionTree(observation, 1).node([]);
  const model: SearchModel = async ({ node }) => ({
    mass: node.mask!.map((enabled) => Number(enabled)),
    value: 0,
  });
  const result = await searchTrainingCommand(observation, model, {
    simulations: 64,
    depth: 16,
    seed: 73127,
  });
  const attack = root.choices.findIndex((choice) => choice.command.type === 'attack');
  assert.ok(attack >= 0);
  assert.ok(result.target[attack] > 0.5);
  assert.equal(result.simulations, 64);
  assert.equal(JSON.stringify(state), before);
  const cached = await searchTrainingCommand(observation, model, {
    simulations: 64,
    cursor: [attack],
    plan: result.plan,
  });
  assert.equal(cached.simulations, 0);
});

test('强制动作族之后重新计算参数先验，不把一项根分布用于四个落点', async () => {
  const state = fixture();
  const source = add(state, 'u12', 1, 3, 5);
  state.pending = [{ kind: 'bounce', owner: 1, source, amount: 30 }];
  const observation = observe(state, 1);
  const tree = new TrainingActionTree(observation, 1);
  assert.equal(tree.node([]).choices.length, 1);
  assert.equal(tree.node([0]).choices.length, 4);
  const result = await searchTrainingCommand(
    observation,
    async ({ node }) => ({
      mass: node.mask!.map(Number),
      value: 0,
    }),
    { simulations: 16, depth: 4, seed: 73127 },
  );
  assert.equal(result.simulations, 16);
  assert.equal(result.target.length, 1);
});
