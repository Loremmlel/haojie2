import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { search, searchAsync } from '../../scripts/training/search/puct';
import {
  bootstrapDecision,
  bootstrapValueDecision,
} from '../../scripts/training/search/bootstrap/policy';
import { positions } from '../../scripts/training/search/positions';
import { valueEvaluator } from '../../scripts/training/search/value-cycle/model';
import { encodingSourceHash } from '../../scripts/training/encode';

test('未知召唤边界不与负网络值比较，常数估值不能让三名可攻击单位直接结束', async () => {
  const observation = structuredClone(positions()[0].observation);
  observation.bases[2] = 100;
  for (let i = 1; i <= 2; i++)
    observation.units.push({ ...structuredClone(observation.units[0]), id: `extra${i}`, x: 4 + i });
  const teacher = bootstrapDecision(observation, 93).command;
  assert.equal(teacher.type, 'attack');
  for (const constant of [-0.25, 0, 0.25]) {
    const result = await bootstrapValueDecision(observation, 93, async () => constant);
    assert.deepEqual(result.command, teacher);
    assert.equal(result.mode, 'terminal-search');
    assert.equal(result.policy, null, '不可比较的访问分布不得成为训练标签');
  }
});

test('教师候选访问数打平时不由微小叶值误差改招', () => {
  const observation = structuredClone(positions()[0].observation);
  observation.bases[2] = 100;
  const unit = observation.units[0];
  const candidates = [
    { type: 'move' as const, unitId: unit.id, x: unit.x + 1, y: unit.y },
    { type: 'move' as const, unitId: unit.id, x: unit.x, y: unit.y - 1 },
  ];
  const result = search(observation, {
    simulations: 2,
    horizon: 1,
    sampleSeed: 93,
    coverRoot: true,
    candidateCommands: () => candidates,
    rootTieBreak: 'candidate-order',
    leafValue: (o) => (o.units[0].y < unit.y ? 0.00002 : 0),
  });
  assert.equal(result.status, 'command');
  if (result.status === 'command') assert.deepEqual(result.command, candidates[0]);
});

test('异步零叶值不重新抽样，网络遇未知阶段时保留教师终局战术', async () => {
  for (const p of positions().filter((p) => p.family === 'immediate-win')) {
    const options = { simulations: 16, horizon: 2, sampleSeed: 391 };
    assert.deepEqual(
      await searchAsync(p.observation, { ...options, leafValue: async () => 0 }),
      search(p.observation, options),
    );
    const { valueStats, ...result } = await bootstrapValueDecision(
      p.observation,
      391,
      async () => 0,
    );
    assert.deepEqual(result.command, bootstrapDecision(p.observation, 391).command);
    if (result.mode === 'teacher-fallback') assert.equal(result.policy, null);
    assert.ok(valueStats.calls <= 16);
  }
});

test('叶值等待期间取消，回包之后仍暂停且不返回可落子命令', async () => {
  const controller = new AbortController();
  const result = await searchAsync(positions()[0].observation, {
    simulations: 16,
    horizon: 2,
    sampleSeed: 391,
    signal: controller.signal,
    leafValue: async () => {
      controller.abort();
      return 0.2;
    },
  });
  assert.equal(result.status, 'paused');
  assert.equal(result.stats.simulations, 0);
});

test('纯终局分支同样拒绝已经取消的网络决策', async () => {
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    bootstrapValueDecision(positions()[0].observation, 93, async () => 0, controller.signal),
    /搜索已取消/,
  );
});

test('网络叶值只传公开根张量，并从实际操作者转换为根视角', async () => {
  const p = positions()[0];
  const value = valueEvaluator(async (input) => {
    assert.deepEqual(Object.keys(input).sort(), [
      'candidate_mask',
      'candidates',
      'entities',
      'entity_mask',
      'globals',
      'kinds',
      'sources',
      'targets',
    ]);
    return { value: 0.5, logits: [] };
  });
  assert.equal(await value(p.observation, p.actor), 0.125);
  assert.equal(await value(p.observation, p.actor === 1 ? 2 : 1), -0.125);
});

test('采样在启动worker前拒绝不完整评测参数和不匹配的门槛报告', () => {
  const root = mkdtempSync(join(tmpdir(), 'haojie-value-preflight-'));
  try {
    // 即使前置校验回归，也由文件占位阻止进入耗时的真实采样。
    writeFileSync(join(root, 'blocked'), '');
    const output = join(root, 'blocked', 'games');
    const base = [
      '--import',
      'tsx',
      'scripts/training/search/bootstrap/sample.ts',
      '--output',
      output,
      '--seed',
      '2026093001',
      '--games',
      '4',
      '--evaluation-seeds',
      '1',
    ];
    const missing = spawnSync(process.execPath, base, { encoding: 'utf8', windowsHide: true });
    assert.notEqual(missing.status, 0);
    assert.match(missing.stderr, /换边评测须提供网络检查点/);
    assert.equal(existsSync(output), false);
    writeFileSync(join(root, 'summary.json'), JSON.stringify({ protocolSha256: 'wrong' }));
    writeFileSync(join(root, 'protocol.json'), '{}');
    const mismatched = spawnSync(
      process.execPath,
      [...base, '--checkpoint', join(root, 'unused.pt'), '--gate', root],
      { encoding: 'utf8', windowsHide: true },
    );
    assert.notEqual(mismatched.status, 0);
    assert.match(mismatched.stderr, /门槛报告与协议不匹配/);
    assert.equal(existsSync(output), false);
    const staleProtocol = JSON.stringify({
      sourceSha256: encodingSourceHash(),
      scriptHashes: { 'training/haojie_training/model.py': 'stale' },
    });
    writeFileSync(join(root, 'protocol.json'), staleProtocol);
    writeFileSync(
      join(root, 'summary.json'),
      JSON.stringify({ protocolSha256: createHash('sha256').update(staleProtocol).digest('hex') }),
    );
    const stale = spawnSync(
      process.execPath,
      [...base, '--checkpoint', join(root, 'unused.pt'), '--gate', root],
      { encoding: 'utf8', windowsHide: true },
    );
    assert.notEqual(stale.status, 0);
    assert.match(stale.stderr, /门槛搜索源码已变化/);
    assert.equal(existsSync(output), false);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
