import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, relative } from 'node:path';
import { build } from 'esbuild';

// 比较完整基线源码与当前工作树，不混用新 AI 和旧引擎。
// 正式 RNG 只用于宿主重放断言，决策仅接收 observe 的白名单。
// 固定 work 预算；任何差异或输入修改立即抛错，报告不写入计时基准。
const ref = process.argv[2] ?? 'cec107917b840f9cb34954fa1635db58ccd2ae66';
const baseline = execFileSync('git', ['rev-parse', '--verify', `${ref}^{commit}`], {
  encoding: 'utf8',
}).trim();
const root = process.cwd();
async function load(old) {
  const result = await build({
    stdin: {
      contents: `export * as engine from './src/engine/index';
        export * as ai from './src/ai/index';
        export * as geometry from './src/engine/core/geometry';
        export * as candidates from './src/ai/planning/candidates';
        export * as helpers from './tests/helpers';`,
      resolveDir: root,
      loader: 'ts',
    },
    bundle: true,
    platform: 'node',
    format: 'esm',
    write: false,
    plugins: old
      ? [
          {
            name: '完整冻结基线',
            setup(api) {
              api.onLoad({ filter: /\.(ts|tsx)$/ }, ({ path }) => ({
                contents: execFileSync(
                  'git',
                  ['show', `${baseline}:${relative(root, path).replaceAll('\\', '/')}`],
                  { encoding: 'utf8' },
                ),
                loader: path.endsWith('.tsx') ? 'tsx' : 'ts',
                resolveDir: dirname(path),
              }));
            },
          },
        ]
      : [],
  });
  return import(
    `data:text/javascript;base64,${Buffer.from(result.outputFiles[0].contents).toString('base64')}`
  );
}
const before = await load(true),
  after = await load(false);
const records = readFileSync('docs/playtests/current/cli-feedback5-20260923.jsonl', 'utf8')
  .trim()
  .split('\n')
  .map(JSON.parse);
let a = before.engine.parseSession(JSON.stringify(records[0].initial)).present;
let b = after.engine.parseSession(JSON.stringify(records[0].initial)).present;
const cases = [
  { name: 'demo', state: before.engine.createDemoGame() },
  { name: 'shrine-draft', state: a },
];
let commands = 0;
for (const row of records.slice(1)) {
  if (!row.command) continue;
  a = before.engine.applyCommand(a, row.command);
  b = after.engine.applyCommand(b, row.command);
  assert.deepEqual(b, a, `完整权威状态在第 ${commands + 1} 条命令发生变化`);
  if (++commands % 15 === 0 && !a.winner) cases.push({ name: `replay-${commands}`, state: a });
}
function fixture(name, setup) {
  const state = before.helpers.fixture();
  setup(state);
  cases.push({ name, state });
}
const { add, card } = before.helpers;
fixture('clones-neutral-loner', (s) => {
  add(s, 'u25', 1, 3, 5);
  add(s, 'u25', 1, 3, 5);
  add(s, 23, 1, 5, 6);
  add(s, 'grave', 1, 7, 6);
  add(s, 23, 2, 5, 9).silenced = true;
  add(s, 9, 1, 3, 3);
  add(s, 24, 2, 3, 7);
  card(s, 1);
  card(s, 'u25');
});
fixture('giants-landmarks', (s) => {
  s.mode = 'shrine';
  s.landmarks = [];
  const live = add(s, 's1', 1, 4, 7);
  s.units.pop();
  s.landmarks.push(live);
  add(s, 2, 1, 4, 7);
  const asleep = add(s, 's1', 2, 7, 8);
  s.units.pop();
  asleep.hp = 0;
  asleep.dormantSince = s.ply;
  s.landmarks.push(asleep);
  add(s, 'u7', 1, 1, 5);
  add(s, 'u12p', 2, 7, 10);
  add(s, 'u20', 1, 7, 3);
  add(s, 24, 2, 6, 6);
  card(s, 'u27');
  card(s, 1);
});
fixture('synthesis', (s) => {
  s.phase = 'synthesis';
  for (const x of [2, 4, 6]) add(s, 'u12p', 1, x, 4);
  add(s, 24, 2, 5, 9);
});
fixture('live-deployment', (s) => {
  add(s, 1, 1, 3, 9);
  add(s, 1, 1, 5, 8);
  add(s, 15, 2, 8, 10);
  card(s, 1);
  card(s, 7);
});
fixture('inherited-equipment', (s) => {
  const u = add(s, 's5', 1, 4, 5);
  u.traits = [23, 15, 'u20'];
  u.abilityCharges = { 15: { charge: 2, readyCharge: 2, chargeType: 'attack', lastCharge: 3 } };
  u.equipment = ['u28'];
  add(s, 'sage', 1, 3, 3);
  add(s, 24, 2, 5, 7);
  add(s, '17p', 2, 6, 7);
  add(s, 2, 1, 4, 3).hp = 10;
  card(s, 18);
  card(s, 22);
});
fixture('dense-48', (s) => {
  let i = 0;
  for (let y = 3; y <= 10; y++)
    for (let x = 2; x <= 7; x++) {
      const u = add(s, [1, 2, 9, 15, 24, 26][i % 6], y <= 6 ? 1 : 2, x, y);
      if (i++ % 7 === 0) u.effects.push({ type: 'freeze', owner: u.owner, until: 7 });
    }
  card(s, 1);
});
let placements = 0,
  paths = 0,
  groups = 0,
  attacks = 0,
  decisions = 0;
const digest = createHash('sha256');
for (const { name, state } of cases) {
  const original = structuredClone(state),
    query = after.geometry.createPlacementQuery(state);
  const units = [
    ...state.units.slice(0, 8),
    ...['u25', 23, 'u7', 'u27', 's1'].map((kind, i) =>
      before.engine.template(kind, i % 2 ? 2 : 1, 0, { x: 1, y: 1 }, `ghost-${i}`),
    ),
  ];
  for (const u of units)
    for (const p of before.geometry.ALL_CELLS)
      for (const deployment of [false, true]) {
        const ignore = (p.x + p.y) % 3 === 0 ? state.units.slice(0, 1).map((v) => v.id) : [];
        assert.equal(
          query.canPlace(u, p, deployment, ignore),
          before.geometry.canPlace(state, u, p, deployment, ignore),
          `${name}:落点`,
        );
        placements++;
      }
  for (const u of state.units.slice(0, 3))
    for (const p of before.geometry.ALL_CELLS.filter((_, i) => i % 11 === 0))
      for (const straight of [false, true]) {
        assert.deepEqual(
          query.movementPath(u, p, 3, straight),
          before.geometry.movementPath(state, u, p, 3, straight),
          `${name}:路径`,
        );
        paths++;
      }
  for (const difficulty of ['easy', 'medium', 'hard']) {
    const expected = before.candidates.candidateGroups(state, difficulty);
    assert.deepEqual(after.candidates.candidateGroups(state, difficulty), expected, `${name}:候选`);
    assert.deepEqual(
      [...after.candidates.iterateCandidateGroups(state, difficulty, true)].map((g) => ({ ...g })),
      expected,
      `${name}:延迟候选`,
    );
    groups += 2;
  }
  for (const rows of [[], [4], [9], [6, 7, 8]]) {
    assert.deepEqual(
      after.candidates.deploymentCandidates(state, rows),
      before.candidates.deploymentCandidates(state, rows),
      `${name}:部署候选`,
    );
    groups++;
  }
  for (const u of state.units.filter((u) => u.owner === state.active).slice(0, 5)) {
    const expected = before.candidates.attackCandidates(state, u.id);
    for (const t of [
      ...before.geometry.targets(state).slice(0, 7),
      { id: 'missing' },
      { id: 'base-2' },
    ]) {
      assert.deepEqual(
        after.candidates.attackCandidates(state, u.id, t.id),
        expected.filter((c) => c.targetId === t.id),
        `${name}:定向攻击`,
      );
      attacks++;
    }
  }
  const observation = before.ai.observe(state),
    saved = structuredClone(observation),
    side = before.ai.decisionOwner(observation);
  for (const [difficulty, simulations] of [
    ['easy', 40],
    ['medium', 320],
    ['hard', 800],
  ]) {
    const limits = { mode: 'work', simulations, trace: true };
    const expected = before.ai.decide(observation, side, difficulty, limits);
    assert.deepEqual(
      after.ai.decide(observation, side, difficulty, limits),
      expected,
      `${name}:${difficulty}:完整决策`,
    );
    assert.deepEqual(observation, saved, `${name}:观察被修改`);
    digest.update(JSON.stringify(expected) + '\n');
    decisions++;
  }
  assert.deepEqual(state, original, `${name}:输入被修改`);
  console.log(`一致：${name}`);
}
const report = {
  baseline,
  cases: cases.map(({ name, state }) => ({ name, units: state.units.length, phase: state.phase })),
  commands,
  placements,
  paths,
  groups,
  attacks,
  decisions,
  decisionSha256: digest.digest('hex'),
  fullStateEqual: true,
  fixedWorkDecisionsEqual: true,
};
mkdirSync('artifacts', { recursive: true });
writeFileSync('artifacts/ai-reuse-equivalence-report.json', JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
