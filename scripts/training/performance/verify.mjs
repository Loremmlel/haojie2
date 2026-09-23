import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { build } from 'esbuild';
import { gunzipSync } from 'node:zlib';

// 仅适用于本次 geometry 优化：其余源码共用工作树，以指定提交的原几何实现作对照。
// 完整状态包含正式 rng、事件和日志，留在宿主断言中，绝不送给教师。
const [ref, ...paths] = process.argv.slice(2);
assert.ok(ref && paths.length, '参数为基线提交及教师/CLI JSONL 路径');
const oldGeometry = execFileSync('git', ['show', `${ref}:src/engine/core/geometry.ts`], {
  encoding: 'utf8',
});
async function engine(old) {
  const result = await build({
    entryPoints: ['src/engine/index.ts'],
    bundle: true,
    platform: 'node',
    format: 'esm',
    write: false,
    plugins: old
      ? [
          {
            name: '原几何实现',
            setup(api) {
              api.onLoad({ filter: /[/\\]engine[/\\]core[/\\]geometry\.ts$/ }, (args) => ({
                contents: oldGeometry,
                loader: 'ts',
                resolveDir: dirname(args.path),
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
const before = await engine(true);
const after = await engine(false);
let placements = 0;
const demo = before.createDemoGame();
for (const unit of demo.units)
  for (const size of [1, 2])
    for (const at of before.ALL_CELLS)
      for (const deployment of [false, true]) {
        const u = { ...unit, size };
        assert.equal(
          after.canPlace(demo, u, at, deployment),
          before.canPlace(demo, u, at, deployment),
        );
        placements++;
      }
let commands = 0;
for (const path of paths) {
  let a, b;
  const bytes = readFileSync(path);
  const text = (path.endsWith('.gz') ? gunzipSync(bytes) : bytes).toString('utf8');
  for (const line of text.trim().split('\n')) {
    const row = JSON.parse(line);
    if (row.type === 'game') {
      assert.equal(
        row.format,
        'haojie-training-record-v1',
        '旧训练轨迹请重新生成；历史实验使用当时脚本',
      );
      a = row.initial
        ? before.parseSession(JSON.stringify(row.initial)).present
        : before.createGame(row.seed, row.rules);
      b = row.initial
        ? after.parseSession(JSON.stringify(row.initial)).present
        : after.createGame(row.seed, row.rules);
    } else if (row.format === 'haojie-cli-v1') {
      a = before.parseSession(JSON.stringify(row.initial)).present;
      b = after.parseSession(JSON.stringify(row.initial)).present;
    } else if (row.command && !['rejected', 'pause', 'error'].includes(row.type)) {
      a = before.applyCommand(a, row.command);
      b = after.applyCommand(b, row.command);
      assert.deepEqual(b, a, `${path} 第${commands + 1}条命令状态漂移`);
      commands++;
    }
  }
}
console.log(JSON.stringify({ baseline: ref, placements, commands, fullStateEqual: true }));
