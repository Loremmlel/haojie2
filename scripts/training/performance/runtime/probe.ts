import { build } from 'esbuild';
import ts from 'typescript';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve, join, relative } from 'node:path';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';
import assert from 'node:assert/strict';

const { values } = parseArgs({
  options: { output: { type: 'string' }, ref: { type: 'string' } },
});
assert.ok(values.output);
const output = resolve(values.output);
mkdirSync(output);
const ref = values.ref
  ? execFileSync('git', ['rev-parse', '--verify', `${values.ref}^{commit}`], {
      encoding: 'utf8',
    }).trim()
  : undefined;
writeFileSync(join(output, 'probe.ts'), readFileSync(new URL(import.meta.url)), { flag: 'wx' });
const names = new Set([
  'prepareMove',
  'moveDestinations',
  'prepareAttack',
  'prepareInspection',
  'skillInspection',
  'prepareSkillInspection',
  'movement',
  'attackPath',
  'attackRoutes',
  'getStats',
  'sourceObjects',
  'forkPosition',
  'cloneRuleData',
  'trainingPosition',
  'observe',
  'transition',
  'inspect',
  'actorCommandError',
  'parseCommand',
  'encodeBasePosition',
  'positionRows',
  'layer',
  'logits',
  'evaluate',
  '#expand',
  'node',
  'placement',
  'targets',
  'chooseMode',
]);
// 探针只在冻结 bundle 注入；生产源码无计时分支。含探针结果不能用作加速倍率。
const built = await build({
  entryPoints: ['scripts/training/performance/api.ts'],
  bundle: true,
  platform: 'node',
  format: 'esm',
  target: 'node22',
  write: false,
  banner: {
    js: `globalThis.__runtimeProbe = { enabled: false, rows: {}, stack: [], count(name, n=1) { if(this.enabled) { const r=this.rows[name]??={calls:0}; r.calls+=n; } }, enter(name, args) { if (!this.enabled) return null; const r = this.rows[name] ??= {calls:0, inclusiveMs:0, selfMs:0, units:0, entities:0, width:0, shapes:{}}; r.calls++; r.units += args[0]?.units?.length ?? 0; r.entities += args[0]?.entities?.length ?? args[1]?.entities?.length ?? 0; r.width += args[0]?.length ?? 0; if(args[0]?.length !== undefined) r.shapes[args[0].length]=(r.shapes[args[0].length]??0)+1; const f = {r, start:performance.now(), children:0}; this.stack.push(f); return f; }, leave(f) { if (!f) return; const ms = performance.now()-f.start; f.r.inclusiveMs += ms; f.r.selfMs += ms-f.children; this.stack.pop(); if(this.stack.length) this.stack.at(-1).children += ms; } };`,
  },
  plugins: [
    {
      name: 'bounded-runtime-probe',
      setup(b) {
        b.onLoad({ filter: /\.ts$/ }, ({ path }) => {
          const key = relative(process.cwd(), path).replaceAll('\\', '/');
          if (key === 'scripts/training/performance/api.ts') return;
          if (!ref && !path.includes('src') && !path.includes('economics')) return;
          const source = ref
            ? execFileSync('git', ['show', `${ref}:${key}`], {
                encoding: 'utf8',
              })
            : readFileSync(path, 'utf8');
          const ast = ts.createSourceFile(path, source, ts.ScriptTarget.Latest, true);
          const edits: { at: number; text: string }[] = [];
          const walk = (node: ts.Node) => {
            if (
              (ts.isFunctionDeclaration(node) || ts.isMethodDeclaration(node)) &&
              node.body &&
              node.name &&
              names.has(node.name.getText(ast))
            ) {
              const name =
                path.replaceAll('\\', '/').split('/').slice(-3).join('/') +
                ':' +
                node.name.getText(ast);
              edits.push({
                at: node.body.getStart(ast) + 1,
                text:
                  `const __probeFrame = (globalThis as any).__runtimeProbe.enter(${JSON.stringify(name)}, arguments); try {` +
                  (node.name.getText(ast) === 'positionRows'
                    ? `if (arguments[1]) { const p=(globalThis as any).__runtimeProbe, base=arguments[1], borrowed=arguments[2]===true; p.count('encoding.fixedRowsCopied',borrowed?0:base.entities.length); p.count('encoding.fixedScalarsCopied',borrowed?0:base.entities.length*64); p.count('encoding.identityEntriesCopied',borrowed?0:base.identities.size); p.count('encoding.indexEntriesCopied',borrowed?0:base.indices.size); }`
                    : ''),
              });
              edits.push({
                at: node.body.getEnd() - 1,
                text: '} finally { (globalThis as any).__runtimeProbe.leave(__probeFrame); }',
              });
            }
            ts.forEachChild(node, walk);
          };
          walk(ast);
          let contents = source;
          for (const e of edits.sort((a, b) => b.at - a.at))
            contents = contents.slice(0, e.at) + e.text + contents.slice(e.at);
          if (path.endsWith('clone.ts'))
            contents = contents.replace(
              'const result = array ? value.slice() : { ...value };',
              "(globalThis as any).__runtimeProbe.count('clone.objects'); const result = array ? value.slice() : { ...value };",
            );
          if (path.endsWith('branch.ts')) {
            contents = contents.replace(
              'const target = array ? [] : {};',
              "(globalThis as any).__runtimeProbe.count('branch.proxyObjects'); const target = array ? [] : {};",
            );
            contents = contents.replace(
              'const result: any = Array.isArray(source)',
              "(globalThis as any).__runtimeProbe.count('branch.detachedObjects'); const result: any = Array.isArray(source)",
            );
          }
          if (path.endsWith('game.ts'))
            contents = contents.replace(
              's = queries ?',
              "(globalThis as any).__runtimeProbe.count('settlement.' + c.type); s = queries ?",
            );
          if (path.endsWith('geometry.ts')) {
            contents = contents.replaceAll(
              'const next = [...path, p];',
              "(globalThis as any).__runtimeProbe.count('spatial.expandedPathArrays'); (globalThis as any).__runtimeProbe.count('spatial.copiedPathPoints', path.length + 1); const next = [...path, p];",
            );
            contents = contents.replace(
              'return result.reverse();',
              "(globalThis as any).__runtimeProbe.count('spatial.reconstructedPaths'); (globalThis as any).__runtimeProbe.count('spatial.reconstructedPoints', result.length); return result.reverse();",
            );
          }
          return { contents, loader: 'ts' };
        });
      },
    },
  ],
});
writeFileSync(join(output, 'api.mjs'), built.outputFiles[0].contents, {
  flag: 'wx',
});
writeFileSync(
  join(output, 'manifest.json'),
  JSON.stringify(
    {
      argv: process.argv,
      ref,
      node: process.version,
      bundleSha256: createHash('sha256').update(built.outputFiles[0].contents).digest('hex'),
    },
    null,
    2,
  ),
  { flag: 'wx' },
);
