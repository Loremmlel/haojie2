import { build } from 'esbuild';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';

const folder = 'training/haojie_training/console/assets';
const result = await build({
  entryPoints: ['scripts/training/native/teacher/index.ts'],
  bundle: true,
  write: false,
  format: 'iife',
  globalName: 'HaojieTeacher',
  platform: 'browser',
  target: 'es2022',
  minify: true,
  metafile: true,
});
const data = result.outputFiles[0].contents;
const sources = {};
for (const name of Object.keys(result.metafile.inputs).sort())
  sources[name] = createHash('sha256')
    .update((await readFile(name, 'utf8')).replace(/\r\n/g, '\n'))
    .digest('hex');
const manifest = {
  format: 'haojie-teacher-bundle-v1',
  commit: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
  sha256: createHash('sha256').update(data).digest('hex'),
  sources,
};
await mkdir(folder, { recursive: true });
if (process.argv.includes('--check')) {
  if (!Buffer.from(data).equals(await readFile(`${folder}/teacher.js`))) throw Error('教师包过期');
  const saved = JSON.parse(await readFile(`${folder}/teacher.json`, 'utf8'));
  if (saved.sha256 !== manifest.sha256 || JSON.stringify(saved.sources) !== JSON.stringify(sources))
    throw Error('教师来源指纹过期');
} else {
  await writeFile(`${folder}/teacher.js`, data);
  await writeFile(`${folder}/teacher.json`, JSON.stringify(manifest, null, 2) + '\n');
  // 单独发行的控制台消费既有游戏色板，避免复制两套主题值。
  await writeFile(`${folder}/tokens.css`, await readFile('src/ui/styles/tokens.css'));
}
console.log(`教师静态包 ${data.length} bytes；运行阶段不需要 Node`);
