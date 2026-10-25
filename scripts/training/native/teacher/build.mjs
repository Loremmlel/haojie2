import { build } from 'esbuild';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir } from 'node:fs/promises';

const folder = 'training/evaluation/assets';
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
  format: 'haojie-native-evaluation-teacher-v1',
  sha256: createHash('sha256').update(data).digest('hex'),
  sources,
};
await mkdir(folder, { recursive: true });
if (process.argv.includes('--check')) {
  if (
    !Buffer.from(data).equals(await readFile(`${folder}/teacher.js`)) ||
    JSON.stringify(JSON.parse(await readFile(`${folder}/teacher.json`, 'utf8'))) !==
      JSON.stringify(manifest)
  )
    throw new Error('测评 AI 包或源码指纹过期');
} else {
  await writeFile(`${folder}/teacher.js`, data);
  await writeFile(`${folder}/teacher.json`, JSON.stringify(manifest, null, 2) + '\n');
}
console.log(`测评 AI 静态包 ${data.length} bytes`);
