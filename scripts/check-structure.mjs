import { readdir } from 'node:fs/promises';
import { join } from 'node:path';

// 只统计项目维护文件；依赖、版本库及测试/构建产物不属于人工组织范围。
const excluded = new Set([
  '.git',
  'node_modules',
  'dist',
  'artifacts',
  '.venv',
  '__pycache__',
  '.ruff_cache',
]);
async function inspect(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = entries.filter((entry) => entry.isFile());
  if (files.length > 10)
    throw new Error(`${directory} 有 ${files.length} 个文件，请按职责分组（上限10）。`);
  for (const entry of entries) {
    if (entry.isDirectory() && !excluded.has(entry.name) && !entry.name.endsWith('.egg-info'))
      await inspect(join(directory, entry.name));
  }
}
await inspect('.');
console.log('目录组织检查通过：每个维护目录最多10个文件。');
