import { readFile, writeFile } from 'node:fs/promises';
import { buildGame } from './build.mjs';

// Pages 从 main 根目录发布；这里只生成文件，提交并推送后才进入发布流程。
const check = process.argv.includes('--check');
const html = await buildGame();
if (check) {
  const published = await readFile('index.html', 'utf8').catch(() => '');
  if (published !== html)
    throw new Error(
      'Pages index.html is missing or stale. Run npm run deploy and commit index.html with the source.',
    );
  await readFile('.nojekyll');
  console.log('Pages entry is byte-identical to the freshly built offline HTML.');
} else {
  await writeFile('index.html', html);
  await writeFile('.nojekyll', '');
  console.log(
    'Prepared main/(root): index.html + .nojekyll. Commit and push them with the source to publish.',
  );
}
