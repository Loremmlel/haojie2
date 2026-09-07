import { readFile, writeFile } from 'node:fs/promises';
import { buildGame } from './build.mjs';

// Branch publishing serves main/(root). This prepares files; git commit/push publishes them.
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
