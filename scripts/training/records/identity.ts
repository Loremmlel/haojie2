import { createHash } from 'node:crypto';
import { readFileSync, readdirSync } from 'node:fs';
import { relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

/** 审计记录实际规则、公开编码和重放源码，不覆盖历史记录中的指纹。 */
export function encodingSourceHash() {
  const root = fileURLToPath(new URL('../../../', import.meta.url));
  const hash = createHash('sha256');
  for (const folder of ['src/engine', 'src/ai', 'src/match', 'scripts/training/records']) {
    const path = resolve(root, folder);
    for (const name of readdirSync(path, { recursive: true })
      .map(String)
      .filter((p) => p.endsWith('.ts'))
      .sort()) {
      const file = resolve(path, name);
      hash.update(relative(root, file).replaceAll('\\', '/'));
      hash.update(readFileSync(file, 'utf8').replaceAll('\r\n', '\n'));
    }
  }
  return hash.digest('hex');
}
