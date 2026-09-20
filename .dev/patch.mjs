import { readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
const read = (p) => readFileSync(p, 'utf8');
const write = (p, s) => writeFileSync(p, s);
function replace(p, from, to) {
  const source = read(p);
  if (!source.includes(from)) throw new Error(`Missing anchor: ${p}: ${from}`);
  write(p, source.replace(from, to));
}
replace('src/engine/state.ts', '  s.rng = x >>> 0;\n  return s.rng / 4294967296;', '  const next = x >>> 0;\n  s.rng = next;\n  return next / 4294967296;');
write('src/engine/authority.ts', read('src/engine/authority.ts') + `
/** Only the independent, once-per-game secret choices can safely rebase after the other seat
 * committed. The host remains ignorant of individual game modes; all ordinary stale actions fail. */
export function canRebasePlayerCommand(s: GamePosition, actor: Player, c: Command): boolean {
  return s.phase === 'shrine-draft' && c.type === 'choose-shrine' &&
    actorCommandError(s, actor, c) === null;
}
`);
write('src/engine/index.ts', read('src/engine/index.ts') + "\nexport { canRebasePlayerCommand } from './authority';\n");
replace('src/ui/online/types.ts', "if (!update.view.state || 'seed' in update.view.state || 'rng' in update.view.state)", "if (!update.view.state || typeof update.view.state !== 'object' || 'seed' in update.view.state || 'rng' in update.view.state)");
replace('src/ui/online/useOnlineController.ts', "if (update.kind === 'snapshot') presentation.clear();", "if (update.kind === 'snapshot' || config.current.connection !== 'connected') presentation.clear();");
replace('src/ui/online/useOnlineController.ts', '  function connectionBlock(): string | null {', `  // An equal-revision reconnect snapshot clears effects, never newer state or a pending ack.
  useEffect(() => {
    if (props.update.kind === 'snapshot' && props.update.revision >= current.current.revision)
      presentation.clear();
  }, [props.update]);

  function connectionBlock(): string | null {`);
const pkg = JSON.parse(read('package.json'));
pkg.scripts['test:browser:online'] = 'node tests/browser/online.mjs';
write('package.json', JSON.stringify(pkg, null, 2) + '\n');
const tsconfig = JSON.parse(read('tsconfig.json'));
tsconfig.include = ['src', 'tests/**/*.ts', 'tests/**/*.tsx', 'scripts/**/*.ts', 'examples/**/*.ts', 'examples/**/*.tsx'];
write('tsconfig.json', JSON.stringify(tsconfig, null, 2) + '\n');
execFileSync('git', ['add', 'tsconfig.json']);
replace('tests/session/online.test.ts', "s = applyPlayerCommand(s, 2, { type: 'choose-shrine', shrineKind: s.shrineDraft!.offers[2][0] });", "s = applyPlayerCommand(s, 2, { type: 'choose-shrine', shrineKind: s.shrineDraft!.offers[2][0], parity: 'odd' });");
