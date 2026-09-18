/** Test-only fixtures use public engine commands; no hooks are shipped in the game. */
import { mkdirSync, writeFileSync } from 'node:fs';
import {
  applyCommand,
  createGame,
  createSession,
  validState,
  type GameState,
  type Player,
} from '../../src/engine';
import { addUnit } from '../../src/engine/state';
import { demolish, captureClockFrame } from '../../src/engine/shrines';
import { fixture, add, card } from '../helpers';
mkdirSync('artifacts/shrine-fixtures', { recursive: true });
function base() {
  let s = createGame(90, 'shrine');
  for (const p of [1, 2] as Player[])
    s = applyCommand(s, { type: 'choose-shrine', player: p, shrineKind: p === 1 ? 's8' : 's13' });
  s = applyCommand(applyCommand(s, { type: 'finish-shrine-setup' }), {
    type: 'finish-shrine-setup',
  });
  return Object.assign(s, fixture());
}
function save(name: string, s: GameState) {
  if (!validState(s)) throw new Error(`Invalid fixture ${name}`);
  writeFileSync(
    `artifacts/shrine-fixtures/${name}.json`,
    JSON.stringify(
      createSession(s, { mode: 'local', human: 1, difficulty: 'easy', rules: 'shrine' }),
    ),
  );
}
save('draft', createGame(90, 'shrine'));
let s = base();
card(s, 's8');
save('flag-card', s);
s = base();
addUnit(s, 's8', 1, { x: 7, y: 9 });
add(s, 1, 1, 7, 9);
save('flag-stack', s);
s = base();
demolish(s, addUnit(s, 's8', 1, { x: 7, y: 9 }) as import('../../src/engine').Landmark);
save('dormant', s);
s = base();
addUnit(s, 's1', 1, { x: 3, y: 6 });
card(s, 1);
save('charge', s);
s = base();
s.auras = { 1: [{ kind: 's13' }], 2: [] };
s.phase = 'summon';
s.summonSlots = 2;
s.regularSummons = 2;
save('offer', s);
s = base();
[2, 3, 4].forEach((x) => add(s, 'u13', 1, x, 3));
s.phase = 'synthesis';
s.summonSlots = 2;
s.regularSummons = 2;
save('synthesis', s);
s = base();
s.auras = { 1: [{ kind: 's10' }], 2: [] };
const clock = add(s, 'u6', 1, 3, 4);
captureClockFrame(s);
s.ply += 2;
s.turns[1]++;
captureClockFrame(s);
clock.hp = 5;
clock.onceUsed = true;
clock.operations = 1;
clock.x = 4;
save('clock', s);
s = base();
s.auras = { 1: [{ kind: 's9', parity: 'odd' }], 2: [] };
add(s, 9, 1, 3, 4).hp = 12;
add(s, 1, 2, 3, 5);
save('jade', s);
s = base();
add(s, 's4', 1, 3, 4);
add(s, 1, 1, 3, 5).hp = 10;
save('heal', s);
s = base();
card(s, 's2');
add(s, 1, 1, 3, 4);
save('weapon', s);
s = base();
addUnit(s, 's1', 2, { x: 3, y: 6 });
add(s, 1, 2, 3, 6);
add(s, 9, 1, 3, 4);
card(s, 8);
save('layer', s);
