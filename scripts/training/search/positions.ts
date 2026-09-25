import { createGame, applyCommand } from '../../../src/engine';
import { template } from '../../../src/engine/core/state';
import type { GameState, Kind, Player } from '../../../src/engine/types';
import { observe, decisionOwner } from '../../../src/ai/observation';
import type { Observation } from '../../../src/ai/types';

/** 人工短程规则夹具，不冒称自然对局分布或独立训练种子族。 */
export function positions() {
  const result: {
    name: string;
    family: string;
    actor: Player;
    x: number;
    observation: Observation;
  }[] = [];
  for (const actor of [1, 2] as const)
    for (const x of [4, 5, 6]) {
      const opponent = (3 - actor) as Player;
      const y = actor === 1 ? 11 : 3;
      const make = () => {
        const state = createGame(19);
        Object.assign(state, {
          active: actor,
          phase: 'play',
          summonSlots: 0,
          ply: 6,
          turns: { 1: 3, 2: 3 },
          heads: { 1: 0, 2: 0 },
          hands: { 1: [], 2: [] },
          units: [],
          events: [],
          log: [],
        });
        return state;
      };
      const add = (state: GameState, kind: Kind, owner: Player, xx = x, yy = y) => {
        const unit = template(kind, owner, 0, { x: xx, y: yy }, `probe${state.serial++}`);
        state.units.push(unit);
        return unit;
      };
      const emit = (name: string, state: GameState) =>
        result.push({
          name: `${name}-p${actor}-x${x}`,
          family: name,
          actor: decisionOwner(state),
          x,
          observation: observe(state),
        });
      const one = make();
      one.bases[opponent] = 20;
      add(one, 26, actor);
      emit('immediate-win', one);
      const two = make();
      two.bases[opponent] = 30;
      add(two, 26, actor);
      add(two, 26, actor, x + 1);
      emit('same-player-two-attacks', two);
      const chance = make();
      chance.bases[opponent] = 30;
      add(chance, 1, actor);
      emit('chance-win', chance);
      const danger = make();
      danger.bases[actor] = 20;
      const attacker = add(danger, 26, actor);
      const victim = add(danger, 2, opponent, x, y + (actor === 1 ? 1 : -1));
      victim.hp = 1;
      emit('opponent-death-reaction', danger);
      const reaction = applyCommand(danger, {
        type: 'attack',
        unitId: attacker.id,
        targetId: victim.id,
      });
      emit('reaction-to-play', reaction);
    }
  return result;
}
