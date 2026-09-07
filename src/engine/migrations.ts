import type { Session } from './history';
import type { Player } from './types';

/** 2.0.1 author correction: old U17 cards lacked an expiry. Keep schema v2 and every snapshot. */
export function normalizeHornStorage(input: Session): Session {
  const session = structuredClone(input);
  for (const state of [
    session.present,
    ...session.past,
    ...session.future,
    ...(session.humanAnchor ? [session.humanAnchor] : []),
  ]) {
    for (const player of [1, 2] as Player[]) {
      state.hands[player] = state.hands[player].filter((card) => {
        if (card.kind !== 'u17') return true;
        card.expiresAt ??= card.drawnAt + 8;
        return card.expiresAt > state.turns[player];
      });
    }
  }
  return session;
}
