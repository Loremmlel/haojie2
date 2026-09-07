import type { GameState } from '../../engine';
export type GameModal = 'rules' | 'codex' | 'new' | 'log' | null;
export interface HaojieGameProps {
  initialState?: GameState;
  storageKey?: string | null;
  onStateChange?: (state: GameState) => void;
}
