import type { Command, GameState, Player } from '../engine/types';
export type Difficulty = 'easy' | 'medium' | 'hard';
/** No real seed, PRNG state, event history or logs cross the planning boundary. */
export type Observation = Omit<GameState, 'seed' | 'rng' | 'events' | 'log'>;
export interface SearchLimits {
  simulations: number;
  milliseconds: number;
}
export interface PlanStep {
  before: string;
  command: Command;
}
export interface Decision {
  command: Command | null;
  plan: PlanStep[];
  stats: {
    simulations: number;
    candidates: number;
    depth: number;
    replies: number;
    sampled: number;
    exhausted: boolean;
  };
}
export interface SearchRequest {
  id: number;
  observation: Observation;
  side: Player;
  difficulty: Difficulty;
  limits?: Partial<SearchLimits>;
}
export interface SearchResponse {
  id: number;
  decision?: Decision;
  error?: string;
}
