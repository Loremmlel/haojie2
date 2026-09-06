export type Player = 1 | 2;
export type Kind = number | '3p' | 'grave' | 'wall';
export interface Point {
  x: number;
  y: number;
}
export interface Definition {
  id: Kind;
  name: string;
  glyph: string;
  role: string;
  attack: number;
  health: number;
  range: number;
  actions: number;
  move: number;
  description: string;
  skill?: string;
  spell?: number;
  size?: number;
}
export interface Effect {
  type: 'attack' | 'immune' | 'execute' | 'convert' | 'mark';
  from: number;
  until: number;
  owner: Player;
  amount?: number;
}
export interface Unit extends Point {
  id: string;
  kind: Kind;
  owner: Player;
  hp: number;
  maxHp: number;
  born: number;
  spent: number;
  attacked: string[];
  charge: number;
  lastCharge: number;
  fired: boolean;
  upgrades: number;
  kills: number;
  attackBonus: number;
  rangeBonus: number;
  guardUsed: boolean;
  effects: Effect[];
  expiresAt?: number;
}
export interface Card {
  id: string;
  kind: Kind;
  drawnAt: number;
  expiresAt?: number;
}
export interface Reaction {
  kind: 'death-shot' | 'reflect';
  owner: Player;
  source: Unit;
  amount: number;
}
export interface GameEvent {
  id: string;
  type: 'spawn' | 'move' | 'attack' | 'damage' | 'heal' | 'death' | 'skill' | 'shield' | 'turn';
  path?: Point[];
  from?: Point;
  to?: Point;
  unitId?: string;
  owner?: Player;
  amount?: number;
  text?: string;
}
export interface GameState {
  version: 1;
  seed: number;
  rng: number;
  serial: number;
  ply: number;
  active: Player;
  turns: Record<Player, number>;
  bases: Record<Player, number>;
  baseEffects: Record<Player, Effect[]>;
  hands: Record<Player, Card[]>;
  bonus: Record<Player, number>;
  deployRows: Record<Player, number[]>;
  units: Unit[];
  pending: Reaction[];
  log: string[];
  events: GameEvent[];
  winner?: Player | 'draw';
}
export type Command =
  | { type: 'end' }
  | { type: 'deploy'; cardId: string; x: number; y: number; charge?: boolean }
  | { type: 'move'; unitId: string; x: number; y: number }
  | { type: 'attack'; unitId: string; targetId: string }
  | {
      type: 'skill';
      unitId: string;
      mode?: 'attack' | 'range' | 'charge' | 'dash';
      targetId?: string;
      x?: number;
      y?: number;
      column?: number;
    }
  | {
      type: 'cast';
      cardId: string;
      targetId?: string;
      x?: number;
      y?: number;
      mode?: 'single' | 'double';
      sacrificeIds?: string[];
    }
  | { type: 'react'; targetId?: string };
export interface Stats {
  attack: number;
  range: number;
  actions: number;
  remaining: number;
  move: number;
  sleeping: boolean;
}
export interface Target extends Point {
  id: string;
  owner: Player;
  unit?: Unit;
}
