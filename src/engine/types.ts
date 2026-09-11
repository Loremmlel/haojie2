import type { EventFacts } from './event-facts';
/** All game state is data. No renderer, network or wall clock is required. */
export type Player = 1 | 2;
export type Kind = number | '3p' | 'grave' | 'wall' | `u${number}` | 'u12p' | 'firelord';
export type Mode = 'none' | 'move' | 'attack' | 'skill' | 'charge';
export interface Point {
  x: number;
  y: number;
}
export interface Definition {
  id: Kind;
  name: string;
  glyph: string;
  role: string;
  tier: 'normal' | 'ultimate' | 'derived';
  attack: number;
  health: number;
  range: number;
  actions: number;
  move: number;
  description: string;
  skill?: string;
  spell?: number;
  weapon?: number;
  size?: number;
  mage?: boolean;
}
export interface Effect {
  type:
    | 'attack'
    | 'immune'
    | 'execute'
    | 'convert'
    | 'mark'
    | 'freeze'
    | 'burn'
    | 'stun'
    | 'inner-fire';
  from: number;
  until: number;
  owner: Player;
  amount?: number;
  sourceId?: string;
  global?: boolean;
}
export interface Unit extends Point {
  id: string;
  kind: Kind;
  owner: Player;
  hp: number;
  maxHp: number;
  size: number;
  born: number;
  offset: number;
  deployedAt: number;
  chargedOnDeploy: boolean;
  mode: Mode;
  operations: number;
  shots: number;
  moves: number;
  attacked: string[];
  bonusAttacks: number;
  bonusSequence: boolean;
  weaponFirstUsed: boolean;
  charge: number;
  readyCharge: number;
  chargeType: 'move' | 'attack' | 'skill';
  lastCharge: number;
  upgrades: number;
  kills: number;
  attackBonus: number;
  rangeBonus: number;
  guardUsed: boolean;
  effects: Effect[];
  equipment: Kind[];
  silenced: boolean;
  freeUsed: number;
  onceUsed: boolean;
  group?: string;
  expiresAt?: number;
  hookReadyAt?: number;
  hookExpiresAt?: number;
}
export interface Card {
  id: string;
  kind: Kind;
  drawnAt: number;
  expiresAt?: number;
  group?: string;
  rerolled?: boolean;
  summonedPly: number;
}
export interface Reaction {
  kind: 'death-shot' | 'reflect' | 'bounce' | 'hut-spawn';
  owner: Player;
  source: Unit;
  amount: number;
}
export interface GameEvent extends EventFacts {
  id: string;
  type:
    | 'spawn'
    | 'move'
    | 'attack'
    | 'damage'
    | 'heal'
    | 'death'
    | 'skill'
    | 'shield'
    | 'turn'
    | 'summon';
  path?: Point[];
  from?: Point;
  to?: Point;
  unitId?: string;
  owner?: Player;
  amount?: number;
  text?: string;
  ultimate?: boolean;
}
export interface DeathRecord {
  id: string;
  kind: Kind;
  owner: Player;
  ply: number;
  revived: boolean;
  group?: string;
}
export interface Hazard {
  id: string;
  owner: Player;
  sourceId?: string;
  axis: 'row' | 'column';
  line: number;
  due: number;
}
export interface Siphon {
  id: string;
  sourceId: string;
  owner: Player;
  fromId: string;
  toId: string;
}
export interface IceMark extends Point {
  id: string;
  sourceId: string;
  owner: Player;
  due: number;
}
export interface GameState {
  version: 2;
  seed: number;
  rng: number;
  serial: number;
  ply: number;
  active: Player;
  phase: 'summon' | 'play';
  summonSlots: number;
  turns: Record<Player, number>;
  bases: Record<Player, number>;
  baseEffects: Record<Player, Effect[]>;
  heads: Record<Player, number>;
  hands: Record<Player, Card[]>;
  bonus: Record<Player, number>;
  deployRows: Record<Player, number[]>;
  units: Unit[];
  pending: Reaction[];
  deaths: DeathRecord[];
  hazards: Hazard[];
  siphons: Siphon[];
  iceMarks: IceMark[];
  log: string[];
  events: GameEvent[];
  winner?: Player | 'draw';
}
/** Typed command payload shared by UI, saved replays and future server adapters. */
export interface Command {
  type:
    | 'end'
    | 'summon'
    | 'begin'
    | 'reroll'
    | 'deploy'
    | 'move'
    | 'attack'
    | 'charge'
    | 'skill'
    | 'cast'
    | 'equip'
    | 'craft'
    | 'react'
    | 'finish-mode';
  unitId?: string;
  cardId?: string;
  targetId?: string;
  secondId?: string;
  deathId?: string;
  x?: number;
  y?: number;
  column?: number;
  row?: number;
  mode?: string;
  ultimate?: boolean;
  charge?: boolean;
  cardIds?: string[];
  sacrificeIds?: string[];
}
export interface Stats {
  attack: number;
  range: number;
  actions: number;
  remaining: number;
  move: number;
  sleeping: boolean;
  operationLimit: number;
  operationsLeft: number;
  frozen: boolean;
  stunned: boolean;
  mode: Mode;
}
export interface Target extends Point {
  id: string;
  owner: Player;
  unit?: Unit;
}
export interface Source {
  owner?: Player;
  unit?: Unit;
  kind: 'attack' | 'spell' | 'skill' | 'status' | 'collision' | 'reflect' | 'sacrifice' | 'expire';
  path?: Point[];
  retaliated?: boolean;
}
