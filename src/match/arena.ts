/** Headless command port. No readline/DOM/timers: the CLI and tests use the real match engine. */
import { createGame, createSession, dispatch, parseSession } from '../engine';
import type { Command, GameState, Session } from '../engine';
import { decisionOwner, fingerprint, observe } from '../ai/observation';
import { decide } from '../ai/search';
import type { Decision, PlanStep, SearchLimits } from '../ai/types';
import { allocateBudget, emptyBudget } from '../ai/budget';
import type { MatchSettings } from './settings';
import { matchSettings, ownsComputerDecision } from './history';

export interface ArenaEntry {
  actor: 'human' | 'ai';
  owner: 1 | 2;
  ply: number;
  command: Command;
  before: string;
  after: string;
  events: GameState['events'];
  decision?: Decision;
}
export class Arena {
  session: Session;
  private budget = emptyBudget();
  private cache: PlanStep[] = [];
  constructor(session: Session) {
    this.session = parseSession(JSON.stringify(session));
  }
  static create(seed: number, match: MatchSettings) {
    return new Arena({ ...createSession(createGame(seed)), match });
  }
  get computerTurn() {
    return ownsComputerDecision(this.session);
  }
  play(command: Command, actor: 'human' | 'ai' = 'human', decision?: Decision): ArenaEntry {
    if (actor === 'human' && this.computerTurn)
      throw new Error('当前是AI的决策；使用 go 或 step。');
    if (actor === 'ai' && !this.computerTurn) throw new Error('当前决策属于人类，AI不能代选。');
    const before = this.session.present;
    const next = dispatch(this.session, command);
    this.session = next;
    return {
      actor,
      owner: decisionOwner(before),
      ply: before.ply,
      command,
      before: fingerprint(before),
      after: fingerprint(next.present),
      events: next.present.events,
      ...(decision ? { decision } : {}),
    };
  }
  step(limits: Partial<SearchLimits> = {}): ArenaEntry | null {
    if (!this.computerTurn) return null;
    const s = this.session.present;
    if (this.budget.ply !== s.ply) this.budget = { ...emptyBudget(), ply: s.ply };
    if (this.budget.commands >= 200) throw new Error('本回合达到200条自动命令，已暂停。');
    const difficulty = matchSettings(this.session).difficulty;
    const start = performance.now();
    const predicted = this.cache[0];
    let decision: Decision;
    if (predicted?.before === fingerprint(s)) {
      decision = {
        command: predicted.command,
        plan: this.cache,
        stats: {
          simulations: 0,
          candidates: 0,
          depth: 0,
          replies: 0,
          sampled: 0,
          exhausted: false,
        },
      };
    } else
      decision = decide(observe(s), decisionOwner(s), difficulty, {
        ...allocateBudget(s, difficulty, this.budget),
        ...limits,
      });
    this.budget.nodes += decision.stats.simulations;
    this.budget.ms += performance.now() - start;
    this.budget.commands++;
    this.cache = decision.plan.slice(1);
    if (!decision.command) throw new Error(`AI无合法候选，停在ply=${s.ply}，未跳过局面。`);
    return this.play(decision.command, 'ai', decision);
  }
}
