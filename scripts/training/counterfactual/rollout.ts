import assert from 'node:assert/strict';
import { decisionOwner, fingerprint, imagined, hash } from '../../../src/ai/observation';
import { evaluate } from '../../../src/ai/evaluation/evaluate';
import { TrainingTeacher } from '../../../src/ai/training/teacher';
import { sampleTrainingTransition } from '../../../src/ai/training/simulation';
import type { Observation } from '../../../src/ai/types';
import type { Command } from '../../../src/engine/types';
import { checkPosition, terminalValue } from '../search/puct';

export interface RolloutJob {
  observation: Observation;
  command: Command;
  sample: number;
  maxCommands: number;
  maxWork: number;
  terminal?: boolean;
  maxPlies?: number;
}

/**
 * 研究续弈只接收公开根；每个分支重新创建相同的 easy40 热教师，复用共享计划缓存。
 * 先执行指定候选，再完成己方及对手窗口，停在根方下一次召唤前且无反应的边界。
 * 每步使用独立模拟编号；同场景只耦合随机数，不假设不同命令消耗相同的规则随机数。
 * 未知没有评分，超限和不支持不跳过；输出命令与指纹，实际边界观察仅在内存返回。
 */
export function rollout(job: RolloutJob) {
  const original = JSON.stringify(job.observation);
  const root = decisionOwner(job.observation);
  const teacher = new TrainingTeacher('easy', 40);
  const steps: {
    actor: 1 | 2;
    before: string;
    after: string;
    command: Command;
    sample: number;
    work: number;
    cached: boolean;
  }[] = [];
  let o = job.observation;
  let work = 0;
  let unexecutedWork = 0;
  let stop = 'command-limit';
  let reason: string | undefined;
  const started = performance.now();
  const boundary = () =>
    o.ply === job.observation.ply + 2 &&
    o.active === job.observation.active &&
    !o.pending.length &&
    ['summon', 'synthesis'].includes(o.phase);
  while (true) {
    if (o.winner !== undefined) {
      stop = 'terminal';
      break;
    }
    if (!job.terminal && boundary()) {
      stop = 'boundary';
      break;
    }
    if (steps.length >= job.maxCommands) break;
    if (job.terminal && o.ply >= job.observation.ply + (job.maxPlies ?? 120)) {
      stop = 'ply-limit';
      break;
    }
    if (steps.length && work + 40 > job.maxWork) {
      stop = 'work-limit';
      break;
    }
    try {
      checkPosition(o);
    } catch (error) {
      stop = 'unsupported';
      reason = String(error);
      break;
    }
    let command = job.command;
    let cost = 0;
    let cached = false;
    if (steps.length) {
      try {
        const decision = teacher.next(o);
        assert.ok(decision.command);
        command = decision.command;
        cost = decision.stats.simulations;
        cached = decision.stats.cached ?? false;
        assert.ok(Number.isSafeInteger(cost) && cost >= 0);
      } catch (error) {
        stop = 'teacher-paused';
        reason = String(error);
        break;
      }
    }
    // 40是规划器的请求预算，完整概率枚举可原子性越界；按实际成本扣账，不能误判教师暂停。
    if (work + cost > job.maxWork) {
      work += cost;
      unexecutedWork = cost;
      stop = 'work-limit';
      break;
    }
    const actor = decisionOwner(o);
    const sample = hash(`counterfactual-transition:${job.sample}:${steps.length}`);
    const next = sampleTrainingTransition(o, actor, command, sample);
    steps.push({
      actor,
      before: fingerprint(o),
      after: fingerprint(next),
      command,
      sample,
      work: cost,
      cached,
    });
    work += cost;
    o = next;
  }
  assert.equal(JSON.stringify(job.observation), original);
  const complete = stop === 'terminal' || stop === 'boundary';
  return {
    stop,
    reason,
    complete,
    work,
    unexecutedWork,
    steps,
    elapsedMs: performance.now() - started,
    final: fingerprint(o),
    phase: o.phase,
    ply: o.ply,
    pending: o.pending.length,
    value: terminalValue(o, root),
    heuristic: complete ? evaluate(imagined(o), root) : null,
    observation: o,
  };
}

/** 仅共同完成的场景可排名；任一未知都保留，不用零分替代，不丢掉此前完整配对。 */
export function pairedRanking(rows: { complete: boolean; heuristic: number | null }[][]) {
  assert.ok(rows.length > 0 && rows.every((r) => r.length === rows[0].length));
  const paired = rows[0].map((_, i) => i).filter((i) => rows.every((r) => r[i].complete));
  const scores = rows.map((r) =>
    paired.length
      ? paired.reduce((sum, i) => {
          assert.notEqual(r[i].heuristic, null);
          return sum + r[i].heuristic!;
        }, 0) / paired.length
      : null,
  );
  const best = paired.length ? scores.reduce<number>((a, v, i) => (v! > scores[a]! ? i : a), 0) : 0;
  return { paired, scores, best, fallback: !paired.length };
}
