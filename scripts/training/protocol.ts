import { TrainingEnvironment, type TrainingOptions } from '../../src/match/training';
import {
  inspectTrainingCommand,
  trainingActionSpace,
  trainingGeometry,
} from '../../src/ai/training/queries';
import { sampleTrainingTransition, trainingDistribution } from '../../src/ai/training/simulation';
import { ensure } from '../../src/engine/core/state';
import type { Player } from '../../src/engine/types';

type Request = Record<string, unknown>;
function object(value: unknown): Request {
  ensure(value && typeof value === 'object' && !Array.isArray(value), '请求必须是JSON对象。');
  return value as Request;
}
function player(value: unknown): Player {
  ensure(value === 1 || value === 2, 'actor/viewer 必须是 1 或 2。');
  return value;
}

/**
 * 本地管道协议持有多个独立环境；所有返回值都是公开观察或公开查询。
 * batch 按顺序逐项返回成功/失败，不承诺跨环境事务；单条游戏命令仍保持原子性。
 * reset/close 是显式生命周期操作，错误请求不删除旧环境；无磁盘持久化与网络监听。
 */
export class TrainingService {
  readonly #environments = new Map<string, TrainingEnvironment>();

  handle(input: unknown): unknown {
    let id: string | number | null = null;
    try {
      const request = object(input);
      if (typeof request.id === 'string' || typeof request.id === 'number') id = request.id;
      if (request.op === 'batch') {
        ensure(
          Array.isArray(request.requests) && request.requests.length <= 256,
          'batch 最多256项。',
        );
        ensure(
          request.requests.every(
            (v) => !v || typeof v !== 'object' || (v as Request).op !== 'batch',
          ),
          '不支持嵌套batch。',
        );
        return { id, ok: true, results: request.requests.map((v) => this.handle(v)) };
      }
      ensure(
        typeof request.env === 'string' && request.env.length > 0 && request.env.length <= 128,
        'env 必须是非空短字符串。',
      );
      const key = request.env;
      if (request.op === 'reset') {
        ensure(
          this.#environments.has(key) || this.#environments.size < 128,
          '单进程最多128个环境，请先close。',
        );
        const options = request.options === undefined ? {} : object(request.options);
        for (const name of Object.keys(options))
          ensure(
            ['seed', 'rules', 'maxCommands', 'maxPlies'].includes(name),
            `不支持reset参数 ${name}。`,
          );
        const env = new TrainingEnvironment(options as TrainingOptions);
        const result = env.frame(request.viewer === undefined ? undefined : player(request.viewer));
        this.#environments.set(key, env);
        return { id, ok: true, result };
      }
      const env = this.#environments.get(key);
      ensure(env, '环境不存在，请先reset。');
      if (request.op === 'close') {
        this.#environments.delete(key);
        return { id, ok: true };
      }
      // 返回观察的席位不能改变落子操作者；只有 observe 使用 viewer 选择查询席位。
      const actor = player(
        (request.op === 'observe' ? (request.viewer ?? request.actor) : request.actor) ??
          env.status().toPlay ??
          1,
      );
      let result: unknown;
      switch (request.op) {
        case 'observe':
          result = env.frame(actor);
          break;
        case 'step': {
          // viewer/observe 的校验必须先于突变，不能成功落子后才返回参数错误。
          const viewer = request.viewer === undefined ? undefined : player(request.viewer);
          ensure(
            request.observe === undefined || typeof request.observe === 'boolean',
            'observe 必须是布尔值。',
          );
          const status = env.step(actor, request.command);
          result = request.observe === false ? status : env.frame(viewer);
          break;
        }
        case 'actions':
          result = trainingActionSpace(env.observation(actor), actor);
          break;
        case 'inspect':
          result = inspectTrainingCommand(env.observation(actor), actor, request.command);
          break;
        case 'geometry':
          result = trainingGeometry(env.observation(actor), request.command);
          break;
        case 'sample':
        case 'distribution': {
          ensure(typeof request.sampleSeed === 'number', '请显式提供独立的sampleSeed。');
          result = (request.op === 'sample' ? sampleTrainingTransition : trainingDistribution)(
            env.observation(actor),
            actor,
            request.command,
            request.sampleSeed,
          );
          break;
        }
        default:
          throw new Error('未知训练操作。');
      }
      return { id, ok: true, result };
    } catch (error) {
      return { id, ok: false, error: error instanceof Error ? error.message : String(error) };
    }
  }
}
