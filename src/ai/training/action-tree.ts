import {
  ALL_CELLS,
  attackRoutes,
  cells,
  expansionAnchors,
  HEIGHT,
  neighbors,
  selectableAttackRoutes,
  targets,
  validAttackRoute,
  WIDTH,
} from '../../engine/core/geometry';
import { allegiance, ensure, getStats, piercing } from '../../engine/core/state';
import {
  entityAt,
  entityHandle,
  pieceById,
  registerPublicPosition,
  type EntityHandle,
} from '../../engine/runtime/position';
import { parseCommand } from '../../engine/online/authority';
import { SYNTHESIS_RECIPES, synthesisDestinations } from '../../engine/setup/synthesis';
import type { SelectionStep } from '../../engine/commands/options';
import type { Command, Player, Unit, Target } from '../../engine/types';
import type { Observation } from '../types';
import {
  createTrainingCommandInspector,
  trainingActionSpace,
  trainingPosition,
  type TrainingAction,
} from './queries';

type Step = SelectionStep | { kind: 'material' | 'chosen' };
interface Prefix {
  action: TrainingAction;
  command: Command;
  steps: Step[];
}
export interface ActionChoice {
  key: string;
  command: Command;
  status: 'parameter' | 'available' | 'uncertain';
  subject?: string;
  next?: Prefix;
}
export interface ActionNode {
  cursor: number[];
  prefix?: Command;
  stage: string;
  choices: ActionChoice[];
}

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
const matches = (prefix: Command, command: Command) =>
  Object.entries(prefix).every(([key, value]) => {
    if (value === undefined) return true;
    const target = command[key as keyof Command];
    return Array.isArray(value)
      ? Array.isArray(target) && value.every((v, i) => same(v, target[i]))
      : value === target;
  });

/** 统一规则默认值、旧合成别名和非玉碎的无效奇偶参数，不合并不同方向或路径。 */
export function canonicalTrainingCommand(
  observation: Observation,
  actor: Player,
  input: unknown,
): Command {
  return canonicalEngineCommand(observation, actor, parseCommand(input));
}

/** 引擎已经类型化的命令只补齐规则默认值，外部 JSON 校验留在上面的边界。 */
function canonicalEngineCommand(observation: Observation, actor: Player, input: Command): Command {
  const c = { ...input };
  if (c.type === 'craft') {
    c.type = 'synthesize';
    c.recipeId = 'firelord';
    c.materialIds = c.cardIds;
    delete c.cardIds;
  }
  if (c.type === 'deploy') c.charge ??= false;
  if (c.type === 'choose-shrine') {
    c.player = actor;
    // 网页选择器始终带奇偶值，规则只在玉碎上读取该参数。
    if (c.shrineKind !== 's9') delete c.parity;
  }
  if (c.type === 'summon' || c.type === 'extra-summon') c.ultimate ??= false;
  if (c.type === 'attack' && c.path) c.mode ??= 'damage';
  if (c.type === 'cast' && observation.hands[actor].find((v) => v.id === c.cardId)?.kind === 25)
    c.mode ??= 'single';
  return c;
}

/**
 * 在同一公开局面上惰性展开动作参数，游标只包含逐层下标；不修改局面、不采样正式随机数。
 * 中间parameter是公开选择域，不承诺存在合法补全；叶子才经共享公开预检过滤。
 * 路径逐格延伸并显式确认，材料逐个选择；没有教师评分裁剪、实体上限或静默截断。
 * 调用者遇到空分支须回溯，预算耗尽须明确暂停，不能改成自动结束回合。
 */
export class TrainingActionTree {
  readonly position;
  readonly actions: TrainingAction[];
  readonly #nodes = new Map<string, ActionNode>();
  readonly #inspect;
  readonly #targets: Target[];
  readonly #targetIndex = new Map<string, number>();
  readonly #stats = new Map<Unit, ReturnType<typeof getStats>>();
  readonly #routeCache = new Map<EntityHandle, Map<number, ReturnType<typeof attackRoutes>>>();
  constructor(
    readonly observation: Observation,
    readonly actor: Player,
  ) {
    this.position = registerPublicPosition(trainingPosition(observation));
    this.#targets = targets(this.position);
    for (const [index, target] of this.#targets.entries())
      if (!this.#targetIndex.has(target.id)) this.#targetIndex.set(target.id, index);
    this.#inspect = createTrainingCommandInspector(observation, actor);
    this.actions = trainingActionSpace(observation, actor, this.#inspect).actions;
  }

  #prepare(prefix: Prefix): Prefix {
    let { steps } = prefix;
    const { command } = prefix;
    while (steps[0]?.kind === 'direction' && this.#routes(command).length === 0)
      steps = steps.slice(1);
    return { ...prefix, steps };
  }

  #routes(command: Command) {
    const source =
      command.unitId === undefined ? undefined : entityHandle(this.position, command.unitId);
    const target = this.#targetIndex.get(command.targetId!);
    if (source === undefined || target === undefined) return [];
    const cached = this.#routeCache.get(source)?.get(target);
    if (cached) return cached;
    const u = entityAt(this.position, source)!;
    const t = this.#targets[target];
    const selectable = selectableAttackRoutes(this.position, u, t);
    const routes = selectable.length
      ? selectable
      : attackRoutes(this.position, u, t, this.#attributes(u).range, piercing(u));
    let records = this.#routeCache.get(source);
    if (!records) this.#routeCache.set(source, (records = new Map()));
    records.set(target, routes);
    return routes;
  }
  #attributes(u: Unit) {
    let stats = this.#stats.get(u);
    if (!stats) {
      stats = getStats(this.position, u);
      this.#stats.set(u, stats);
    }
    return stats;
  }

  #choice(key: string, prefix: Prefix, subject?: string): ActionChoice | null {
    prefix = this.#prepare(prefix);
    if (prefix.steps.length)
      return { key, command: prefix.command, next: prefix, subject, status: 'parameter' };
    const result = this.#inspect(prefix.command);
    return result.status === 'invalid'
      ? null
      : { key, command: prefix.command, subject, status: result.status };
  }

  #start(action: TrainingAction): Prefix {
    const command = canonicalEngineCommand(this.observation, this.actor, action.command);
    const steps: Step[] = [
      ...Array.from({ length: action.materialCount ?? 0 }, (): Step => ({ kind: 'material' })),
      ...action.steps,
      ...(action.chosenKinds?.length ? [{ kind: 'chosen' as const }] : []),
    ];
    if (command.type === 'attack' && !command.path) steps.push({ kind: 'direction', label: '' });
    return { action, command, steps };
  }

  #expand(prefix: Prefix): ActionChoice[] {
    const { command: c, action, steps } = prefix;
    const step = steps[0];
    const result: ActionChoice[] = [];
    // 同一节点的后续参数说明只读；需要推进时才创建共享尾部，空分支不分配。
    let remaining: Step[] | undefined;
    const add = (key: string, command: Command, subject?: string, repeat = false) => {
      const choice = this.#choice(
        key,
        { action, command, steps: repeat ? steps : (remaining ??= steps.slice(1)) },
        subject,
      );
      if (choice) result.push(choice);
    };
    switch (step.kind) {
      case 'material':
        for (const id of action.materialIds ?? [])
          if (!c.materialIds?.includes(id))
            add(id, { ...c, materialIds: [...(c.materialIds ?? []), id] }, id);
        break;
      case 'chosen':
        // 自选是可用能力，随机召唤仍是合法选择，不强迫消耗本回合自选次数。
        add('random', c);
        for (const kind of action.chosenKinds ?? [])
          add(`kind:${kind}`, { ...c, chosenKind: kind });
        break;
      case 'target':
        for (const t of this.#targets) {
          if (step.unitOnly && !t.unit) continue;
          // 墓地保留来源 owner，但目标关系必须使用引擎派生的当前阵营。
          const owner = t.unit ? allegiance(this.position, t.unit) : t.owner;
          if (step.relation === 'friend' && owner !== this.actor) continue;
          if (step.relation === 'enemy' && owner === this.actor) continue;
          if (step.field === 'sacrificeIds' && c.sacrificeIds?.includes(t.id)) continue;
          add(
            t.id,
            step.field === 'sacrificeIds'
              ? { ...c, sacrificeIds: [...(c.sacrificeIds ?? []), t.id] }
              : { ...c, [step.field ?? 'targetId']: t.id },
            t.id,
          );
        }
        break;
      case 'point': {
        let points = ALL_CELLS;
        if (c.type === 'synthesize' && c.materialIds?.length === 3) {
          const recipe = SYNTHESIS_RECIPES.find((r) => r.id === c.recipeId)!;
          points = synthesisDestinations(this.position, recipe, c.materialIds);
        } else if (action.id.split(':')[0] === 'giant' && c.targetId) {
          const target = pieceById(this.position, c.targetId);
          points = target ? expansionAnchors(this.position, target) : [];
        }
        for (const p of points) add(`${p.x},${p.y}`, { ...c, ...p });
        break;
      }
      case 'death':
        for (const d of this.position.deaths) add(d.id, { ...c, deathId: d.id }, d.id);
        break;
      case 'row':
      case 'column':
        for (let n = 1; n <= (step.kind === 'row' ? HEIGHT : WIDTH); n++)
          add(String(n), { ...c, [step.kind]: n });
        break;
      case 'direction':
        add('auto', c);
        for (const route of this.#routes(c))
          add(route.direction, { ...c, direction: route.direction });
        break;
      case 'path': {
        const u = pieceById(this.position, c.unitId);
        if (!u) break;
        const path = c.path ?? [];
        if (path.length) add('commit-path', c);
        const points = path.length ? neighbors(path.at(-1)!) : cells(u);
        for (const p of points) {
          const extended = [...path, p];
          if (validAttackRoute(this.position, u, extended, this.#attributes(u).range))
            add(`${p.x},${p.y}`, { ...c, path: extended }, undefined, true);
        }
        break;
      }
    }
    return result;
  }

  node(cursor: readonly number[] = []): ActionNode {
    ensure(
      Array.isArray(cursor) &&
        cursor.length <= 256 &&
        cursor.every((n) => Number.isSafeInteger(n) && n >= 0),
      '动作游标必须是有界非负整数数组。',
    );
    const key = cursor.join(',');
    const cached = this.#nodes.get(key);
    if (cached) return cached;
    let node: ActionNode;
    if (!cursor.length) {
      const choices = this.actions.flatMap((a, i) => {
        const choice = this.#choice(String(i), this.#start(a));
        return choice ? [choice] : [];
      });
      node = { cursor: [], stage: 'action', choices };
    } else {
      const parent = this.node(cursor.slice(0, -1));
      const selected = parent.choices[cursor.at(-1)!];
      ensure(selected?.next, '动作游标越界或已经指向完整命令。');
      const next = this.#prepare(selected.next);
      node = {
        cursor: [...cursor],
        prefix: next.command,
        stage: next.steps[0].kind,
        choices: this.#expand(next),
      };
    }
    this.#nodes.set(key, node);
    return node;
  }

  /** 教师标签只用于查找目标分支，不改变候选集合；未知动作明确报错。 */
  trace(input: unknown): { node: ActionNode; selected: number }[] {
    const target = canonicalTrainingCommand(this.observation, this.actor, input);
    ensure(this.#inspect(target).status !== 'invalid', '教师命令未通过公开预检。');
    const visit = (cursor: number[]): { node: ActionNode; selected: number }[] | null => {
      const node = this.node(cursor);
      const options = node.choices
        .map((choice, i) => ({ choice, i }))
        .filter(({ choice }) => matches(choice.command, target))
        .sort(
          (a, b) => Object.keys(b.choice.command).length - Object.keys(a.choice.command).length,
        );
      for (const { choice, i } of options) {
        if (choice.next) {
          const rest = visit([...cursor, i]);
          if (rest) return [{ node, selected: i }, ...rest];
        } else if (matches(target, choice.command)) return [{ node, selected: i }];
      }
      return null;
    };
    const path = visit([]);
    ensure(path, `教师动作未被分解动作树覆盖：${JSON.stringify(target)}`);
    return path;
  }
}
