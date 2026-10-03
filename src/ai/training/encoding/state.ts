import { CATALOG } from '../../../engine/catalog';
import { basePoint, other } from '../../../engine/core/geometry';
import { ensure, getStats } from '../../../engine/core/state';
import { SYNTHESIS_RECIPES } from '../../../engine/setup/synthesis';
import type { Card, Command, Effect, Kind, Landmark, Player, Unit } from '../../../engine/types';
import type { Observation } from '../../types';
import { trainingPosition } from '../queries';
import {
  category,
  COMMANDS,
  DIRECTIONS,
  EFFECTS,
  kindFromKey,
  kindIndex,
  knownKeys,
  MODES,
  numeric,
  PHASES,
  REACTIONS,
  ROLES,
  UNIT_FIELDS,
  valuesInto,
  valueInto,
  type EntityRole,
} from './schema';

const recipeIds = SYNTHESIS_RECIPES.map((r) => r.id);
const printedValues = new Map(
  CATALOG.map((d) => [
    d.id,
    [
      d.attack,
      d.health,
      d.range,
      d.actions,
      d.move,
      d.size ?? 1,
      d.mage,
      d.spell,
      d.weapon,
      d.aura,
    ].map((value) => (value === undefined ? undefined : numeric(value))),
  ]),
);
function printedInto(row: number[], offset: number, kind: Kind) {
  const values = printedValues.get(kind)!;
  for (let i = 0; i < values.length; i++) {
    if (values[i] === undefined) continue;
    const at = offset + i;
    row[8 + at] = values[i]!;
    row[60 + Math.floor(at / 13)] += 2 ** (at % 13) / 8191;
  }
}

const OBSERVATION_FIELDS = [
  'version',
  'serial',
  'ply',
  'active',
  'phase',
  'mode',
  'landmarks',
  'auras',
  'shrineDraft',
  'shrineSetupDone',
  'regularSummons',
  'summonOffer',
  'clockFrames',
  'summonSlots',
  'turns',
  'bases',
  'baseEffects',
  'heads',
  'hands',
  'bonus',
  'deployRows',
  'units',
  'pending',
  'deaths',
  'hazards',
  'siphons',
  'iceMarks',
  'winner',
];
const CARD_FIELDS = [
  'id',
  'kind',
  'drawnAt',
  'expiresAt',
  'group',
  'rerolled',
  'summonedPly',
  'summonPool',
  'parity',
];
const UNIT_KEYS = [
  ...UNIT_FIELDS,
  'id',
  'kind',
  'owner',
  'attacked',
  'guardSourceIds',
  'effects',
  'equipment',
  'traits',
  'abilityUsage',
  'abilityCharges',
  'equipmentIds',
  'receivedDamage',
  'group',
];
const roleIndices = new Map(ROLES.map((role, i) => [role, i]));
type Scalar = number | boolean | undefined;
interface RecordOptions {
  id?: string;
  owner?: Player;
  kind?: Kind;
  parent?: string;
  source?: string;
  target?: string;
  group?: string;
  order?: number;
  fields?: Scalar[];
  selectable?: boolean;
}

interface PositionRows {
  entities: number[][];
  kinds: number[];
  indices: Map<string, number>;
  identities: Map<string, number>;
}

/** 每个前缀拥有独立的张量与引用编号，调用者修改返回值不能污染后续编码。 */
function positionRows(viewer: Player, base?: PositionRows, borrowFixed = false) {
  const entities = base
    ? borrowFixed
      ? [...base.entities]
      : base.entities.map((row) => [...row])
    : [];
  const kinds: number[] = base ? [...base.kinds] : [];
  const indices = borrowFixed && base ? base.indices : new Map<string, number>(base?.indices);
  const identities = new Map<string, number>(borrowFixed ? undefined : base?.identities);
  const fixedIdentities = borrowFixed ? base?.identities : undefined;
  const reference = (id?: string): number => {
    if (id === undefined) return 0;
    ensure(typeof id === 'string' && !!id, '实体引用必须是非空字符串。');
    const fixed = fixedIdentities?.get(id);
    if (fixed !== undefined) return fixed;
    if (!identities.has(id)) identities.set(id, (fixedIdentities?.size ?? 0) + identities.size + 1);
    return identities.get(id)!;
  };
  const owner = (p?: Player) => {
    ensure(p === undefined || p === 1 || p === 2, '实体所属玩家不合法。');
    return p === undefined ? 0 : p === viewer ? 1 : -1;
  };
  const add = (role: EntityRole, o: RecordOptions = {}) => {
    const row = Array<number>(64).fill(0);
    const roleIndex = roleIndices.get(role)!;
    row[0] = (roleIndex + 1) / 32;
    row[1] = owner(o.owner);
    row[2] = reference(o.id) / 256;
    row[3] = reference(o.parent) / 256;
    row[4] = reference(o.source) / 256;
    row[5] = reference(o.target) / 256;
    row[6] = reference(o.group === undefined ? undefined : `group:${o.group}`) / 256;
    row[7] = (o.order ?? 0) / 128;
    valuesInto(row, o.fields ?? []);
    if (o.selectable && o.id && !indices.has(o.id)) indices.set(o.id, entities.length);
    entities.push(row);
    kinds.push(o.kind === undefined ? 128 + roleIndex : kindIndex(o.kind));
    return row;
  };
  return { entities, kinds, indices, identities, reference, owner, add };
}

/**
 * 公开观察的纯编码器，不读取正式PRNG、不修改输入、不解析ID中的数字。
 * 主对象与附属状态各为一行；引用重编号保留同一来源/克隆组的关联，数组顺序保留叠放语义。
 * 所有已知嵌套字段有明确映射；未知字段或词表项报错，不能悄悄忽略新规则。
 * 不截断实体、亡者、反应或时钟快照。返回的索引表供动作编码使用，不是网络输入。
 */
function encodeBasePosition(observation: Observation, viewer: Player) {
  const s = trainingPosition(observation);
  ensure(viewer === 1 || viewer === 2, '编码观察方必须为1或2。');
  knownKeys(observation, OBSERVATION_FIELDS, 'Observation');
  ensure(observation.version === 2, '不支持此观察版本。');
  const rows = positionRows(viewer);
  const { entities, indices, reference, owner, add } = rows;
  const effect = (e: Effect, parent: string, order: number) => {
    knownKeys(e, ['type', 'from', 'until', 'owner', 'amount', 'sourceId', 'global'], 'Effect');
    add('effect', {
      owner: e.owner,
      parent,
      source: e.sourceId,
      order,
      fields: [category(e.type, EFFECTS), e.from, e.until, e.amount, e.global],
    });
  };
  const unit = (
    u: Unit | Landmark,
    role: 'unit' | 'landmark' | 'snapshot',
    parent?: string,
    order = 0,
  ) => {
    knownKeys(u, UNIT_KEYS, 'Unit');
    const instance = role === 'snapshot' ? `${parent}:snapshot:${order}` : u.id;
    if (!indices.has(u.id)) indices.set(u.id, entities.length);
    const row = add(role, {
      id: instance,
      source: role === 'snapshot' ? u.id : undefined,
      kind: u.kind,
      owner: u.owner,
      parent,
      group: u.group,
      order,
    });
    valueInto(row, 0, u.x);
    valueInto(row, 1, u.y);
    valueInto(row, 2, u.hp);
    valueInto(row, 3, u.maxHp);
    valueInto(row, 4, u.size);
    valueInto(row, 5, u.born);
    valueInto(row, 6, u.offset);
    valueInto(row, 7, u.deployedAt);
    valueInto(row, 8, u.chargedOnDeploy);
    valueInto(row, 9, category(u.mode, MODES));
    valueInto(row, 10, u.operations);
    valueInto(row, 11, u.shots);
    valueInto(row, 12, u.moves);
    valueInto(row, 13, u.bonusAttacks);
    valueInto(row, 14, u.bonusSequence);
    valueInto(row, 15, u.weaponFirstUsed);
    valueInto(row, 16, u.charge);
    valueInto(row, 17, u.readyCharge);
    valueInto(row, 18, category(u.chargeType, MODES));
    valueInto(row, 19, u.lastCharge);
    valueInto(row, 20, u.upgrades);
    valueInto(row, 21, u.kills);
    valueInto(row, 22, u.attackBonus);
    valueInto(row, 23, u.rangeBonus);
    valueInto(row, 24, u.guardUsed);
    valueInto(row, 25, u.rerollUsedPly);
    valueInto(row, 26, u.silenced);
    valueInto(row, 27, u.freeUsed);
    valueInto(row, 28, u.onceUsed);
    valueInto(row, 29, u.extraOperations);
    valueInto(row, 30, u.bannerHp);
    valueInto(row, 31, u.overMaxFromBanner);
    valueInto(row, 32, u.bladeQualified);
    valueInto(row, 33, u.expiresAt);
    valueInto(row, 34, u.hookReadyAt);
    valueInto(row, 35, u.hookExpiresAt);
    valueInto(row, 36, (u as Landmark).dormantSince);
    valueInto(row, 37, (u as Landmark).rebuildTicks);
    if (role !== 'snapshot') {
      const st = getStats(s, u);
      valueInto(row, 38, st.attack);
      valueInto(row, 39, st.range);
      valueInto(row, 40, st.actions);
      valueInto(row, 41, st.remaining);
      valueInto(row, 42, st.move);
      valueInto(row, 43, st.operationLimit);
      valueInto(row, 44, st.operationsLeft);
      valueInto(row, 45, st.sleeping);
      valueInto(row, 46, st.frozen);
      valueInto(row, 47, st.stunned);
    }
    u.effects.forEach((e, i) => effect(e, instance, i));
    u.attacked.forEach((id, i) =>
      add('attacked', {
        parent: instance,
        target: id,
        owner: u.owner,
        order: i,
      }),
    );
    u.guardSourceIds?.forEach((id, i) =>
      add('guard', { parent: instance, source: id, owner: u.owner, order: i }),
    );
    u.receivedDamage?.forEach((hit, i) => {
      knownKeys(hit, ['ply', 'amount'], 'receivedDamage');
      add('received', {
        parent: instance,
        owner: u.owner,
        order: i,
        fields: [hit.ply, hit.amount],
      });
    });
    const equipment = [
      ...new Set([...u.equipment, ...Object.keys(u.equipmentIds ?? {}).map(kindFromKey)]),
    ];
    equipment.forEach((kind, i) => {
      const row = add('equipment', {
        parent: instance,
        kind,
        owner: u.owner,
        id: u.equipmentIds?.[kind],
        order: i,
        fields: [u.equipment.includes(kind)],
      });
      printedInto(row, 1, kind);
    });
    const abilities = [
      ...new Set([
        ...(u.traits ?? []),
        ...Object.keys(u.abilityUsage ?? {}).map(kindFromKey),
        ...Object.keys(u.abilityCharges ?? {}).map(kindFromKey),
      ]),
    ];
    for (const kind of abilities) {
      const usage = u.abilityUsage?.[kind],
        charge = u.abilityCharges?.[kind];
      if (usage) knownKeys(usage, ['once', 'free'], 'abilityUsage');
      if (charge)
        knownKeys(charge, ['charge', 'readyCharge', 'chargeType', 'lastCharge'], 'abilityCharges');
      add('ability', {
        parent: instance,
        kind,
        owner: u.owner,
        fields: [
          u.traits?.includes(kind) ?? false,
          usage?.once,
          usage?.free,
          charge?.charge,
          charge?.readyCharge,
          category(charge?.chargeType, MODES),
          charge?.lastCharge,
        ],
      });
    }
  };
  const card = (c: Card, p: Player, role: 'card' | 'summon-offer', parent?: string, order = 0) => {
    knownKeys(c, CARD_FIELDS, 'Card');
    const row = add(role, {
      id: c.id,
      kind: c.kind,
      owner: p,
      parent,
      group: c.group,
      order,
      selectable: true,
      fields: [
        c.drawnAt,
        c.expiresAt,
        c.rerolled,
        c.summonedPly,
        category(c.summonPool, ['normal', 'ultimate']),
        category(c.parity, ['odd', 'even']),
      ],
    });
    printedInto(row, 6, c.kind);
  };
  const sides = [viewer, other(viewer)];
  // 预注册主对象；重命名ID不会改变网络张量，附属引用也不会扰乱后续主对象的编号。
  for (const p of sides) reference(`base-${p}`);
  for (const u of [...s.units, ...(s.landmarks ?? [])]) reference(u.id);
  for (const p of sides) for (const c of s.hands[p]) reference(c.id);
  for (const d of s.deaths) reference(d.id);
  for (const p of sides) {
    for (const pair of [s.turns, s.bases, s.heads, s.hands, s.bonus, s.baseEffects, s.deployRows])
      knownKeys(pair, ['1', '2'], '玩家记录');
    const at = basePoint(p);
    add('base', {
      id: `base-${p}`,
      owner: p,
      selectable: true,
      fields: [
        at.x,
        at.y,
        s.bases[p],
        s.turns[p],
        s.heads[p],
        s.bonus[p],
        ...Array.from({ length: 13 }, (_, i) => s.deployRows[p].includes(i + 1)),
      ],
    });
    s.baseEffects[p].forEach((e, i) => effect(e, `base-${p}`, i));
  }
  s.units.forEach((u, i) => unit(u, 'unit', undefined, i));
  s.landmarks?.forEach((u, i) => unit(u, 'landmark', undefined, i));
  for (const p of sides) s.hands[p].forEach((c, i) => card(c, p, 'card', undefined, i));
  s.deaths.forEach((d, i) => {
    knownKeys(d, ['id', 'kind', 'owner', 'ply', 'revived', 'group'], 'DeathRecord');
    const row = add('death', {
      ...d,
      order: i,
      selectable: true,
      fields: [d.ply, d.revived],
    });
    printedInto(row, 2, d.kind);
  });
  s.hazards.forEach((h, i) => {
    knownKeys(h, ['id', 'owner', 'sourceId', 'axis', 'line', 'due'], 'Hazard');
    add('hazard', {
      id: h.id,
      owner: h.owner,
      source: h.sourceId,
      order: i,
      fields: [category(h.axis, ['row', 'column']), h.line, h.due],
    });
  });
  s.siphons.forEach((l, i) => {
    knownKeys(l, ['id', 'sourceId', 'owner', 'fromId', 'toId'], 'Siphon');
    add('siphon', {
      id: l.id,
      owner: l.owner,
      parent: l.fromId,
      source: l.sourceId,
      target: l.toId,
      order: i,
    });
  });
  s.iceMarks.forEach((m, i) => {
    knownKeys(m, ['id', 'sourceId', 'owner', 'due', 'x', 'y'], 'IceMark');
    add('ice', {
      id: m.id,
      owner: m.owner,
      source: m.sourceId,
      order: i,
      fields: [m.x, m.y, m.due],
    });
  });
  s.pending.forEach((r, i) => {
    knownKeys(r, ['kind', 'targetId', 'owner', 'source', 'amount'], 'Reaction');
    const id = `reaction:${i}`;
    add('reaction', {
      id,
      owner: r.owner,
      source: r.source.id,
      target: r.targetId,
      order: i,
      fields: [category(r.kind, REACTIONS), r.amount],
    });
    unit(r.source, 'snapshot', id);
  });
  if (s.auras) knownKeys(s.auras, ['1', '2'], 'auras');
  if (s.clockFrames) knownKeys(s.clockFrames, ['1', '2'], 'clockFrames');
  for (const p of sides) {
    s.auras?.[p].forEach((a, i) => {
      knownKeys(a, ['kind', 'parity', 'usedPly'], 'Aura');
      add('aura', {
        kind: a.kind,
        owner: p,
        order: i,
        fields: [category(a.parity, ['odd', 'even']), a.usedPly],
      });
    });
    const frames = s.clockFrames?.[p];
    if (frames) knownKeys(frames, ['current', 'previous'], 'ClockFrames');
    for (const [i, name] of (['current', 'previous'] as const).entries()) {
      const frame = frames?.[name];
      if (!frame) continue;
      knownKeys(frame, ['ply', 'turns', 'units'], 'ClockFrame');
      knownKeys(frame.turns, ['1', '2'], 'ClockFrame.turns');
      const id = `frame:${p}:${name}`;
      add('frame', {
        id,
        owner: p,
        order: i,
        fields: [frame.ply, frame.turns[viewer], frame.turns[other(viewer)]],
      });
      frame.units.forEach((u, j) => unit(u, 'snapshot', id, j));
    }
  }
  const draft = s.shrineDraft;
  if (draft) {
    knownKeys(draft, ['offers', 'committed', 'choices', 'revealed'], 'ShrineDraft');
    for (const pair of [draft.offers, draft.committed, draft.choices])
      knownKeys(pair, ['1', '2'], 'ShrineDraft玩家记录');
    ensure(
      draft.revealed || draft.choices[other(viewer)] === undefined,
      '编码输入泄露尚未揭示的对方神龛选择。',
    );
    for (const p of sides) {
      draft.offers[p].forEach((kind, i) => add('draft-offer', { kind, owner: p, order: i }));
      const choice = draft.choices[p];
      if (choice) {
        knownKeys(choice, ['kind', 'parity'], 'ShrineChoice');
        add('draft-choice', {
          kind: choice.kind,
          owner: p,
          fields: [category(choice.parity, ['odd', 'even'])],
        });
      }
    }
  }
  if (s.summonOffer) {
    knownKeys(s.summonOffer, ['owner', 'groups', 'count'], 'SummonOffer');
    for (const [i, group] of s.summonOffer.groups.entries())
      group.forEach((c, j) => card(c, s.summonOffer!.owner, 'summon-offer', `offer:${i}`, j));
  }
  const globals = [
    viewer / 2,
    owner(s.active),
    ...PHASES.map((phase) => Number(s.phase === phase)),
    Number(s.mode === 'shrine'),
    numeric(s.ply),
    numeric(s.summonSlots),
    numeric(s.turns[viewer]),
    numeric(s.turns[other(viewer)]),
    numeric(s.bases[viewer]),
    numeric(s.bases[other(viewer)]),
    numeric(s.heads[viewer]),
    numeric(s.heads[other(viewer)]),
    numeric(s.bonus[viewer]),
    numeric(s.bonus[other(viewer)]),
    numeric(s.regularSummons ?? 0),
    s.winner === undefined || s.winner === 'draw' ? 0 : owner(s.winner),
    Number(s.winner !== undefined),
    numeric(s.pending.length),
    numeric(s.hands[viewer].length),
    numeric(s.hands[other(viewer)].length),
    Number(s.shrineSetupDone?.includes(viewer) ?? false),
    Number(s.shrineSetupDone?.includes(other(viewer)) ?? false),
    Number(draft?.committed[viewer] ?? false),
    Number(draft?.committed[other(viewer)] ?? false),
    Number(draft?.revealed ?? false),
    numeric(s.summonOffer?.count ?? 0),
    s.summonOffer ? owner(s.summonOffer.owner) : 0,
    0,
  ];
  ensure(globals.length === 32, '全局编码长度发生变化。');
  return { ...rows, globals };
}

/** 单次编码保留原入口；多节点调用者显式创建局面内编码器。 */
export function encodePosition(observation: Observation, viewer: Player, prefix?: Command) {
  return createPositionEncoder(observation, viewer)(prefix);
}

/**
 * 仅在同一不可变公开观察和观察方内惰性复用固定编码；改变局面须新建编码器。
 * 固定实体、词表校验与主身份表只构造一次，前缀和候选注册的引用始终隔离。
 */
export function createPositionEncoder(observation: Observation, viewer: Player) {
  return positionEncoder(observation, viewer, false);
}
/** 内部采样只读固定实体行；前缀身份编号使用局部增量，不能暴露给可修改张量调用者。 */
export function createSamplingPositionEncoder(
  observation: Observation,
  viewer: Player,
  reuseBuffers = false,
) {
  return positionEncoder(observation, viewer, true, reuseBuffers);
}
function positionEncoder(
  observation: Observation,
  viewer: Player,
  borrowFixed: boolean,
  reuseBuffers = false,
) {
  let base: ReturnType<typeof encodeBasePosition> | undefined;
  let workspace: ReturnType<typeof positionRows> | undefined;
  let globalBuffer: number[] | undefined;
  return (prefix?: Command) => {
    base ??= encodeBasePosition(observation, viewer);
    const rows = reuseBuffers
      ? (workspace ??= positionRows(viewer, base, true))
      : positionRows(viewer, base, borrowFixed);
    if (reuseBuffers) {
      // 上一个同步消费者已经结束；固定列表只建一次，前缀身份不能跨节点残留。
      rows.entities.length = base.entities.length;
      rows.kinds.length = base.kinds.length;
      rows.identities.clear();
    }
    const { entities, kinds, indices, reference, owner, add } = rows;
    if (prefix) {
      add('prefix', {
        id: 'prefix',
        owner: viewer,
        source: prefix.unitId ?? prefix.cardId,
        target: prefix.targetId,
        fields: [
          category(prefix.type, COMMANDS),
          category(prefix.mode, MODES),
          prefix.x,
          prefix.y,
          prefix.row,
          prefix.column,
          prefix.ultimate,
          prefix.charge,
          category(prefix.direction, DIRECTIONS),
          prefix.ability === undefined ? undefined : kindIndex(prefix.ability),
          prefix.chosenKind === undefined ? undefined : kindIndex(prefix.chosenKind),
          prefix.shrineKind === undefined ? undefined : kindIndex(prefix.shrineKind),
          category(prefix.parity, ['odd', 'even']),
          prefix.recipeId === undefined ? undefined : category(prefix.recipeId, recipeIds),
          prefix.player === undefined ? undefined : owner(prefix.player),
        ],
      });
      for (const [field, ids] of [
        ['secondId', prefix.secondId ? [prefix.secondId] : []],
        ['deathId', prefix.deathId ? [prefix.deathId] : []],
        ['materialIds', prefix.materialIds ?? []],
        ['cardIds', prefix.cardIds ?? []],
        ['sacrificeIds', prefix.sacrificeIds ?? []],
      ] as const)
        ids.forEach((id, i) =>
          add('argument', {
            parent: 'prefix',
            target: id,
            order: i,
            fields: [
              category(field, ['secondId', 'deathId', 'materialIds', 'cardIds', 'sacrificeIds']),
            ],
          }),
        );
      prefix.path?.forEach((p, i) =>
        add('path', { parent: 'prefix', order: i, fields: [p.x, p.y] }),
      );
      prefix.offerIndices?.forEach((n, i) =>
        add('argument', { parent: 'prefix', order: i, fields: [6, n] }),
      );
    }
    const globals = reuseBuffers ? (globalBuffer ??= [...base.globals]) : [...base.globals];
    globals[31] = Number(prefix !== undefined);
    return { entities, kinds, globals, indices, reference };
  };
}
