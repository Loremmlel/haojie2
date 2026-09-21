import test from 'node:test';
import assert from 'node:assert/strict';
import {
  CATALOG,
  KEYWORD_IDS,
  definition,
  effectKeyword,
  keywordDefinition,
  type Effect,
} from '../../src/engine';
import { referenceParts } from '../../src/ui/library/references';
import { unitStatus } from '../../src/ui/inspector/unit-status';
import { add, fixture } from '../helpers';

test('30个词条的来源和图鉴显式引用均可解析，状态类型没有遗漏', () => {
  assert.equal(KEYWORD_IDS.length, 30);
  assert.equal(new Set(KEYWORD_IDS.map((id) => keywordDefinition(id).name)).size, 30);
  for (const id of KEYWORD_IDS) {
    const d = keywordDefinition(id);
    assert.ok(d.description && d.sources.length);
    for (const kind of d.sources) assert.ok(definition(kind));
  }
  for (const d of CATALOG)
    for (const [text, id] of Object.entries(d.keywordReferences ?? {})) {
      assert.ok(d.description.includes(text), `${d.name}缺少显式引用文案：${text}`);
      assert.ok(keywordDefinition(id));
    }
  const types: Effect['type'][] = [
    'attack',
    'immune',
    'execute',
    'convert',
    'mark',
    'freeze',
    'burn',
    'stun',
    'inner-fire',
  ];
  for (const type of types)
    assert.ok(keywordDefinition(effectKeyword({ type, owner: 1, from: 0, until: 1 })));
  assert.equal(
    effectKeyword({ type: 'attack', owner: 2, from: 0, until: 1, amount: -15 }),
    'attack-down',
  );
});

test('长单位名称优先，规则同名词显式消歧，文本原样保留且战报不扩展关键词', () => {
  const text = '冲锋怪与冲锋号令、冲锋；金身与心灵之火，<script>冰冻</script>。';
  const parts = referenceParts(text, true, { 金身: 'immune', 心灵之火: 'inner-fire' });
  assert.equal(parts.map((p) => p.text).join(''), text);
  assert.deepEqual(
    parts.filter((p) => p.reference).map((p) => [p.text, p.reference]),
    [
      ['冲锋怪', { type: 'unit', kind: 1 }],
      ['冲锋号令', { type: 'unit', kind: 'u17' }],
      ['冲锋', { type: 'keyword', id: 'charge' }],
      ['金身', { type: 'keyword', id: 'immune' }],
      ['心灵之火', { type: 'keyword', id: 'inner-fire' }],
      ['冰冻', { type: 'keyword', id: 'freeze' }],
    ],
  );
  assert.deepEqual(referenceParts('金身', true)[0].reference, { type: 'unit', kind: 17 });
  assert.deepEqual(referenceParts('冲锋与冰冻')[0], { text: '冲锋与冰冻', reference: undefined });
  assert.deepEqual(referenceParts('地标休眠', true)[0], { text: '地标休眠', reference: undefined });
});

test('状态身份不随前一条效果移除而漂移，数值、来源和计时来自最新局面', () => {
  const s = fixture(),
    source = add(s, 'u6', 2, 3, 5),
    u = add(s, 9, 1, 3, 4);
  u.effects.push(
    { type: 'attack', owner: 2, from: 5, until: 6, global: true, amount: -15 },
    { type: 'burn', owner: 2, from: 5, until: 9, global: true, amount: 10, sourceId: source.id },
  );
  const before = structuredClone(s),
    rows = unitStatus(s, u),
    burn = rows.find((r) => r.keywords?.includes('burn'))!;
  assert.match(burn.detail, /各受10伤害.*大法师/);
  assert.match(rows[0].detail, /记录未保存来源/);
  assert.deepEqual(s, before);
  s.ply++;
  u.effects.shift();
  const current = unitStatus(s, u);
  assert.equal(current[0].key, burn.key);
  assert.match(current[0].detail, /余 3 次实际回合切换/);
  assert.ok(!current.some((r) => r.key === rows[0].key));
  s.units = s.units.filter((v) => v.id !== source.id);
  assert.match(unitStatus(s, u)[0].detail, /来源已离场/);
});

test('待生效附魔与同值多层效果各自可查，不生成额外局面数据', () => {
  const s = fixture(),
    u = add(s, 9, 1, 3, 4);
  const mark: Effect = { type: 'mark', owner: 2, from: 5, until: 7, global: true };
  u.effects.push({ type: 'execute', owner: 1, from: 7, until: 8, global: true }, mark, { ...mark });
  const rows = unitStatus(s, u);
  assert.equal(new Set(rows.map((r) => r.key)).size, 3);
  assert.equal(rows[0].pending, true);
  assert.match(rows[0].detail, /下个己方回合生效；冲锋号令不能提前/);
  assert.deepEqual(rows[0].keywords, ['execute']);
});
