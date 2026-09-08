import { definition } from '../../../src/engine/catalog';
import { cells } from '../../../src/engine/geometry';
import { getStats } from '../../../src/engine/state';
import { decisionOwner } from '../../../src/ai/observation';
import { unitActions, cardActions } from '../../../src/engine/options';
import type { GameState, Command } from '../../../src/engine/types';
export function render(s: GameState): string {
  const out = [
    `ply ${s.ply} | P${s.active} ${s.phase} | 决策 P${decisionOwner(s)} | 基地 ${s.bases[1]}/${s.bases[2]} | 人头 ${s.heads[1]}/${s.heads[2]} | 召唤槽 ${s.summonSlots}`,
  ];
  out.push('     ' + Array.from({ length: 9 }, (_, i) => String(i + 1).padStart(4)).join(''));
  for (let y = 1; y <= 13; y++) {
    let line = String(y).padStart(3) + '  ';
    for (let x = 1; x <= 9; x++) {
      const index = s.units.findIndex((u) => cells(u).some((p) => p.x === x && p.y === y));
      line += (
        x === 5 && (y === 1 || y === 13)
          ? y === 1
            ? 'B1'
            : 'B2'
          : index < 0
            ? '·'
            : `${s.units[index].owner}${String(index + 1).padStart(2, '0')}`
      ).padStart(4);
    }
    out.push(line);
  }
  s.units.forEach((u, i) => {
    const st = getStats(s, u);
    out.push(
      `${String(i + 1).padStart(2, '0')} P${u.owner} ${u.id} ${definition(u.kind).name} (${u.x},${u.y}) HP${u.hp}/${u.maxHp} 攻${st.attack} 射${st.range} 剩攻${st.remaining} 操作${st.operationsLeft} ${u.mode} 蓄${u.charge}/${u.readyCharge}${st.sleeping ? ' 休' : ''}${st.frozen ? ' 冻' : ''}${u.effects.length ? ' [' + u.effects.map((e) => e.type).join(',') + ']' : ''}`,
    );
  });
  for (const p of [1, 2] as const)
    out.push(
      `P${p}手牌: ` +
        s.hands[p]
          .map(
            (c) =>
              `${c.id}=${definition(c.kind).name}${c.expiresAt ? `(余${c.expiresAt - s.turns[p]})` : ''}`,
          )
          .join(' | '),
    );
  if (s.pending.length)
    out.push('待反应: ' + s.pending.map((r) => `${r.kind}/P${r.owner}/${r.source.id}`).join(', '));
  if (s.winner) out.push(`结果: ${s.winner}`);
  return out.join('\n');
}
export function describe(s: GameState, c: Command): string {
  const name = (id?: string) =>
    id
      ? (s.units.find((u) => u.id === id)?.kind ?? s.hands[s.active].find((v) => v.id === id)?.kind)
      : undefined;
  const u = name(c.unitId),
    card = name(c.cardId),
    t = name(c.targetId);
  return `${c.type} ${u !== undefined ? definition(u).name : (c.unitId ?? '')}${card !== undefined ? ' ' + definition(card).name : ''}${c.targetId ? ' → ' + (t !== undefined ? definition(t).name : c.targetId) : ''}${c.x !== undefined ? ` (${c.x},${c.y})` : ''}${c.ultimate ? ' 终极' : ''}${c.mode ? ' ' + c.mode : ''}`.trim();
}
export function actions(s: GameState, id?: string): string {
  const unit = s.units.find((u) => u.id === id),
    card = [...s.hands[1], ...s.hands[2]].find((c) => c.id === id);
  const all = unit ? unitActions(s, unit) : card ? cardActions(s, card) : [];
  return (
    all
      .map(
        (a) =>
          `${a.label}: ${JSON.stringify(a.command)} ${a.steps.map((p) => p.label).join(' → ')}`,
      )
      .join('\n') || '使用 actions 单位ID/手牌ID 查看完整能力；也可直接输入任意引擎JSON命令。'
  );
}
