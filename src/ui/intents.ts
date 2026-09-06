import { asTarget, attackPath, definition, getStats, isLegal, movementPath, targetAt, targets } from '../engine';
import type { Command, GameState, Point } from '../engine';
export type Intent =
  | { kind:'none' }
  | { kind:'deploy'; cardId:string; charge:boolean }
  | { kind:'move' | 'attack' | 'hook-target' | 'sacrifice-pick' | 'wall' | 'dash-move'; unitId:string }
  | { kind:'hook-drop' | 'sacrifice-column'; unitId:string; targetId:string }
  | { kind:'dash-attack'; unitId:string; to:Point }
  | { kind:'cast'; cardId:string }
  | { kind:'reforge'; cardId:string; targets:string[] };
export function commandFor(s:GameState,i:Intent,p:Point): Command | null {
  const target = targetAt(s,p);
  if (s.pending.length) return target ? { type:'react',targetId:target.id } : null;
  switch (i.kind) {
    case 'deploy': return { type:'deploy',cardId:i.cardId,...p,charge:i.charge };
    case 'move': return { type:'move',unitId:i.unitId,...p };
    case 'attack': return target ? { type:'attack',unitId:i.unitId,targetId:target.id } : null;
    case 'hook-drop': return { type:'skill',unitId:i.unitId,targetId:i.targetId,...p };
    case 'sacrifice-column': return { type:'skill',unitId:i.unitId,targetId:i.targetId,column:p.x };
    case 'wall': return { type:'skill',unitId:i.unitId,...p };
    case 'dash-attack': return target ? { type:'skill',unitId:i.unitId,mode:'dash',...i.to,targetId:target.id } : null;
    case 'cast': {
      const card = s.hands[s.active].find(c => c.id === i.cardId);
      if (card?.kind === 8) return { type:'cast',cardId:i.cardId,...p };
      return target ? { type:'cast',cardId:i.cardId,targetId:target.id } : null;
    }
    default: return null;
  }
}
export function canChoose(s:GameState,i:Intent,p:Point): boolean {
  if (s.winner) return false;
  const command = commandFor(s,i,p);
  if (command) return isLegal(s,command);
  const t = targetAt(s,p), u = 'unitId' in i ? s.units.find(u => u.id === i.unitId) : undefined;
  if (i.kind === 'reforge') return !!t?.unit && t.owner === s.active && t.unit.hp*2 >= t.unit.maxHp;
  if (!u || u.owner !== s.active || getStats(s,u).remaining <= 0) return false;
  if (i.kind === 'hook-target') return !!t?.unit && t.owner !== u.owner && !!attackPath(s,u,t,getStats(s,u).range);
  if (i.kind === 'sacrifice-pick') return !!t?.unit && t.id !== u.id && t.owner === u.owner && Array.from({length:9},(_,n) => n+1).some(column => isLegal(s,{type:'skill',unitId:u.id,targetId:t.id,column}));
  if (i.kind === 'dash-move') {
    if (u.charge < 2 || u.lastCharge >= s.turns[u.owner] || u.hp <= 10 || !movementPath(s,u,p,6)) return false;
    return targets(s).some(t => t.owner !== u.owner && !!attackPath(s,{...u,...p},t,getStats(s,u).range));
  }
  return false;
}
export function nextIntent(s:GameState,i:Intent,p:Point): Intent {
  const t = targetAt(s,p);
  if (i.kind === 'hook-target' && t) return { kind:'hook-drop',unitId:i.unitId,targetId:t.id };
  if (i.kind === 'sacrifice-pick' && t) return { kind:'sacrifice-column',unitId:i.unitId,targetId:t.id };
  if (i.kind === 'dash-move') return { kind:'dash-attack',unitId:i.unitId,to:p };
  if (i.kind === 'reforge' && t) return { ...i, targets:i.targets.includes(t.id) ? i.targets.filter(id => id !== t.id) : [...i.targets,t.id] };
  return i;
}
export function instruction(s:GameState,i:Intent): string {
  const reaction = s.pending[0];
  if (reaction) return `${reaction.owner === 1 ? '苍穹方' : '赤焰方'}选择${reaction.kind === 'death-shot' ? '奶妈的临终攻击 / 治疗目标' : `伤害转化目标（${reaction.amount}伤害）`}`;
  switch (i.kind) {
    case 'deploy': return '点击高亮空格部署；大体型以所选格为左上角。';
    case 'move': return '点击圆点落位；移动会消耗一次行动。';
    case 'attack': return '点击红色目标攻击；奶妈的友方目标会获得治疗。';
    case 'hook-target': return '牵引 1 / 2：选择射程内的敌方随从。';
    case 'hook-drop': return '牵引 2 / 2：选择它的新位置，仍须在钩子射程内。';
    case 'sacrifice-pick': return '献祭 1 / 2：选择另一枚友方随从。';
    case 'sacrifice-column': return '献祭 2 / 2：点击高亮列，射击进攻方向上的第一个敌方。';
    case 'wall': return '点击射程内的高亮空格制造临时路障。';
    case 'dash-move': return '突袭 1 / 2：选择最多6格内的落点。';
    case 'dash-attack': return '突袭 2 / 2：选择新位置射程内的敌方目标。';
    case 'reforge': return `重铸：选择两个至少半血的友方（已选${i.targets.length}/2），再次点击可取消。`;
    case 'cast': {
      const c = s.hands[s.active].find(c => c.id === i.cardId);
      return c?.kind === 8 ? '点击“田”字区域的左上格；爆弹会伤到双方。' : '点击一个友方随从施放法术。';
    }
    default: return '先部署手中随从，或选择己方棋子行动；法术可以留到后续回合。';
  }
}
export function intentTone(i:Intent): string {
  if (['attack','sacrifice-column','dash-attack'].includes(i.kind)) return 'attack';
  if (i.kind === 'cast' || i.kind === 'reforge' || i.kind === 'sacrifice-pick') return 'magic';
  return 'move';
}
