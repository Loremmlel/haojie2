import { definition } from './catalog';
import { attackPath, frontal, targets } from './geometry';
import { activeEffect, addUnit, asTarget, emit, ensure, faction, getStats, random } from './state';
import type { Effect, GameState, Point, Target, Unit } from './types';

export function findTarget(s: GameState, id?: string): Target {
  const t = targets(s).find(t => t.id === id); ensure(t,'请选择有效的目标。'); return t;
}
const effectsOf = (s: GameState,t: Target): Effect[] => t.unit ? t.unit.effects : s.baseEffects[t.owner];
function removeEffect(s: GameState,t: Target,e: Effect): void {
  if (t.unit) t.unit.effects = t.unit.effects.filter(v => v !== e);
  else s.baseEffects[t.owner] = s.baseEffects[t.owner].filter(v => v !== e);
}
const alive = (s: GameState,u: Unit) => s.units.some(v => v.id === u.id);

/** Direct removal is distinct from damage: sacrifice and execution bypass immunity. */
export function kill(s: GameState, victim: Unit, source?: Unit): void {
  if (!alive(s,victim)) return;
  const snapshot = structuredClone(victim);
  victim.hp = 0;
  s.units = s.units.filter(u => u.id !== victim.id);
  emit(s,{ type:'death', to:victim, unitId:victim.id, owner:victim.owner },`${faction(victim.owner)}的${definition(victim.kind).name}离场`);
  const killer = source && s.units.find(u => u.id === source.id);
  if (killer?.kind === 26 && killer.owner !== victim.owner) {
    const step = killer.kills++ % 3;
    if (step === 0) { killer.maxHp += 10; heal(s,killer,10); }
    if (step === 1) killer.attackBonus += 5;
    if (step === 2) killer.rangeBonus++;
    emit(s,{ type:'skill', to:killer, owner:killer.owner, text:'升级' },`杀手完成第${killer.kills}次击杀，获得成长`);
  }
  if (victim.kind === 11) s.bonus[victim.owner]++;
  if (victim.kind === 12) addUnit(s,'grave',victim.owner,victim);
  if (victim.kind === 2) s.pending.push({ kind:'death-shot', owner:victim.owner, source:snapshot, amount:20 });
  if (victim.kind === 20 && source && source.owner !== victim.owner) {
    const target = s.units.find(u => u.id === source.id);
    if (target) damage(s,asTarget(target),20,snapshot);
  }
}

/** Returns actual HP lost, not nominal or overkill damage. */
export function damage(s: GameState, t: Target, amount: number, source?: Unit, path?: Point[]): number {
  amount = Math.max(0,amount);
  if (!t.unit) {
    const loss = Math.min(s.bases[t.owner],amount);
    s.bases[t.owner] -= loss;
    if (loss) emit(s,{ type:'damage', to:t, amount:loss, unitId:t.id, owner:t.owner },`${faction(t.owner)}基地受到${loss}伤害`);
    return loss;
  }
  const u = t.unit;
  if (!alive(s,u) || amount <= 0) return 0;
  if (u.effects.some(e => e.type === 'immune' && activeEffect(s,e))) {
    emit(s,{ type:'shield', to:u, unitId:u.id, owner:u.owner, text:'免疫' },`${definition(u.kind).name}以金身免疫伤害`); return 0;
  }
  if (u.kind === 24 && path && frontal(path,u.owner)) amount = Math.min(10,amount);
  const before = u.hp;
  const protector = !u.guardUsed && before <= amount && s.units.find(v => v.kind === 3 && v.owner === u.owner && attackPath(s,v,t,getStats(s,v).range));
  if (protector) {
    u.guardUsed = true; u.hp = 1;
    emit(s,{ type:'shield', to:u, unitId:u.id, owner:u.owner, text:'名刀' },`${definition(u.kind).name}获得名刀保护，以1血存活`);
  } else u.hp = Math.max(0,u.hp-amount);
  const loss = before-u.hp, snapshot = structuredClone(u);
  if (loss) emit(s,{ type:'damage', to:u, amount:loss, unitId:u.id, owner:u.owner },`${definition(u.kind).name}受到${loss}伤害`);
  if (u.hp <= 0) kill(s,u,source);
  if (loss > 0 && u.kind === 16 && source && source.id !== u.id && source.owner === u.owner) {
    s.pending.push({ kind:'reflect', owner:u.owner, source:snapshot, amount:loss });
  }
  return loss;
}

export function heal(s: GameState, u: Unit, amount: number): void {
  if (!alive(s,u)) return;
  const gained = Math.min(amount,u.maxHp-u.hp); u.hp += gained;
  emit(s,{ type:'heal', to:u, amount:gained, unitId:u.id, owner:u.owner },`${definition(u.kind).name}回复${gained}生命`);
  if (u.hp === u.maxHp) {
    for (const mark of [...u.effects].filter(e => e.type === 'mark' && activeEffect(s,e))) {
      u.effects = u.effects.filter(e => e !== mark);
      damage(s,asTarget(u),5);
    }
  }
}

export function attack(s: GameState, source: Unit, t: Target, unlimited = false): void {
  const friendly = t.owner === source.owner;
  ensure(!friendly || (t.unit && (source.kind === 2 || (t.unit.kind === 16 && t.id !== source.id))), '只能攻击敌方；奶妈可治疗友方，伤害转化器可承受友伤。');
  ensure(source.kind !== 4 || source.charge >= 2,'定炮至少需要两格蓄力才能开炮。');
  ensure(source.kind !== 9 || !source.attacked.includes(t.id),'射手本回合不能重复攻击同一目标。');
  const path = attackPath(s,source,t,unlimited ? 117 : getStats(s,source).range);
  ensure(path,'目标不在射程内，或所有可行攻击路径均被敌方阻挡。');
  source.attacked.push(t.id);
  if (source.kind === 4) { source.charge = 0; source.fired = true; }
  emit(s,{ type:'attack', from:source, to:t, owner:source.owner, unitId:source.id, text:friendly && source.kind === 2 ? '治疗' : '攻击' },`${definition(source.kind).name}${friendly && source.kind === 2 ? '治疗' : '攻击'}${t.unit ? definition(t.unit.kind).name : faction(t.owner)+'基地'}`);
  if (friendly && source.kind === 2 && t.unit) { heal(s,t.unit,20); return; }
  const execute = !friendly && t.unit && source.effects.find(e => e.type === 'execute' && activeEffect(s,e));
  if (execute && t.unit) {
    source.effects = source.effects.filter(e => e !== execute);
    emit(s,{ type:'skill', from:source, to:t, owner:source.owner, text:'处决' });
    kill(s,t.unit,source); return;
  }
  const mark = !friendly && source.kind !== 10 && effectsOf(s,t).find(e => e.type === 'mark' && e.owner === source.owner && activeEffect(s,e));
  if (mark) removeEffect(s,t,mark);
  let loss = 0;
  if (source.kind === 10 && !friendly) {
    const old = effectsOf(s,t).find(e => e.type === 'mark' && e.owner === source.owner);
    if (old) removeEffect(s,t,old);
    const next: Effect = { type:'mark', owner:source.owner, from:s.ply, until:s.ply+2 };
    effectsOf(s,t).push(next);
    emit(s,{ type:'skill', to:t, owner:source.owner, text:'标记' });
    if (t.unit ? t.unit.hp === t.unit.maxHp : s.bases[t.owner] === 300) { removeEffect(s,t,next); loss = damage(s,t,5,source,path); }
  } else {
    let amount = getStats(s,source).attack;
    if (source.kind === 1) {
      const roll = random(s);
      amount += roll < 1/12 ? 60 : roll < 1/3 ? 20 : 0;
      if (roll < 1/3) emit(s,{ type:'skill', to:t, owner:source.owner, text:roll < 1/12 ? '超级暴击' : '暴击' });
    }
    loss = damage(s,t,amount,source,path);
  }
  if (mark) loss += damage(s,t,5,source);
  const conversion = !friendly && t.unit && source.effects.find(e => e.type === 'convert' && activeEffect(s,e));
  if (conversion && loss > 0 && t.unit) {
    source.effects = source.effects.filter(e => e !== conversion);
    if (alive(s,t.unit)) {
      t.unit.owner = source.owner; t.unit.born = s.turns[source.owner]; t.unit.spent = 0;
      t.unit.effects = []; t.unit.attacked = [];
      emit(s,{ type:'skill', to:t.unit, owner:source.owner, text:'策反' },`${definition(t.unit.kind).name}加入${faction(source.owner)}`);
    }
  }
}

export function react(s: GameState, targetId?: string): void {
  const reaction = s.pending.shift(); ensure(reaction,'当前没有等待结算的效果。');
  if (!targetId) { emit(s,{ type:'skill', owner:reaction.owner, text:'放弃' },`${faction(reaction.owner)}放弃了${reaction.kind === 'death-shot' ? '临终行动' : '伤害转化'}`); return; }
  const target = findTarget(s,targetId);
  if (reaction.kind === 'death-shot') attack(s,reaction.source,target,true);
  else {
    ensure(target.owner !== reaction.owner,'伤害转化只能选择敌方目标。');
    const path = attackPath(s,reaction.source,target,getStats(s,reaction.source).range);
    ensure(path,'请选择伤害转化器射程内的敌方目标。');
    emit(s,{ type:'attack', from:reaction.source, to:target, owner:reaction.owner, text:'转化' });
    damage(s,target,reaction.amount,reaction.source,path);
  }
}
