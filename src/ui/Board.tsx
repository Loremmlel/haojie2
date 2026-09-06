import { useEffect, useMemo, useRef, useState } from 'react';
import { ALL_CELLS, basePoint, cells, definition, distance, faction, getStats, targetAt } from '../engine';
import type { GameEvent, GameState, Player, Point } from '../engine';
import { canChoose, intentTone } from './intents';
import type { Intent } from './intents';
import { Icon, playerStyle } from './visuals';

function Effects({events,reduced}:{events:GameEvent[];reduced:boolean}) {
  const xy=(p:Point)=>({x:(p.x-.5)*100,y:(p.y-.5)*100});
  return <svg className="effects-layer" viewBox="0 0 900 1300" aria-hidden="true">
    {events.filter(e=>e.to).map(e=>{
      const to=xy(e.to!),from=e.from ? xy(e.from) : to;
      const route=(e as GameEvent & {path?:Point[]}).path?.map(xy) ?? [from,to];
      const path=route.map((p,i)=>`${i?'L':'M'}${p.x} ${p.y}`).join(' ');
      return <g key={e.id} className={`fx fx-${e.type} p${e.owner ?? 1}`}>
        {e.type==='attack' && <><path d={path} className="shot-trail"/><circle r="8" fill="currentColor" className="shot-core">{!reduced && <animateMotion path={path} dur="0.42s" fill="freeze"/>}</circle><circle cx={to.x} cy={to.y} r="23" className="impact-ring"/></>}
        {e.type==='move' && <path d={path} className="move-trail"/>}
        {['spawn','shield','skill','death'].includes(e.type) && <g transform={`translate(${to.x} ${to.y})`}>
          <circle r={e.text==='爆弹'?92:e.type==='death'?40:35} className={`aura-ring ${e.text==='爆弹'?'blast':''}`}/>
          {[0,1,2,3,4,5,6,7].map(n=><path key={n} d="M 0 -32 L 0 -52" transform={`rotate(${n*45})`} className="spark-ray"/>)}
          {e.text && e.type!=='spawn' && <text y="-40" className="effect-label">{e.text}</text>}
        </g>}
        {(e.type==='damage'||e.type==='heal') && <g transform={`translate(${to.x} ${to.y-24})`}><text className={`damage-number ${e.type==='heal'?'healing':''}`}>{e.type==='heal'?'+':'−'}{e.amount}</text></g>}
      </g>;
    })}
  </svg>;
}
export function Board({state:s,intent,selectedId,onCell,events}:{state:GameState;intent:Intent;selectedId:string|null;onCell:(p:Point)=>void;events:GameEvent[]}) {
  const ref=useRef<HTMLDivElement>(null),[hover,setHover]=useState<Point|null>(null),[focus,setFocus]=useState<Point>({x:5,y:7}),[reduced,setReduced]=useState(false);
  useEffect(()=>{const media=window.matchMedia('(prefers-reduced-motion: reduce)');const change=()=>setReduced(media.matches);change();media.addEventListener('change',change);return()=>media.removeEventListener('change',change);},[]);
  const choices=useMemo(()=>new Set(ALL_CELLS.filter(p=>canChoose(s,intent,p)).map(p=>`${p.x},${p.y}`)),[s,intent]);
  const owner=s.pending[0]?.owner ?? s.active;
  const acting='unitId' in intent ? s.units.find(u=>u.id===intent.unitId) : undefined;
  const previewKind='cardId' in intent ? s.hands[s.active].find(c=>c.id===intent.cardId)?.kind : undefined;
  const previewSize=intent.kind==='deploy' && previewKind ? definition(previewKind).size ?? 1 : intent.kind==='cast' && previewKind===8 ? 2 : 0;
  const preview=hover && previewSize && choices.has(`${hover.x},${hover.y}`) ? hover : null;
  const tone=s.pending.length?'attack':intentTone(intent);
  function navigate(event:React.KeyboardEvent<HTMLButtonElement>,p:Point) {
    const delta:Record<string,Point>={ArrowUp:{x:0,y:-1},ArrowDown:{x:0,y:1},ArrowLeft:{x:-1,y:0},ArrowRight:{x:1,y:0}};
    const d=delta[event.key];if(!d)return;event.preventDefault();
    const next={x:Math.max(1,Math.min(9,p.x+d.x)),y:Math.max(1,Math.min(13,p.y+d.y))};
    ref.current?.querySelector<HTMLButtonElement>(`[data-cell="${next.x},${next.y}"]`)?.focus();
  }
  return <section className="board-shell" aria-label="对战棋盘" style={playerStyle(owner)}>
    <div className="board-topline"><span><i className="live-dot"/> {faction(owner)}视野</span><span>9 × 13 <b> / </b> 117格</span></div>
    <div className="board-frame">
      <div className="column-labels" aria-hidden="true">{Array.from({length:9},(_,i)=><span key={i}>{i+1}</span>)}</div>
      <div className="row-labels" aria-hidden="true">{Array.from({length:13},(_,i)=><span key={i}>{String(i+1).padStart(2,'0')}</span>)}</div>
      <div className={`board ${intent.kind!=='none'||s.pending.length?'targeting':''}`} ref={ref} role="grid" aria-label="九列十三行战场，方向键选择格子，回车操作" onMouseLeave={()=>setHover(null)}>
        {Array.from({length:13},(_,row)=><div role="row" className="board-row" key={row}>{Array.from({length:9},(_,col)=>{
          const p={x:col+1,y:row+1},t=targetAt(s,p),valid=choices.has(`${p.x},${p.y}`);
          const selected=t?.id===selectedId;
          const range=acting && intent.kind==='attack' && cells(acting).some(c=>distance(c,p)<=getStats(s,acting).range);
          const inPreview=preview && p.x>=preview.x && p.x<preview.x+previewSize && p.y>=preview.y && p.y<preview.y+previewSize;
          const picked=intent.kind==='reforge' && !!t && intent.targets.includes(t.id);
          const area=row<5?'north':row>7?'south':'contested';
          const label=`${p.x}列${p.y}行${t ? `，${faction(t.owner)}${t.unit ? definition(t.unit.kind).name+'，'+t.unit.hp+'生命' : '基地，'+s.bases[t.owner]+'生命'}` : '，空格'}${valid?'，可选择':''}`;
          return <button key={col} role="gridcell" data-cell={`${p.x},${p.y}`} aria-label={label} aria-selected={selected||picked} tabIndex={focus.x===p.x&&focus.y===p.y?0:-1} className={`cell ${area} ${range?'in-range':''} ${valid?`legal ${tone} ${t?'occupied-target':''}`:''} ${selected?'selected-cell':''} ${inPreview?'area-preview':''} ${picked?'picked-cell':''}`} onClick={()=>onCell(p)} onFocus={()=>{setFocus(p);setHover(p);}} onMouseEnter={()=>setHover(p)} onKeyDown={e=>navigate(e,p)}><span className="cell-dot"/>{valid&&!t&&<span className="target-dot"/>}</button>;
        })}</div>)}
        <div className="frontier frontier-one" aria-hidden="true"><span>交 锋 区</span></div>
        <div className="frontier frontier-two" aria-hidden="true"/>
        {([1,2] as Player[]).map(p=>{const at=basePoint(p);return <div key={p} className={`base-token p${p}`} style={{left:`${(at.x-1)/9*100}%`,top:`${(at.y-1)/13*100}%`}} aria-hidden="true"><Icon name="crown" size={25}/><b>{s.bases[p]}</b></div>;})}
        {s.units.map(u=>{
          const d=definition(u.kind),size=d.size??1,stats=getStats(s,u),selected=u.id===selectedId;
          const hurt=events.some(e=>e.type==='damage'&&e.unitId===u.id),spawned=events.some(e=>e.type==='spawn'&&e.unitId===u.id);
          return <div key={u.id} className={`piece-wrap p${u.owner} ${size===2?'large-piece':''} ${selected?'piece-selected':''} ${hurt?'hurt':''} ${spawned?'arriving':''}`} style={{left:`${(u.x-1)/9*100}%`,top:`${(u.y-1)/13*100}%`,width:`${size/9*100}%`,height:`${size/13*100}%`}} aria-hidden="true">
            <div className={`piece ${stats.sleeping?'sleeping':''}`}><span className="piece-heading"/><span className="piece-code">{u.kind==='3p'?'3′':typeof u.kind==='number'?u.kind:'◆'}</span><strong>{d.glyph}</strong><span className="piece-hp">{u.hp}</span><div className="piece-health"><i style={{width:`${u.hp/u.maxHp*100}%`}}/></div>
            {u.effects.some(e=>e.type==='immune'&&e.from<=s.ply&&e.until>s.ply)&&<span className="shield-halo"/>}
            {u.effects.some(e=>e.type==='mark')&&<span className="mark-dot">·</span>}
            </div>
            <div className="action-pips">{Array.from({length:Math.min(6,stats.actions)},(_,i)=><i key={i} className={i<stats.remaining?'available':''}/>)}</div>
            {stats.sleeping&&<span className="sleep-badge">休</span>}
          </div>;
        })}
        {intent.kind==='dash-attack'&&<div className="dash-ghost" style={{left:`${(intent.to.x-1)/9*100}%`,top:`${(intent.to.y-1)/13*100}%`}} aria-hidden="true"><Icon name="move"/></div>}
        <Effects events={events} reduced={reduced}/>
      </div>
    </div>
    <div className="board-legend"><span><i className="legend-dot move"/>可移动 / 部署</span><span><i className="legend-dot attack"/>可攻击目标</span><span className="hover-coordinate">{hover?`(${hover.x}, ${hover.y})`:'方向键也可选格'}</span></div>
  </section>;
}
