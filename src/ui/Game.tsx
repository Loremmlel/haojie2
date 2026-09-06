import { useEffect, useMemo, useRef, useState } from 'react';
import { commandError, createDemoGame, createGame, createSession, definition, dispatch, faction, getStats, isLegal, parseSession, redo, targetAt, undo } from '../engine';
import type { Command, GameEvent, GameState, Player, Point, Session, Unit } from '../engine';
import { Board } from './Board';
import { canChoose, commandFor, instruction, nextIntent } from './intents';
import type { Intent } from './intents';
import { Codex, DefinitionStats, Modal, Rules } from './Library';
import { Icon, numberLabel, playerStyle, Rune } from './visuals';
import { Soundscape } from './sound';
import './styles.css';

export interface HaojieGameProps {
  initialState?: GameState;
  /** Set null when the host owns persistence. Use unique keys for multiple embeds. */
  storageKey?: string | null;
  onStateChange?: (state:GameState)=>void;
}
function freshSeed():number {
  if(typeof crypto!=='undefined'&&crypto.getRandomValues)return crypto.getRandomValues(new Uint32Array(1))[0]||1;
  return Date.now()>>>0||1;
}
function bootstrap(initialState:GameState|undefined,storageKey:string|null) {
  if(initialState)return {session:createSession(initialState),notice:''};
  if(storageKey && typeof window!=='undefined') {
    try {const saved=window.localStorage.getItem(storageKey);if(saved)return {session:parseSession(saved),notice:'已恢复上次对局。'};}
    catch {return {session:createSession(createGame(freshSeed())),notice:'未能读取旧存档，已创建新对局。你仍可通过“载入”导入之前导出的存档。'};}
  }
  return {session:createSession(createGame(freshSeed())),notice:''};
}
const effectNames:Record<string,string>={attack:'攻击+10',immune:'金身',execute:'死吧！',convert:'策反',mark:'投石标记'};

export function HaojieGame({initialState,storageKey='haojie2.session.v1',onStateChange}:HaojieGameProps) {
  const [boot]=useState(()=>bootstrap(initialState,storageKey));
  const [session,setSession]=useState<Session>(boot.session),sessionRef=useRef(session);
  const [intent,setIntent]=useState<Intent>({kind:'none'}),[selectedId,setSelectedId]=useState<string|null>(null),[cardId,setCardId]=useState<string|null>(null);
  const [modal,setModal]=useState<'codex'|'rules'|'new'|'log'|null>(null),[notice,setNotice]=useState(boot.notice),[events,setEvents]=useState<GameEvent[]>([]);
  const [sound,setSound]=useState(false),soundscape=useRef(new Soundscape()),fileInput=useRef<HTMLInputElement>(null),[seedText,setSeedText]=useState('');
  const callback=useRef(onStateChange);callback.current=onStateChange;
  const s=session.present,controller=s.pending[0]?.owner??s.active,hand=s.hands[s.active];
  const unit=s.units.find(u=>u.id===selectedId),card=hand.find(c=>c.id===cardId);
  const reaction=s.pending[0],inspected=reaction?.source??unit;
  const d=card?definition(card.kind):inspected?definition(inspected.kind):undefined;
  const stats=inspected?getStats(s,inspected):undefined;
  const isOwn=!!unit&&unit.owner===s.active&&!reaction&&!s.winner;
  const remainingMinions=hand.filter(c=>!definition(c.kind).spell).length;
  const endError=useMemo(()=>commandError(s,{type:'end'}),[s]);
  const canAct=isOwn&&!!stats&&stats.remaining>0;

  useEffect(()=>{
    if(storageKey) {
      try {window.localStorage.setItem(storageKey,JSON.stringify(session));}
      catch {try {window.localStorage.setItem(storageKey,JSON.stringify(createSession(session.present)));setNotice('存储空间不足，只保存了当前局面。仍可导出完整存档。');}catch{setNotice('浏览器不允许本地保存；请用“导出”保存这盘棋。');}}
    }
    callback.current?.(session.present);
  },[session,storageKey]);
  useEffect(()=>{if(!events.length)return;const timer=window.setTimeout(()=>setEvents([]),1400);return()=>window.clearTimeout(timer);},[events]);
  useEffect(()=>{if(!notice)return;const timer=window.setTimeout(()=>setNotice(''),6500);return()=>window.clearTimeout(timer);},[notice]);
  useEffect(()=>()=>soundscape.current.dispose(),[]);
  function replace(next:Session) {sessionRef.current=next;setSession(next);setIntent({kind:'none'});setCardId(null);setEvents([]);}
  function rewind(forward=false) {const current=sessionRef.current;const next=forward?redo(current):undo(current);if(next===current)return;replace(next);setNotice(forward?'已重做，随机结果保持不变。':'已悔棋，手牌、计时与随机数一并恢复。');}
  useEffect(()=>{
    const handler=(e:KeyboardEvent)=>{
      if(modal || (e.target instanceof HTMLElement && e.target.closest('input,textarea,select,[contenteditable="true"]')))return;
      if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();rewind(e.shiftKey);}
      if(e.key==='Escape'){setIntent({kind:'none'});setCardId(null);}
    };
    window.addEventListener('keydown',handler);return()=>window.removeEventListener('keydown',handler);
  },[modal]);
  function run(command:Command) {
    try {
      const previous=sessionRef.current,next=dispatch(previous,command);sessionRef.current=next;setSession(next);
      setIntent({kind:'none'});setCardId(null);setNotice('');
      setEvents(old=>[...old,...next.present.events].slice(-65));soundscape.current.play(next.present.events,sound);
      if(command.type==='end')setSelectedId(null);
      if(command.type==='deploy')setSelectedId(next.present.events.find(e=>e.type==='spawn')?.unitId??null);
    } catch(error) {setNotice(error instanceof Error?error.message:'操作未完成，请重试。');}
  }
  function onCell(p:Point) {
    const current=sessionRef.current.present;
    if(intent.kind==='none'&&!current.pending.length){setSelectedId(targetAt(current,p)?.id??null);setCardId(null);return;}
    if(!canChoose(current,intent,p)) {
      const command=commandFor(current,intent,p);
      setNotice(command?commandError(current,command)??'请选择高亮位置。':'请选择高亮的有效目标。');return;
    }
    const command=commandFor(current,intent,p);
    if(command){run(command);return;}
    const next=nextIntent(current,intent,p);
    if(next.kind==='reforge'&&next.targets.length===2)run({type:'cast',cardId:next.cardId,mode:'double',sacrificeIds:next.targets});
    else setIntent(next);
  }
  function chooseCard(id:string) {
    if(s.pending.length||s.winner)return;
    const c=hand.find(c=>c.id===id);if(!c)return;
    setCardId(id);setSelectedId(null);setNotice('');
    setIntent(definition(c.kind).spell?(c.kind===25?{kind:'none'}:{kind:'cast',cardId:id}):{kind:'deploy',cardId:id,charge:false});
  }
  function start(demo=false) {
    try {
      const seed=seedText.trim()?Number(seedText):freshSeed();
      const next=demo?createDemoGame():createGame(seed);replace(createSession(next));setSelectedId(null);setModal(null);setSeedText('');
      setNotice(demo?'演示棋局已载入：选中杀手，攻击超级跑得快，再试试悔棋。':'新对局开始。每人基地300生命，苍穹方先手。');
    } catch(error){setNotice(error instanceof Error?error.message:'无法开始对局。');}
  }
  function exportSave() {
    const blob=new Blob([JSON.stringify(sessionRef.current,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');
    a.href=url;a.download=`haojie2-round-${s.ply}.json`;a.click();window.setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  async function importSave(file?:File) {
    if(!file)return;
    try{if(file.size>12_000_000)throw new Error('存档大于12MB，无法载入。');const next=parseSession(await file.text());replace(next);setSelectedId(null);setNotice('存档已载入，包含悔棋与重做历史。');}
    catch(error){setNotice(error instanceof Error?error.message:'存档读取失败。');}
    if(fileInput.current)fileInput.current.value='';
  }
  function actionMode(kind:'move'|'attack'|'hook-target'|'sacrifice-pick'|'wall'|'dash-move') {if(unit){setIntent({kind,unitId:unit.id});setCardId(null);setNotice('');}}
  const directSkill=(mode?:'attack'|'range'|'charge')=>unit&&run({type:'skill',unitId:unit.id,mode});
  const selectedBase=selectedId==='base-1'?1:selectedId==='base-2'?2:null;

  return <div className="hj-game" style={playerStyle(controller)}>
    <header className="site-header"><div className="brand"><span className="brand-mark"><Icon name="sword" size={23}/></span><div><h1>豪杰棋局<span>II</span></h1><p>HAOJIE · TACTICS</p></div></div><nav aria-label="游戏工具"><button onClick={()=>setModal('codex')}><Icon name="book"/><span>棋子图鉴</span></button><button onClick={()=>setModal('rules')}><Icon name="help"/><span>规则</span></button><button className="icon-button sound-toggle" aria-label={sound?'关闭音效':'开启音效'} aria-pressed={sound} title={sound?'音效已开启':'音效默认关闭'} onClick={()=>setSound(v=>!v)}><Icon name={sound?'volume':'mute'}/></button><span className="nav-divider"/><button className="new-game-button" onClick={()=>setModal('new')}><Icon name="plus"/><span>新对局</span></button></nav></header>
    <main className="game-container">
      <div className="match-heading"><p className="eyebrow"><span/> LOCAL TWO-PLAYER DUEL</p><p>同屏对弈 <i/> 每一步，都有可能。</p></div>
      <section className="scoreboard" aria-label="双方基地与回合状态">
        {([1,2] as Player[]).map(p=><div key={p} className={`player-score p${p} ${controller===p?'current-player':''}`} style={playerStyle(p)}><div className="player-seal"><Icon name="crown" size={29}/></div><div className="player-score-main"><div className="player-name"><h2>{faction(p)}</h2><span>PLAYER 0{p}</span>{controller===p&&<b>{reaction?'结算中':'你的回合'}</b>}</div><div className="base-readout"><strong>{s.bases[p]}</strong><span>/ 300</span><small>{s.units.filter(u=>u.owner===p).length} 枚在场</small></div><div className="base-health"><i style={{width:`${s.bases[p]/300*100}%`}}/></div></div></div>)}
        <div className="round-medallion"><span>ROUND</span><strong>{String(Math.ceil(s.ply/2)).padStart(2,'0')}</strong><i>{s.winner?'终局':reaction?'效果结算':remainingMinions?'部署 / 行动':'行动阶段'}</i></div>
      </section>
      {s.winner&&<section className="victory-banner" role="alert"><Icon name="crown" size={36}/><div><h2>{s.winner==='draw'?'势均力敌，平局。':`${faction(s.winner)}，旗开得胜！`}</h2><p>基地战结束于第{Math.ceil(s.ply/2)}轮。也可以悔棋，重新推演这一招。</p></div><button className="primary" onClick={()=>setModal('new')}>再来一局</button></section>}
      <div className="play-layout">
        <aside className="inspection-rail">
          <section className="panel inspector"><div className="panel-heading"><h2>{reaction?'等待效果结算':card?'手牌详情':'棋子情报'}</h2><span>{reaction?'REACTION':card?'CARD':'INSPECT'}</span></div>
            {d?<><div className="inspector-profile"><Rune kind={d.id} owner={card?s.active:inspected!.owner} large/><span className="role-chip">{d.role}{card?' · 手牌':` · ${faction(inspected!.owner)}`}</span><h3>{d.name}</h3></div>
              {inspected&&!card&&stats?<div className="live-stats">{[['sword','攻击',stats.attack],['heart','生命',`${inspected.hp}/${inspected.maxHp}`],['target','射程',stats.range],['clock','行动',`${stats.remaining}/${stats.actions}`],['move','移动',numberLabel(stats.move)]].map(([icon,label,value])=><div key={label}><span><Icon name={String(icon)} size={14}/>{label}</span><b>{value}</b></div>)}</div>:<DefinitionStats d={d}/>}
              <p className="ability-copy">{d.description}</p>
              {inspected&&!card&&<><div className="status-tags">{stats?.sleeping&&<span>部署疲劳 / 休整中</span>}{inspected.guardUsed&&<span>名刀保护已用</span>}{inspected.effects.map((e,i)=><span key={i} className={e.from>s.ply?'upcoming':''}>{e.from>s.ply?'下回合 · ':''}{effectNames[e.type]}</span>)}{inspected.kind===26&&<span>已击杀 {inspected.kills}</span>}{inspected.kind===15&&<span>已强化 {inspected.upgrades}/3</span>}</div>{[4,5,21].includes(Number(inspected.kind))&&<div className="charge-meter"><span>{inspected.kind===5?'蓄步':'蓄力'} <b>{inspected.charge}/{inspected.kind===4?5:inspected.kind===21?2:1}</b></span><div>{Array.from({length:inspected.kind===4?5:inspected.kind===21?2:1},(_,i)=><i className={i<inspected.charge?'filled':''} key={i}/>)}</div></div>}</>}
              {card?.kind===1&&intent.kind==='deploy'&&<label className="charge-choice"><input type="checkbox" checked={intent.charge} onChange={e=>setIntent({...intent,charge:e.target.checked})}/><span><b>以10生命换取冲锋</b><small>部署后即可行动 · 最大生命变为40</small></span></label>}
              {card?.kind===25&&<div className="reforge-choices"><button className="primary" onClick={()=>run({type:'cast',cardId:card.id,mode:'single'})}><Icon name="spark"/>直接召唤一次</button><button className="secondary" onClick={()=>setIntent({kind:'reforge',cardId:card.id,targets:[]})}>献祭两枚，召唤两次</button></div>}
            </>:selectedBase?<div className="base-inspector"><Icon name="crown" size={52}/><h3>{faction(selectedBase)}基地</h3><strong>{s.bases[selectedBase]} <small>/ 300 HP</small></strong><p>这是你必须守护的核心。基地不能被治疗、移动、献祭或策反。</p></div>:<div className="inspector-empty"><div className="empty-orbit"><Icon name="target" size={44}/></div><h3>选择你的下一步</h3><p>点击一枚棋子查看能力，或从手牌中选择本回合的豪杰。</p><button className="text-button" onClick={()=>setModal('rules')}>初次对弈？了解规则 <Icon name="arrow" size={15}/></button></div>}
            {unit&&!card&&!reaction&&<div className="unit-actions"><h4>{unit.owner!==s.active?'敌方棋子 · 仅可查看':stats?.sleeping?'本回合休整':'选择行动'}</h4><div className="action-buttons"><button aria-pressed={intent.kind==='move'} disabled={!canAct||!stats?.move} onClick={()=>actionMode('move')}><Icon name="move"/>移动</button><button aria-pressed={intent.kind==='attack'} disabled={!canAct||(unit.kind===4&&unit.charge<2)} onClick={()=>actionMode('attack')}><Icon name="sword"/>{unit.kind===4?'开炮':unit.kind===2?'攻击 / 治疗':'攻击'}</button></div>
              {isOwn&&<div className="skill-buttons">{unit.kind===5&&<button disabled={!isLegal(s,{type:'skill',unitId:unit.id})} onClick={()=>directSkill()}><Icon name="spark"/>蓄步</button>}{unit.kind===6&&<button disabled={!canAct} onClick={()=>directSkill()}><Icon name="spark"/>鼓舞友军</button>}{unit.kind===7&&<button disabled={!canAct} onClick={()=>actionMode('hook-target')}><Icon name="spark"/>牵引敌方</button>}{unit.kind===14&&<button disabled={!canAct||unit.maxHp<=10} onClick={()=>actionMode('sacrifice-pick')}><Icon name="spark"/>献祭射击</button>}{unit.kind===15&&<><button disabled={!canAct||unit.upgrades>=3} onClick={()=>directSkill('attack')}>强化 · +10攻击</button><button disabled={!canAct||unit.upgrades>=3} onClick={()=>directSkill('range')}>强化 · +1射程</button></>}{unit.kind===19&&<button disabled={!canAct} onClick={()=>actionMode('wall')}><Icon name="spark"/>制造路障</button>}{unit.kind===21&&<><button disabled={!isLegal(s,{type:'skill',unitId:unit.id,mode:'charge'})} onClick={()=>directSkill('charge')}>蓄势 {unit.charge}/2</button><button disabled={!canAct||unit.charge<2||unit.lastCharge>=s.turns[unit.owner]||unit.hp<=10} onClick={()=>actionMode('dash-move')}><Icon name="move"/>神行突袭</button></>}</div>}
            </div>}
          </section>
          <section className="field-note"><span>TACTICAL NOTE</span><h3>谋定而后动。</h3><p>每一个亮点，是一次选择。移动、攻击与技能共享行动次数。</p><div><kbd>Esc</kbd> 取消选点 <kbd>⌘ / Ctrl Z</kbd> 悔棋</div></section>
        </aside>
        <section className="battle-column"><div className={`instruction-bar ${reaction?'reaction-bar':''}`} role="status"><span className="instruction-icon"><Icon name={reaction?'spark':intent.kind==='attack'?'sword':'target'}/></span><p>{instruction(s,intent)}</p>{intent.kind!=='none'&&!reaction&&<button className="icon-button" aria-label="取消当前操作" onClick={()=>{setIntent({kind:'none'});setCardId(null);}}><Icon name="x" size={16}/></button>}</div>
          <Board state={s} intent={intent} selectedId={selectedId} onCell={onCell} events={events}/>
          <div className="command-bar"><div className="history-actions"><button disabled={!session.past.length} onClick={()=>rewind()} title="Ctrl/⌘+Z" aria-label="悔棋"><Icon name="undo"/><span>悔棋</span></button><button disabled={!session.future.length} onClick={()=>rewind(true)} title="Ctrl/⌘+Shift+Z" aria-label="重做"><Icon name="redo"/></button></div>{reaction?<button className="secondary finish-button" onClick={()=>run({type:'react'})}>放弃此效果 <Icon name="arrow"/></button>:<button className="primary finish-button" onClick={()=>run({type:'end'})} disabled={!!endError} title={endError??'将回合交给对手'}>结束回合 <Icon name="arrow"/></button>}</div>
          <p className="turn-hint">{reaction?`现在由${faction(reaction.owner)}处理效果，原回合不会因此结束。`:s.winner?'对局已结束。':remainingMinions?`还有 ${remainingMinions} 枚随从需要在本回合部署。`:`可继续行动，或结束回合交给${faction(s.active===1?2:1)}。`}</p>
        </section>
        <aside className="hand-rail"><section className="panel hand-panel"><div className="panel-heading"><h2>本回合手牌 <b>{hand.length}</b></h2><Icon name="spark" size={17}/></div><p className="hand-intro"><span className={`tiny-side p${s.active}`}/>{faction(s.active)}<span>随从即刻部署 · 法术可储存</span></p><div className="hand-cards">{hand.map(c=>{const def=definition(c.kind),left=c.expiresAt!==undefined?c.expiresAt-s.turns[s.active]:null;return <button key={c.id} className={`hand-card ${def.spell?'spell-card':''} ${c.id===cardId?'chosen':''}`} aria-pressed={c.id===cardId} aria-label={`选择${def.name}${def.spell?'法术':'随从'}`} disabled={!!reaction||!!s.winner} onClick={()=>chooseCard(c.id)}><Rune kind={c.kind} owner={s.active}/><div className="hand-card-main"><div><h3>{def.name}</h3><span>{def.spell?'法术':def.role}</span></div>{def.spell?<p className={left===1?'expires-soon':''}><Icon name="clock" size={12}/>{left===1?'本回合到期':`剩余${left}回合`}</p>:<p><span><Icon name="sword" size={12}/>{c.kind==='3p'?'动态':def.attack}</span><span><Icon name="heart" size={12}/>{def.health}</span><span><Icon name="target" size={12}/>{c.kind==='3p'?'n':def.range}</span></p>}</div>{!def.spell&&<span className="deploy-label">待部署</span>}</button>;})}{!hand.length&&<div className="empty-hand"><Icon name="check" size={30}/><p>手牌已全部处理</p><small>选择棋子继续行动，或结束回合。</small></div>}</div></section>
          <section className="panel battle-log"><div className="panel-heading"><h2>战场纪事</h2><button className="text-button" onClick={()=>setModal('log')}>全部 <Icon name="arrow" size={14}/></button></div><ol>{s.log.slice(-5).reverse().map((line,i)=><li key={`${s.ply}-${i}-${line}`}><i/><p>{line.substring(line.indexOf('·')+1).trim()}</p></li>)}</ol><div className="log-footer"><span className="live-dot"/>局面已同步至本机</div></section>
          <div className="save-tools"><button onClick={exportSave}><Icon name="download" size={15}/>导出存档</button><button onClick={()=>fileInput.current?.click()}><Icon name="upload" size={15}/>载入存档</button><input type="file" accept=".json,application/json" ref={fileInput} hidden onChange={e=>void importSave(e.target.files?.[0])}/></div>
        </aside>
      </div>
      <footer className="game-footer"><span><i className="live-dot"/>离线可玩 · 同屏双人 · 自动保存</span><button onClick={()=>setModal('rules')}>规则解释版 1.0</button><span>SEED {s.seed}</span></footer>
    </main>
    {notice&&<div className="toast" role="status"><Icon name="help" size={18}/><p>{notice}</p><button className="icon-button" aria-label="关闭提示" onClick={()=>setNotice('')}><Icon name="x" size={16}/></button></div>}
    {modal==='codex'&&<Codex onClose={()=>setModal(null)}/>}{modal==='rules'&&<Rules onClose={()=>setModal(null)}/>}
    {modal==='log'&&<Modal title="战场纪事" subtitle="按发生顺序记录最近160条事件；悔棋后记录随局面一起恢复。" onClose={()=>setModal(null)}><ol className="full-log">{s.log.map((line,i)=><li key={i}>{line}</li>)}</ol></Modal>}
    {modal==='new'&&<Modal title="再起一局" subtitle="当前对局会被替换。重要对局可先关闭此窗口并导出存档。" onClose={()=>setModal(null)}><div className="new-match-art"><Icon name="sword" size={46}/><span>9 × 13 · 26种召唤 · 无限可能</span></div><label className="seed-field">对局种子 <span>可选，留空随机</span><input value={seedText} onChange={e=>setSeedText(e.target.value)} inputMode="numeric" placeholder="例如 20260906"/></label><button className="primary full-width" onClick={()=>start(false)}>开始正式对局 <Icon name="arrow"/></button><button className="secondary full-width" onClick={()=>start(true)}>载入演示棋局</button><p className="fine-print">正式对局从空棋盘开始。演示局预设双方阵型和法术，适合熟悉攻击、技能与悔棋操作。</p></Modal>}
  </div>;
}
