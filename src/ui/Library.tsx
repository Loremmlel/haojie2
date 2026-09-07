import { useEffect, useId, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { CATALOG, RULE_NOTES } from '../engine';
import type { Definition } from '../engine';
import { Icon, numberLabel, Rune } from './visuals';
export function Modal({
  title,
  subtitle,
  onClose,
  children,
  wide = false,
}: {
  title: string;
  subtitle?: string;
  onClose: () => void;
  children: ReactNode;
  wide?: boolean;
}) {
  const ref = useRef<HTMLDialogElement>(null),
    id = useId();
  useEffect(() => {
    const dialog = ref.current!;
    if (!dialog.open) dialog.showModal();
    return () => {
      if (dialog.open) dialog.close();
    };
  }, []);
  return (
    <dialog
      className={`modal ${wide ? 'wide' : ''}`}
      ref={ref}
      aria-labelledby={id}
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <header className="modal-header">
        <div>
          <p className="eyebrow">HAOJIE · FIELD MANUAL</p>
          <h2 id={id}>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
        <button className="icon-button" aria-label="关闭弹窗" onClick={onClose}>
          <Icon name="x" />
        </button>
      </header>
      <div className="modal-body">{children}</div>
    </dialog>
  );
}
export function DefinitionStats({ d }: { d: Definition }) {
  if (d.spell !== undefined || d.weapon !== undefined) {
    const limit = d.spell ?? d.weapon!;
    return (
      <div className="spell-duration">
        <Icon name={d.weapon !== undefined ? 'sword' : 'clock'} />
        {d.weapon !== undefined ? '武器 · ' : '法术 · '}
        <strong>{limit < 0 ? '未注明' : limit}</strong>
        {limit < 0 ? '暂不自动过期' : '回合储存'}
      </div>
    );
  }
  const values = [
    ['sword', '攻击', d.id === '3p' ? '40−5n' : d.id === 'u21' ? '±25' : numberLabel(d.attack)],
    ['heart', '生命', d.health],
    ['target', '射程', d.id === '3p' ? 'n' : d.range],
    ['clock', '攻次', numberLabel(d.actions)],
    ['move', '移动', numberLabel(d.move)],
  ];
  return (
    <div className="definition-stats">
      {values.map(([icon, label, value]) => (
        <div key={label}>
          <span>
            <Icon name={String(icon)} size={13} />
            {label}
          </span>
          <b>{value}</b>
        </div>
      ))}
    </div>
  );
}
export function Codex({ onClose }: { onClose: () => void }) {
  const [query, setQuery] = useState(''),
    [filter, setFilter] = useState('全部');
  const entries = CATALOG.filter((d) => {
    const match =
      filter === '全部' ||
      (filter === '普通' && d.tier === 'normal') ||
      (filter === '终极' && d.tier === 'ultimate') ||
      (filter === '法术' && d.spell !== undefined) ||
      (filter === '武器' && d.weapon !== undefined) ||
      (filter === '变体' && d.tier === 'derived');
    return (
      match &&
      `${d.name}${d.id}${d.description}${d.role}`.toLowerCase().includes(query.trim().toLowerCase())
    );
  });
  return (
    <Modal
      title="浩劫图鉴"
      subtitle="26种普通召唤、28种终极召唤，以及变体与合成。第四属性现为选择攻击后的攻击次数。"
      wide
      onClose={onClose}
    >
      <div className="codex-tools">
        <label className="search-field">
          <Icon name="target" />
          <input
            autoFocus
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="搜索名称、能力或编号…"
            aria-label="搜索图鉴"
          />
          <span>{entries.length}</span>
        </label>
        <div className="filter-tabs">
          {['全部', '普通', '终极', '法术', '武器', '变体'].map((f) => (
            <button
              key={f}
              aria-pressed={f === filter}
              className={f === filter ? 'active' : ''}
              onClick={() => setFilter(f)}
            >
              {f}
            </button>
          ))}
        </div>
      </div>
      <div className="codex-grid">
        {entries.map((d) => (
          <article
            key={d.id}
            className={`codex-card ${d.spell !== undefined ? 'is-spell' : ''} ${d.tier === 'ultimate' ? 'ultimate-codex' : ''} ${d.weapon !== undefined ? 'is-weapon' : ''}`}
          >
            <div className="codex-card-heading">
              <Rune kind={d.id} large />
              <div>
                <span className="piece-index">
                  {d.tier === 'normal'
                    ? `普通 ${String(d.id).padStart(2, '0')}`
                    : d.tier === 'ultimate'
                      ? `终极 ${String(d.id).slice(1).padStart(2, '0')}`
                      : '变体 / 合成'}
                </span>
                <h3>{d.name}</h3>
                <span className="role-chip">
                  {d.role}
                  {d.mage ? ' · 法师' : ''}
                  {d.size ? ' · 2×2' : ''}
                </span>
              </div>
            </div>
            <DefinitionStats d={d} />
            <p>{d.description}</p>
            {d.skill && (
              <div className="skill-label">
                <Icon name="spark" size={14} />
                {d.skill}
              </div>
            )}
          </article>
        ))}
      </div>
      {entries.length === 0 && (
        <p className="empty-state">没有匹配条目。可以搜索“冰冻”“人头”“复活”或“法师”。</p>
      )}
      <p className="fine-print">
        图鉴与引擎共用同一份棋子定义。末尾“善铁”配方的输入与判定尚未定义，因此不加入图鉴或随机池。
      </p>
    </Modal>
  );
}
export function Rules({ onClose }: { onClose: () => void }) {
  return (
    <Modal
      title="浩劫 · 规则手册"
      subtitle="以最新《浩劫.docx》及文末回复为准。已确认的条款不再列作旧版猜测。"
      onClose={onClose}
      wide
    >
      <div className="rule-hero">
        <Icon name="crown" size={42} />
        <div>
          <h3>守住300生命，掌握终极召唤。</h3>
          <p>
            9列13行，117格。苍穹基地(5,1)，赤焰基地(5,13)；基地归零判负。每击杀敌方获得1人头，回合开始可花2人头将一次普通召唤改为终极召唤。
          </p>
        </div>
      </div>
      <div className="rule-steps">
        <section>
          <b>01</b>
          <h3>先决定召唤类型，再揭示结果</h3>
          <p>
            每个己方回合开始通常有2次召唤。选择普通或终极，然后查看结果；可由改判小法师重抽刚召唤的牌。所有召唤完成后开始行动。
          </p>
          <p>
            随从必须当回合部署。苍穹默认可部署1–8行，赤焰6–13行；回合开始在默认区域外某行己方数量至少多2，可获得该回合部署权。
          </p>
        </section>
        <section>
          <b>02</b>
          <h3>每回合一种模式，不混用</h3>
          <p>
            普通随从在移动、攻击、技能、蓄力中选择一种。第四属性是选择攻击后可攻击几次：射手可攻击两次，但不能射一次再移动。矿工下一回合、靴子、免费能力与冲锋号令按专属规则例外。
          </p>
          <p>
            分数行为需要主动蓄力。定炮须回合开始已有2层才可开炮；名刀、大肉比、免疫塔等半速随从，先蓄力一回合，后续可消耗1层移动一格。
          </p>
        </section>
        <section>
          <b>03</b>
          <h3>装备、冰冻与叠放</h3>
          <p>
            武器可像法术一样储存并装备到友方。冰冻者不能行动，并对双方都视为中立阻挡；仍保留归属记录用于计时和死亡结算。灼烧、虹吸等持续效果可在战场上查看。
          </p>
          <p>
            克隆军团一次8枚，同格可叠放。单体攻击和敌方技能只命中栈顶，法术命中全部叠放者；整批最后死亡才算一颗人头，不可献祭。点击同格可切换己方克隆进行操作。
          </p>
        </section>
      </div>
      <section className="rules-key">
        <h3>伤害、状态和路径</h3>
        <div className="stat-explainer">
          <span>
            <Icon name="sword" />
            攻击：单次基础伤害
          </span>
          <span>
            <Icon name="clock" />
            攻次：攻击模式次数
          </span>
          <span>
            <Icon name="move" />
            移动：移动操作的距离
          </span>
        </div>
        <p>
          四向距离为|c−a|+|d−b|。普通移动不穿过单位或基地；普通攻击可以越过友方，不能穿过敌方或冰冻中立者，但允许射程内绕路。SZF、小BW冲撞和炎魔之心穿透按专属规则例外。
        </p>
        <p>
          金身免疫伤害及所有敌方技能、法术，包括死吧！与策反。己方献祭不是敌方效果。投石机对满血目标立即引爆一次，此后不满血标记须被另一友方非投石机的攻击命中才引爆。
        </p>
        <p>
          策反先结算伤害，目标存活且未被免疫才换边；本回合疲劳，下一己方回合即可行动。奶妈临终行动、伤害转化、小屋召唤和SZF弹出会明确提示由哪一方处理。
        </p>
      </section>
      <details className="interpretations">
        <summary>
          已确认规则与剩余实施解释<span>展开查看</span>
        </summary>
        <p className="notice">
          “善铁”的输入棋子、合成结果名称和3n判定对象未定义；冲锋号令也未提供储存期限。这些缺口保留标注，不冒充作者确认。
        </p>
        <ol>
          {RULE_NOTES.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ol>
      </details>
      <section className="keyboard-guide">
        <Icon name="help" />
        <p>
          <b>操作和悔棋：</b>
          方向键选择棋盘格，Enter操作；Esc取消选点，Ctrl/⌘+Z悔棋，Ctrl/⌘+Shift+Z重做。悔棋最多60步，连同人头、装备、召唤、连锁反应和随机数一起恢复，不会重新掷骰。旧1.0存档不静默转为2.0。
        </p>
      </section>
    </Modal>
  );
}
