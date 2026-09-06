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
  if (d.spell)
    return (
      <div className="spell-duration">
        <Icon name="clock" />
        储存上限 <strong>{d.spell}</strong> 个己方回合
      </div>
    );
  const values = [
    ['sword', '攻击', d.id === '3p' ? '40−5n' : numberLabel(d.attack)],
    ['heart', '生命', d.health],
    ['target', '射程', d.id === '3p' ? 'n' : numberLabel(d.range)],
    ['clock', '行动', numberLabel(d.actions)],
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
    const category = typeof d.id !== 'number' ? '变体与衍生物' : d.spell ? '法术' : '随从';
    return (
      (filter === '全部' || filter === category) &&
      `${d.name}${d.role}${d.description}${d.id}`
        .toLocaleLowerCase()
        .includes(query.toLocaleLowerCase().trim())
    );
  });
  return (
    <Modal
      title="棋子图鉴"
      subtitle="26张召唤牌，3种变体与衍生物。了解每一位豪杰的独门本领。"
      onClose={onClose}
      wide
    >
      <div className="codex-tools">
        <label className="search-field">
          <Icon name="target" />
          <input
            autoFocus
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="搜索名称、定位或能力…"
            aria-label="搜索图鉴"
          />
          <span>{entries.length}</span>
        </label>
        <div className="filter-tabs">
          {['全部', '随从', '法术', '变体与衍生物'].map((f) => (
            <button
              key={f}
              className={f === filter ? 'active' : ''}
              aria-pressed={f === filter}
              onClick={() => setFilter(f)}
            >
              {f}
            </button>
          ))}
        </div>
      </div>
      <div className="codex-grid">
        {entries.map((d) => (
          <article className={`codex-card ${d.spell ? 'is-spell' : ''}`} key={d.id}>
            <div className="codex-card-heading">
              <Rune kind={d.id} large />
              <div>
                <span className="piece-index">
                  {typeof d.id === 'number'
                    ? `No. ${String(d.id).padStart(2, '0')}`
                    : d.id === '3p'
                      ? 'No. 03′'
                      : '衍生单位'}
                </span>
                <h3>{d.name}</h3>
                <span className="role-chip">
                  {d.role}
                  {d.size ? ' · 2×2占位' : ''}
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
      {!entries.length && <p className="empty-state">没有匹配的棋子，试试“治疗”“蓄力”或“法术”。</p>}
      <p className="fine-print">
        图鉴中的基础数据与规则引擎共用同一份定义。残影的攻击和射程在战场中实时计算。
      </p>
    </Modal>
  );
}
export function Rules({ onClose }: { onClose: () => void }) {
  return (
    <Modal
      title="落子之前"
      subtitle="同屏双人轮流操作。没有倒计时，每一步都可以想清楚。"
      onClose={onClose}
      wide
    >
      <div className="rule-hero">
        <Icon name="crown" size={42} />
        <div>
          <h3>攻破对方基地，成为最后的豪杰。</h3>
          <p>
            棋盘9列×13行；苍穹基地在(5,1)，赤焰基地在(5,13)。双方基地初始各300生命，降至0即告负。
          </p>
        </div>
      </div>
      <div className="rule-steps">
        <section>
          <b>01</b>
          <h3>召唤与部署</h3>
          <p>
            每个己方回合开始，随机召唤两张牌。随从必须当回合部署；法术按图鉴期限储存。可以交替进行部署、行动与施法。
          </p>
          <p>
            苍穹默认可部署第1–8行，赤焰默认第6–13行。回合开始时，在默认区域外某一行，己方随从数量至少比敌方多2，即取得该回合该行的部署权。
          </p>
        </section>
        <section>
          <b>02</b>
          <h3>移动、攻击、技能</h3>
          <p>
            新部署随从通常下个己方回合才能行动；冲锋例外。每一点行动可选择移动、攻击或主动技能一次。选中棋子后，棋盘会标出合法目标。
          </p>
          <p>
            距离按四向格数计算。例如(2,3)到(5,7)为7格。移动不能穿过任何单位；攻击可越过友方，敌方则会阻挡。只要存在射程内合法绕行路径，就能命中。
          </p>
        </section>
        <section>
          <b>03</b>
          <h3>结算与悔棋</h3>
          <p>
            奶妈的临终行动、伤害转化等会暂停普通操作，提示效果所属玩家选择目标。完成后回到原来的玩家继续行动。
          </p>
          <p>
            部署完成后结束回合，交给对手。悔棋与重做会完整恢复随机数和回合状态，不会重新抽卡。支持跨回合撤销；新操作会清空重做分支。
          </p>
        </section>
      </div>
      <section className="rules-key">
        <h3>读懂一枚棋子</h3>
        <div className="stat-explainer">
          <span>
            <Icon name="sword" />
            攻击：普通攻击伤害
          </span>
          <span>
            <Icon name="heart" />
            生命：当前 / 上限
          </span>
          <span>
            <Icon name="target" />
            射程：攻击路径上限
          </span>
          <span>
            <Icon name="clock" />
            行动：每回合行动次数
          </span>
          <span>
            <Icon name="move" />
            移动：每次移动距离
          </span>
        </div>
        <p>
          棋子下方亮点代表剩余行动，“休”表示尚不可行动。大肉比占2×2格但只有一条生命条；同一个范围法术只对它结算一次。
        </p>
      </section>
      <details className="interpretations">
        <summary>
          本版本的规则解释与待确认事项 <span>展开阅读</span>
        </summary>
        <p className="notice">
          原规则有少数公式、计时和结算顺序的歧义。以下是当前实现采用的明确解释，不代表这些歧义已经由作者确认。
        </p>
        <ol>
          {RULE_NOTES.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ol>
      </details>
      <section className="keyboard-guide">
        <Icon name="help" />
        <p>
          <b>键盘操作：</b>
          方向键在棋盘上选格，Enter或空格操作；Esc取消当前选点；Ctrl/⌘+Z悔棋，Ctrl/⌘+Shift+Z重做。所有按钮也可通过Tab访问。
        </p>
      </section>
    </Modal>
  );
}
