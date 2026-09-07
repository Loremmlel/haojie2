import { RULE_NOTES } from '../../engine';
import { Modal } from '../shared/Modal';
import { Icon } from '../shared/visuals';
export function Rules({ onClose }: { onClose: () => void }) {
  return (
    <Modal
      title="浩劫 · 规则手册"
      subtitle="以《浩劫.docx》、文末回复及作者后续补充为准。冲锋号令储存期限已确认为8回合。"
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
          冲锋号令储存8个己方回合，包含抽到的回合。善铁及相关合成属于3.0神龛模式，作者已决定暂缓；本版不提前实现。
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
