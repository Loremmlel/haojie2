# 浩劫 · 引擎与页面边界

## 数据流

```text
React 展示/交互 → ActionSpec / 分步Intent → Command
                                           ↓
                     applyCommand(previous, command)
                       验证 → 原子结算 → 新GameState
                                           ↓
                   React棋盘、事件动画、战报、存档
```

`src/engine`为纯TypeScript，不引用React、DOM、音频、本地存储、网络或系统时间。`createGame(seed)`与`applyCommand(state, command)`是公开的局面入口；`commandError`同样通过纯引擎验证，不编写另一份仅供界面的规则。

## 模块职责

| 模块           | 职责                                                                     |
| -------------- | ------------------------------------------------------------------------ |
| `types.ts`     | 版本2可序列化状态、操作、来源与事件协议                                  |
| `catalog.ts`   | 59条棋子/法术/武器定义，两个抽取池和公开规则说明；图鉴与引擎共享唯一数据 |
| `state.ts`     | 状态内PRNG、实时属性、模式操作预算、蓄力快照、个体效果时钟               |
| `geometry.ts`  | 占位、叠放、路径、攻击阻挡、行控制；不导入页面                           |
| `combat.ts`    | 伤害、保护、死亡、人头、反击、命中效果；区别普通伤害与消灭               |
| `abilities.ts` | 主动蓄力、技能、施法、装备、合成和改判                                   |
| `movement.ts`  | 普通移动、SZF/小BW连续冲撞、反应队列选点                                 |
| `lifecycle.ts` | 全局回合开始/结束、持续区域、虹吸、DOT和个体推进                         |
| `game.ts`      | 唯一命令边界、阶段约束、胜负和明确标注的演示局                           |
| `history.ts`   | v2存档验证、60步撤销/重做，不接触任何存储API                             |
| `options.ts`   | 提供可展示的操作描述与选点步骤；不含React类型                            |

复杂技能不在React里直接扣血。UI仅暂存还没选完的参数（目标、落点、行列或阵亡记录），完整指令仍由引擎验证。一个动作不合法时，输入状态、手牌和随机数都不变。金身、免疫塔等不能只在UI禁用按钮，必须在效果结算层生效。

## 时钟、操作和占位

状态分别保存全局`ply`、各方`turns`以及单位`offset`。冲锋号令只推进目标单位的有效时间。`deployedAt`与`chargedOnDeploy`独立于该有效时间，记录实际部署和冲锋，避免用“提前行动”绕过部署当回合装备限制。

单位`operations`是已经消耗的完整操作；`mode`锁定当前连续操作，`shots`记录本次攻击次数，`moves`记录冲撞剩余步数。`bonusAttacks`单独表示靴子的额外攻击操作，不能当通用行动点。`charge`存层数，`readyCharge`是回合开始可用层数。

位置不是简单的一格一ID：`occupants`返回全部叠放者，`targetAt`选择栈顶，2×2单位用整个footprint判定。克隆`group`关联一批8枚，并与尚未部署的同批牌联合决定是否发生最后死亡。冰冻保留所有权字段，但`allegiance`在敌我判断中返回中立。

## 可恢复的结算

所有等待玩家的反应都在`pending`中，不放进React回调或定时器。回合末仍有反应时，显式等待队列完成再切换，存档和悔棋能停在这一步。固定顺序与状态内RNG保证同命令重放结果相同。

`GameEvent`保存坐标/路径及可选的来源、目标身份/尺寸快照，没有对随后可变棋子的引用。`event-facts.ts`提供同步事实与因果作用域，不增加事件ID或消耗PRNG。表现层`board/vfx`负责纯映射、命中时序、有上限批次与可取消的棋子运动；持续状态仍由Board读取当前局面。视觉与生命周期契约见[VFX.md](VFX.md)。动画和音效是状态结果的观察者，动画结束不影响规则结果；减少动态效果或关闭声音不会改变对局。

## 接入React / Next.js

```tsx
'use client';
import { HaojieGame } from './haojie/src';

export function Match() {
  return (
    <HaojieGame
      storageKey="site.haojie.match-001"
      onStateChange={(state) => console.debug(state.ply)}
    />
  );
}
```

入口导入作用域CSS，不重置宿主`body`。设置`storageKey={null}`关闭组件自动持久化；宿主通过回调负责保存。`initialState`只在首次挂载读取，需要替换整个对局用新的React `key`。每个嵌入实例用不同存储键；页面同时有多实例时，快捷键已按聚焦的游戏实例隔离。

只复用引擎：

```ts
import { createGame, applyCommand, commandError } from './haojie/src/engine';
let state = createGame(20260907);
const command = { type: 'summon', ultimate: false } as const;
if (!commandError(state, command)) state = applyCommand(state, command);
```

将来可以在服务端运行同一引擎，但本版本没有网络房间。身份、权威状态、命令版本号、防作弊、重连和双方同意悔棋仍需另行实现。本地存档验证只是输入校验，不是网络认证边界。

## 发行与验证

esbuild把React、ReactDOM、CSS与游戏代码打成IIFE并内联进唯一HTML。图标和棋子使用内联矢量/CSS，音效为用户主动开启后的Web Audio合成，无外部字体、图片或音频资源。构建拒绝额外chunk/资产和外部导入，保留第三方许可文本。

`tests/core`维护普通库和基础规则，`tests/ultimate`维护终极库，`tests/session`维护历史、状态图与跨能力交互。测试不按每个修复堆一组重复回归，不绑定CSS排列或React内部状态。

浏览器验收默认离线`file://`，测试真实DOM操作、手机视口和导出导入；产出PNG和JSON证据，不用截图像素作为脆弱断言。仅当环境不允许页面导航时使用明确标识的内存渲染模式做布局/交互辅助验证，最终CI仍走默认文件模式。

## 2.0.1工程组织补充

UI入口仍为`src/index.ts`的HaojieGame及同名Props，Game.tsx仅组合视图。game/useGameController协调意图；session/useGameSession持有会话及声音/事件生命周期；session/storage提供可测试存储适配；SaveTools负责文件导入导出。各展示组件显式传props，不自行修改规则状态。

CSS与各功能就近，styles/index.css保持单一可审阅的顺序，motion.css最后处理减少动态效果。原styles.css、v2.css、refinements.css已移除，不再追加版本补丁样式。公开入口不需要宿主另导入散落样式文件。

已把快捷键限制在当前获得焦点的游戏实例；宿主输入及其他游戏不受影响。多个实例应使用不同storageKey或设为null。版本发布与schema分离，2.0.1保持v2存档；旧号令期限的兼容修正位于engine/migrations.ts，跨所有历史快照统一处理。

结构选型见ENGINEERING.md，Pages生成入口与本机续局语义见PAGES.md。

## 2.1：AI和人机历史的边界

`src/ai`在引擎外生成/评分/搜索命令，`src/match`保存模式设置及人类决策撤销；React opponent/useComputer只做Worker调度、取消与命令提交。引擎可选模拟RandomSource在WeakMap里短暂生效，不改变正式命令的随机行为，不被序列化。

AI接收显式白名单Observation，不接收真seed/rng或Session；小随机分支按概率展开，复杂随机采样。正式执行与假想局面不一致则重新规划；不会按真实存档预测将来的抽卡。

`Session.match`可选，缺省同屏双人；`humanAnchor`保存超长AI回合之前的人类决策。旧schema和storageKey不变。对外Props新增initialMatch只影响初次挂载/新实例，与initialState一致；不改变原onStateChange只传GameState的契约。

标准单文件构建先打包独立Worker，把它内联进主包并以Blob启动。宿主未注入内联常量或Worker不支持时，以相同搜索Generator分片运行；没有偷偷请求worker.js。每次任务都检查取消编号和状态引用。完整搜索与预算边界见AI.md。
