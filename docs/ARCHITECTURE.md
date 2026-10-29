# 浩劫 · 架构与接入边界

## 运行入口

| 入口             | 调用关系与职责                                                        |
| ---------------- | --------------------------------------------------------------------- |
| 离线游戏         | React 提交 Command，TS 引擎原子结算；UI 观察新的局面、事件和战报      |
| 联机组件         | 宿主保存权威 GameState，按玩家下发 PlayerView；组件提交命令和显示回执 |
| 本地 AI / TS CLI | 手工搜索仅接收公开 Observation，通过同一引擎执行命令                  |
| 原生规则验证     | Rust 规则库与 TS 参照比较预检、正式命令及公开观察                     |

`src/engine/index.ts` 是纯规则入口，不依赖 React、DOM、存储、音频、网络或系统时钟。公开状态、命令、随机边界和规则数据在 TS / Rust 两端保持对应；语言内部所有权实现见[运行内核](ai/RUNTIME.md)。

## 代码归属

| 位置                                             | 职责                                                             |
| ------------------------------------------------ | ---------------------------------------------------------------- |
| `src/engine/types.ts`、`catalog.ts`              | 可序列化状态和命令协议；棋子、法术、武器、神龛及正式数值         |
| `src/engine/core/`、`runtime/`                   | 状态、时钟、身份、几何查询、实体表示与工作分支                   |
| `src/engine/commands/`、`setup/`                 | 操作、原子结算、召唤、合成、神龛与反应队列                       |
| `src/engine/session/`、`online/`                 | 存档和增量回放；玩家视图与操作者权限                             |
| `src/ai/planning/`、`evaluation/`、`simulation/` | AI 规划、缓存、启发式评分与独立概率模拟                          |
| `src/match/`                                     | 本地对局、CLI 与人机设置                                         |
| `src/ui/`                                        | 按 game、board、inspector、hand、library、session 等功能组织页面 |
| `native/engine/src/`                             | Rust 类型、规则命令、状态分支及公开观察                          |
| `scripts/native/engine/`                         | 规则包生成、规则夹具与双端差分                                   |

## 规则与状态

所有游戏变化通过引擎命令边界。`applyCommand(state, command)` 成功时返回新局面；非法命令保留原状态、手牌、资源及正式随机数。内部连续采样共用同一个规则内核和提交边界。UI 只暂存未选完的目标、落点、路径和材料，完整命令仍由引擎校验。

全局 `ply`、各方 `turns`、单位 `offset` 和实际 `deployedAt` 分开。完整操作、连续攻击、冲撞步数、武器额外攻击各有预算，不能混用。`pending` 保存需要玩家决策的反应，支持在连锁中暂停、保存与恢复。

占位查询统一处理叠放和 2×2 棋子；单体目标、区域作用和覆盖格遵循各自规则。图鉴、属性和规则描述来自同一个 catalog；状态词条来自 `engine/library/keywords.ts`。来源优先级和具体规则见[维护约束](maintenance/ENGINEERING-CONSTRAINTS.md)及[规则手册](RULES.md)。

## AI 与训练

网页三档 AI 位于 `src/ai`，评分和搜索位于规则引擎之外。AI 只接收 `observe` 的公开白名单，正式 seed/rng、整个 Session 和未揭示暗选留在权威宿主。独立模拟随机源不改变正式命令的随机序列。预算、收尾与缓存细节见[AI 算法](ai/AI.md)。

React 的 opponent 控制层负责 Worker 调度、展示节奏和取消。标准构建将 Worker 内联为 Blob；宿主未注入 Worker 时使用相同搜索生成器分片运行。新局、载入、悔棋或卸载会取消旧任务，返回结果在执行前仍须检查当前局面。

普通局训练使用 Python 常驻多进程、进程内 Rust 规则/三档手工 AI、共享内存交换和单 GPU PyTorch 原生 BF16 学习/推理。5.50M 实体网络复用状态主干，按来源、动作、参数解码；完整命令级随机 PUCT 采用渐进扩宽和局面复用。先完成独立模仿预训练，再启动自对弈及跨时期对手采集。实现与数据语义见[普通局训练](ai/TRAINING.md)，命令见[运行入口](../training/README.md)。

## 页面、样式与表现

`Game.tsx` 是组合根。控制器协调选点与命令，会话层负责持久化和视听生命周期，展示组件通过 props 接收数据。复杂技能结算不放进 React。

组件的 CSS、状态样式和响应式规则就近维护。`styles/index.css` 控制导入顺序，最后处理减少动态效果；跨功能颜色与尺寸使用 tokens。`.hj-game` 限定组件作用域，宿主页面的 body 重置不进入组件。

当游戏容器宽度达到 1100px 时，完整 9×13 正方形棋盘和主要控制一屏可用，长侧栏独立滚动。棋盘按可用宽高等比适配，宿主可用 `--haojie-height` 指定高度；手机保留自然文档流。快捷键仅作用于当前获得焦点的游戏实例。

`GameEvent` 保存事实、路径和来源 / 目标快照。特效、动画和默认关闭的音效观察结算结果；表现不驱动伤害、反应或计时，也不消耗规则随机数。会话替换或卸载时释放表现任务。细节见[特效说明](VFX.md)。

`UnitText` / `UnitLink` 共用图鉴与名称链接，`RuleReference` 以稳定类型区分同名单位和词条。查看实例时从最新公开局面派生，不把旧棋子对象保存为界面状态。

## React / Next.js 接入

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

`initialState` 和 `initialMatch` 在初次挂载读取；替换对局使用新的 React `key`。多个实例使用不同 `storageKey`，或设为 `null` 交给宿主持久化。`onStateChange` 接收完整 GameState。

联机入口 `HaojieOnlineGame` 接收宿主修订快照和明确回执，与本地入口共用 GameSurface、交互和表现。服务端使用 `applyPlayerCommand` 校验结构及操作者权限，以 `getPlayerView` 生成各玩家的嵌套公开视图。账号、房间、WebSocket、去重、重连和持久化由宿主负责，见[联机协议](ONLINE-ADAPTATION.md)。

## 保存、发行与验证

`Session` 保留 60 步运行时撤销缓存；持久化使用初始局面、全部实际命令路线和当前局面的增量记录。导入同版重放并核对终态后才替换会话。2.x / 3.0 继续使用 GameState v2 与 `haojie.session.v2`，旧存档兼容见[存档说明](session/SAVES.md)。

esbuild 将 React、CSS、游戏和 Worker 内联为一个 HTML。构建拒绝运行时外部导入与额外分包，保留第三方许可。Pages 根发行物与源码一同提交，CI 用 `deploy:check` 检查一致性，见[发布说明](PAGES.md)。

行为测试维护可观察规则、非法操作原子性和数据恢复。浏览器测试直接操作离线发行物；原生验证比较 TS / Rust 预检、正式规则、完整后继、公开观察及驻留恢复。入口见根 [README](../README.md)。
