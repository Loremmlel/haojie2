# 浩劫联机接入：实施与宿主契约

基线：`6809bafb8cbe696810da34c8320a4de1fbd791c8`（反馈3）。来源为作者提供的 `haojie2-online-adaptation.md`，以及2026-09-20审阅补充；本轮只改浩劫，不接入网站账号、Cloudflare或真实WS。

## 范围与不变量

这是有边界的会话/交互重构，不复制棋盘，不复制规则，不维护本地和在线两套技能。现有 `HaojieGame` 保留 `initialState`、`initialMatch`、`storageKey`、`onStateChange` 的语义，v2存档、单HTML、人机Worker和历史回放不变。新增 `HaojieOnlineGame` 只接受持续更新的玩家视图，客户端提出命令，最终结果由宿主裁定。

游戏仓库维护：规则、运行时命令校验、游戏内操作者权限、玩家视图及事件可见性、受控交互、表现生命周期。网站维护：认证席位、房间协调、WS、持久化、提交编号/幂等、修订号、重连、部署版本匹配。网站不包含棋子技能特判，也不必固定使用D1。

## 已落实的抽象

```text
HaojieGame：本地Session、历史、存档和AI调度
        │
        ├── useGameInteraction：操作菜单、选点、选路、完整Command
        ├── GameSurface：同一棋盘、面板、规则、图鉴
        └── useGamePresentation：事件、动画、音效
        │
HaojieOnlineGame：宿主快照、连接状态、明确提交回执
```

本地仍通过 `dispatch/applyCommand` 执行，并同步得到结果；在线提交只返回等待状态，不能直接推进游戏或随机抽卡。在线入口不挂载本地会话、AI和存档工具，不注册撤销/重做快捷键，不提供本地新局、换边或AI接管。规则、图鉴、查看棋盘和音效仍可用。

`GamePosition` 不含 `seed/rng`，供共享展示与查询使用；完整 `GameState extends GamePosition` 才能用于权威结算。没有把公开视图强制转换为GameState，也没有补假随机数。

## 公共入口

服务端直接导入 `src/engine/index.ts`，不从React根入口导入：

```ts
import {
  createGame,
  applyPlayerCommand,
  getPlayerView,
  parseCommand,
  actorCommandError,
  canRebasePlayerCommand,
  HAOJIE_RULESET,
  PLAYER_VIEW_VERSION,
} from './haojie/src/engine';
```

`applyPlayerCommand(state, authenticatedPlayer, unknownCommand)` 先验证外来JSON字段、参数类型/范围和操作者，再调用原规则结算。操作者必须来自网站认证后的席位；命令中的 `player` 不可冒名。普通行动、反应、双方独立暗选与超级BW回合外技能均由游戏模块判断。非法命令抛出 `RuleError`，原状态和随机数不变。

`actorCommandError` 只做操作者权限判断，可接受尚未选完目标的UI意图，不代替完整规则校验。`parseCommand` 接受普通JSON对象、拒绝未知字段，并限制ID、数组、路径长度及坐标；宿主仍需限制消息总字节数、提交频率和账号权限。

`getPlayerView(state, viewer)` 返回 `PlayerView`，包含 `viewVersion`、`ruleset`、`viewer` 和 `state`。顶层、反应源、时钟快照、棋子效果、事件身份/坐标均显式投影；不含Session、撤销历史、随机状态。双方手牌和神龛候选保持公开。共同揭示前仅保留查看者自己的最终选择/玉碎奇偶，暗选日志与动画使用封闭的公开描述。

React入口可以从 `src/index.ts` 导入，也可以直接导入受控文件：

```tsx
'use client';
import {
  HaojieOnlineGame,
  type HaojieOnlineGameProps,
} from './haojie/src/ui/online/HaojieOnlineGame';

export function Match(props: HaojieOnlineGameProps) {
  return <HaojieOnlineGame {...props} />;
}
```

入口自带作用域样式，不需要复制棋盘或从站点重写样式。无需发布npm包；按固定Git提交引用/同步即可。

## 宿主输入与提交契约

```ts
interface OnlineUpdate {
  matchId: string;
  revision: number;
  kind: 'update' | 'snapshot';
  view: PlayerView;
}

type CommandReceipt = { ok: true; revision: number } | { ok: false; message: string };
```

组件属性是 `update`、`connection`（`connecting/connected/disconnected`）、可选 `disabled`、可选 `error: { id, message }`，以及 `onCommand(command, { baseRevision, signal }) => Promise<CommandReceipt>`。

宿主每次传入完整玩家视图。`revision` 是房间单调递增修订号，不是 `ply`、`serial` 或存档 `version`。同一对局的重复/较旧更新不覆盖新局面，也不重复播放动画。`kind: 'snapshot'` 用于首次载入与重连，恢复局面而不回放历史。普通修订更新不会重挂载棋盘；只有真正更换 `matchId` 或认证席位才重建会话。不要重复使用同一 `matchId` 表示新的对局。

提交时组件立即上锁。广播与回执独立：只有收到成功回执，并已收到至少该回执修订号的局面，才解除本次提交锁，二者先后顺序不限。对手广播不能充当本次回执。显式拒绝显示错误并恢复操作；宿主异常抛出也会显示错误。`error.id` 用于区分宿主主动推送的新错误。

请求编号由网站生成并维护。连接重试须沿用同一个请求编号，服务端去重，不能再次执行已经提交的攻击。成功回执必须对应已持久化的状态、修订号及请求结果；不要在持久化前确认成功。

断线或组件销毁会触发 `AbortSignal`，但这只取消客户端等待，不能撤销已经提交的服务端命令。对于超时、丢回执等“结果不确定”情况，网站应将连接标为 `connecting/disconnected` 并同步快照/查询原请求结果，确认后再恢复 `connected`；不能当作明确失败直接盲重放。宿主负责超时策略，组件不创建传输重试循环。

`disabled` 表示网站的维护、同步等外部锁，不应简单设为 `active !== viewer`。回合外免费技能和对手反应仍可能属于本人；游戏模块决定哪些操作可用。

## 并发暗选与过期命令

`canRebasePlayerCommand(state, actor, command)` 是游戏侧的狭窄策略入口：目前只允许仍处于首次暗选窗口、尚未提交的玩家完成独立选择。两人可以基于同一旧修订提交神龛，先到的一方不应使另一方永远卡住。网站调用此接口，无需自行判断神龛技能或模式。

其他过期移动、攻击等一律重新同步，由玩家重新选择，不自动重放。宿主仍须确保房间命令串行/原子提交；这个查询不是并发锁，不提供身份认证或持久化。未来若新增可重复进入的秘密选择窗口，应同时引入并校验对应窗口标识，而不能扩大旧命令重放范围。

## 公开查询的准确含义

`inspectCommand(position, command)` 返回 `available/uncertain/invalid`。公开预检共享原规则代码，但在首次需要私有随机值时停止，不猜测结果，也不返回假想的新状态。`queryCommandError` 和 `canAttemptCommand` 支持现有操作菜单、目标和路径选择；“可尝试”不等于服务端保证执行成功。

公开信息能确定的冻结、回合、射程、占位等仍可正常提示。依赖随机结算之后才能确定的情况，可能最终由服务端拒绝，必须走上述失败恢复路径。完整本地GameState的预检继续执行原 `commandError`，不改变本地提示、随机序列或正式结算行为。

## 例子与验证

可运行的最小示例：`examples/online/TwoPlayerDemo.tsx`。它用 `examples/online/room.ts` 的 `DemoRoom` 驱动两个受控棋盘，视图和命令经过JSON边界，展示明确回执、修订号、请求去重和独立选择接收方式。

**DemoRoom只是内存演示宿主，不是生产房间服务。** 为了演示，示例把它放在同一浏览器中；真实网站必须将权威状态、引擎调用和去重记录移到服务器。测试专用 `inspect()` 和 `window.onlineDemo` 不能进入生产客户端。

自动测试位于 `tests/session/online.test.ts` 及 `tests/browser/online.mjs`（故障注入宿主在 `tests/browser/online-host.tsx`）。后者在离线浏览器中挂载两个真实组件，覆盖连续对局、JSON视图、权限、路径、重复/旧更新、回执先后顺序、明确拒绝、断线、延迟回执、神龛并发暗选、无本机存储/AI及窄屏布局。执行：

```sh
npm run check
npm test
npm run deploy
npm run deploy:check
npm run test:browser:online
```

标准CI还保留本地双人、存档导入导出/撤销、人机三档、Pages嵌入、VFX、神龛、反馈3和各历史规则版本CLI回放。测试截图与报告输出到 `artifacts/online-*.png`、`artifacts/online-browser-report.json`；报告记录实际完成的场景，不用模拟宿主测试替代真实WS验收。

## 维护与已知边界

前端组件与服务端引擎必须固定到同一提交/发行版本，并检查 `HAOJIE_RULESET` 与 `PLAYER_VIEW_VERSION`。前者标识规则，后者标识公开视图协议，均不同于v2存档格式和房间修订号。规则或视图变化时维护对应版本；活动对局不得默默切换规则。常量检查不能代替双端锁定同一份代码。

服务端必须生成并保存建局种子，不能无参调用 `createGame()` 而使用公开固定默认值。示例固定种子仅用于复现。此轮保留已有确定性PRNG以兼容本地存档/回放，没有声称它具备密码学安全性或实现完整反作弊。

网站自行选择房间协调及持久化设施，例如一个房间一个Durable Object；不得把普通进程内存当作可恢复存储，也不应在两处独立修改同一权威局面。本轮不包含真实WS、Cloudflare绑定、账户、生产房间、联网悔棋协商、观战、掉线超时判负或上线部署。Next.js实际构建及网站集成仍需宿主验收，不能由这里的React浏览器测试代替。

## 本次执行记录

2026-09-20，运行代码与单文件成品提交 `5c1d228d8d3b1a40fd10e9f13f257ea4ebd3c156` 通过[完整分支验收](https://github.com/Loremmlel/haojie2/actions/runs/35485444677)：TypeScript、全部行为测试、构建/发行一致性、本地浏览器、Pages、三档AI、VFX、神龛、反馈3、受控双客户端、神龛CLI冒烟与反馈3命令指纹。此后的交接整理仅改文档/格式和测试配置，收尾流程会检查src、tests、examples、scripts、package及index.html与该通过版本没有差异。历史规则版本的录制另由标准PR CI按固定旧提交验证。

宿主应把每次发布的update/view视为不可变快照，更新时提供新对象，不要原地修改已发布局面。受控组件只缓存已接受的宿主视图，不提供自己的第二份权威结算。
