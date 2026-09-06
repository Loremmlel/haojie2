# 架构与嵌入

## 边界

本版本是同屏双人游戏，没有服务器、房间匹配或AI对手。规则引擎与React界面分离，不需要为了本地对战额外启动后端。

```text
React界面 → Intent分步选点 → Command → 纯规则引擎
                                  ↓
                      新GameState + GameEvent[]
                                  ↓
                棋盘 / 动画 / 战报 / 存档 / 悔棋
```

`src/engine`不引用React、DOM、localStorage、音频、HTTP或系统时钟。`createGame(seed)`产生确定性初始状态；`applyCommand(state, command)`先克隆状态，再验证和结算。失败时抛出`RuleError`，输入状态不会被部分修改。随机数状态、棋子、手牌、基地标记、效果计时、待结算反应均可序列化。

`src/ui/intents.ts`只处理“先选棋子、再选目标/落点”的交互步骤，不自行扣血或判断胜负。完整操作通过引擎的`commandError`验证后执行。`GameEvent`只描述视觉事件，坐标采用快照，动画完成与否不会影响对战结果。

`history.ts`保存最多60步快照，支持跨回合撤销/重做。新操作会丢弃重做分支。悔棋连同随机数一起恢复，不允许通过撤销再做来重掷同一攻击或重新抽同一组牌。

## 复用整个React组件

将本项目的`src`作为子目录放入现有React工程，从公开入口导入：

```tsx
'use client'; // Next.js App Router中需要；普通React项目不需要
import { HaojieGame } from './haojie/src';

export default function GamePage() {
  return (
    <HaojieGame
      storageKey="my-site.haojie.match-1"
      onStateChange={(state) => console.debug('回合', state.ply)}
    />
  );
}
```

入口会一并引入样式。应用样式以`.hj-game`为作用域，不重置宿主站点的`body`、全局按钮或标题。独立页面的根节点重置仅由构建脚本生成的HTML提供。

`initialState`只在首次挂载时读取；需要从宿主替换整盘棋时，用新的React `key`重新挂载。设置`storageKey={null}`可关闭组件的自动存档，由宿主通过`onStateChange`自行保存。每个嵌入实例应使用不同存储键；建议一个页面只放一个正在操作的棋盘，避免全局悔棋快捷键同时作用于多个实例。

## 只复用规则引擎

```ts
import { createGame, applyCommand, commandError } from './haojie/src/engine';

let state = createGame(20260906);
const command = { type: 'end' } as const;
const error = commandError(state, command);
if (error === null) state = applyCommand(state, command);
else console.info(error); // 例如仍有待部署随从
```

公开类型包括`GameState`、`Command`、`Unit`、`Card`、`GameEvent`。客户端不需要接触引擎内部的伤害结算函数。

未来做联网对战时，可以把同一引擎放在服务端，服务端持有权威状态与随机数，接收命令后广播状态和事件。还需要自行实现身份、轮次/版本校验、房间、重连、悔棋双方确认与防作弊；当前本地存档的结构验证不能替代这些网络安全边界。

## 单HTML构建

`scripts/build.mjs`用esbuild将React、ReactDOM、游戏代码和CSS打包，所有内容内联到`dist/index.html`。脚本拒绝外部运行时导入与额外资产输出；产物是IIFE，不依赖模块请求、CDN、字体服务或图片服务器。棋子和图标使用内联矢量与CSS，音效由Web Audio在用户主动开启后合成。

`npm run dev`监听源码并重建页面；修改后手动刷新浏览器，不提供HMR。`npm run build`生成离线发行版。

## 验证分层

`tests/engine.test.ts`验证26种棋子、特殊结算、部署、寻路、计时、随机性、胜负及存档行为，不依赖React内部状态或DOM实现。

`tests/browser.mjs`直接以`file://`加载生产HTML，并将浏览器网络设为离线。验证真实按钮操作、伤害、悔棋/重做、刷新恢复、导出导入、图鉴、规则、手机宽度和减少动态效果设置；记录外部请求和未捕获异常。测试使用可访问名称及棋盘坐标，不使用整页截图像素断言。

GitHub Actions产出可下载的`haojie2-single-html`、源码快照和验收截图/JSON报告。截图是验收证据，不作为脆弱的像素级测试基线。
