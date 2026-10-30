# 浩劫 3.0 · 神龛 / haojie2

当前 AI 与训练维护入口见[状态说明](docs/ai/STATUS.md)。

React + TypeScript 回合制战棋，支持经典 / 神龛模式、同屏双人和三档本地 AI。游戏发行物是一个可离线打开的 HTML，React、样式、棋子、特效和 AI Worker 全部内联。

本分支重写普通局训练为 Python 多进程、进程内 Rust 规则/三档手工 AI、PyTorch 原生 BF16，网络约 5.50M，支持分层动作解码与图搜索。安装与运行见[训练入口](training/README.md)，研究依据与数据语义见[训练设计](docs/ai/TRAINING.md)。网页继续使用现有手工搜索 AI。

## 游戏开发与发行

开发需要 Node.js 22：

```sh
npm ci
npm run dev
```

打开 `http://127.0.0.1:4173`。源码修改后自动重建，刷新页面查看结果。

```sh
npm run build        # 生成独立的 dist/index.html
npm run deploy       # 同时更新 GitHub Pages 使用的根 index.html
npm run deploy:check # 检查根发行物与源码构建一致
```

将 `dist/index.html` 复制到任意目录，在现代浏览器打开即可游玩。GitHub Pages 使用 `main / (root)`；影响网页的源码修改须连同重新生成的根 `index.html` 提交。详情见[发布说明](docs/PAGES.md)。

正式对局从空棋盘开始。经典模式先完成召唤，再进入行动；神龛模式先暗选与设置神龛。支持悔棋、重做、自动保存、JSON 导入导出和演示棋局。当前规则见[规则手册](docs/RULES.md)，原始材料、版本变更和作者反馈分别保存在 `docs/source/`、`docs/changes/`、`docs/feedback/`。

## 本地对局与原生验证

CLI 对局与回放使用 `npm run play:cli`，参数见[CLI 手册](docs/CLI.md)。Rust 规则库和双端差分的构建入口见[原生引擎](native/engine/README.md)。

## 仓库结构

| 目录                              | 职责                                      |
| --------------------------------- | ----------------------------------------- |
| `src/engine/`                     | TypeScript 权威规则、公开视图、存档与回放 |
| `src/ai/`、`src/match/`           | 手工 AI 搜索与本地对局环境                |
| `src/ui/`                         | React 游戏界面，组件与样式按功能就近维护  |
| `native/engine/`                  | Rust 规则、公开观察与可信开发协议         |
| `scripts/`                        | 构建、CLI、数据工具和原生差分验证         |
| `scripts/native/engine/fixtures/` | TS / Rust 规则差分夹具                    |
| `tests/`                          | 游戏、浏览器、状态内核及手工 AI 验证      |
| `docs/`                           | 当前接口、规则与维护说明                  |

职责与接入边界见[架构说明](docs/ARCHITECTURE.md)；TS / Rust 的状态所有权和采样对应见[运行内核](docs/ai/RUNTIME.md)。

## 嵌入现有网站

```tsx
'use client';
import { HaojieGame } from './haojie/src';

export default function GamePage() {
  return (
    <HaojieGame
      storageKey="my-site.haojie.match"
      initialMatch={{ mode: 'ai', human: 1, difficulty: 'medium' }}
    />
  );
}
```

本地组件还支持 `initialState`、`storageKey={null}` 和 `onStateChange`。联机宿主使用 `HaojieOnlineGame`，在服务端调用 `applyPlayerCommand`，并向各玩家下发 `getPlayerView`；账号、房间、WebSocket 和持久化由宿主负责。见[联机接入协议](docs/ONLINE-ADAPTATION.md)和[双玩家示例](examples/online/TwoPlayerDemo.tsx)。

## 验证与维护

按[测试范围](docs/maintenance/TESTING.md)选择受影响的检查。游戏全套为 `npm run test:game`，训练图表为 `npm run test:training-ui`；Python 训练和 Web 控制分别运行自己的 unittest。只改训练代码无需运行游戏引擎/浏览器测试，只改训练 UI 无需启动 GPU。

`npm run check` 检查类型，`npm run check:structure` 检查目录；格式使用 `config/prettier.json`。`npm run deploy:check` 自身完成游戏构建和发行一致性检查，无需重复先运行 build。原生规则/AI 改动才运行对应 Rust 测试及 TS/Rust 差分。

浏览器验收默认在网络离线的 `file://` 页面执行真实操作；特效开发预览使用 `npm run preview:vfx`。CLI 对局和回放使用 `npm run play:cli`，参数见[CLI 手册](docs/CLI.md)。

提交规则与数据兼容边界见 [AGENTS.md](AGENTS.md)及[维护约束](docs/maintenance/ENGINEERING-CONSTRAINTS.md)。
