# 浩劫 3.0 · 神龛 / haojie2

原生 V-trace 训练见[快速启动](training/README.md)和 [Colab Notebook](training/notebooks/root-vtrace.ipynb)，当前验收进度见[实测记录](training/online/VALIDATION.md)。

React + TypeScript 回合制战棋，支持经典 / 神龛模式、同屏双人和三档本地 AI。游戏发行物是一个可离线打开的 HTML，React、样式、棋子、特效和 AI Worker 全部内联。

仓库也包含 **Rust 原生多局运行时 + Python 单 GPU 训练所有者**，通过命令控制并发、保存和恢复未完成对局。训练模型尚未接入网页发行版，网页继续使用现有手工搜索 AI。

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

## 本地自对弈与训练

先按[训练安装说明](training/README.md)安装 Python 3.12 环境、PyTorch 和训练包，并构建正式引擎：

```sh
npm run native:build
python -m haojie_training.online run --directory artifacts/root-vtrace/my-run
```

使用已安装训练包的 Python。通过同一模块的 `status`、`resize`、`save`、`stop` 命令控制运行；停止保存未完成对局，恢复使用 `run --resume`。

| 用途                           | 入口                                       |
| ------------------------------ | ------------------------------------------ |
| 安装、启动、并发控制与保存恢复 | [Python 训练包](training/README.md)        |
| 模型输入、连续轨迹与运行时     | [新系统契约](training/online/README.md)    |
| 规则参照与历史记录审核         | [TypeScript 验证工具](docs/ai/TRAINING.md) |
| 真实硬件结果与尚未完成的验收   | [验收记录](training/online/VALIDATION.md)  |
| 当前实现和维护导航             | [AI / 训练状态](docs/ai/STATUS.md)         |

训练只维护新 V-trace 路径，旧模型和实验产物保留原样。自然棋力需要固定对手的独立评测；历史实验和测量见[历史索引](docs/ai/HISTORY.md)，原生接口见[引擎说明](native/engine/README.md)。

## 仓库结构

| 目录                        | 职责                                       |
| --------------------------- | ------------------------------------------ |
| `src/engine/`               | TypeScript 权威规则、公开视图、存档与回放  |
| `src/ai/`、`src/match/`     | 本地 AI 搜索、公开训练编码、对局环境       |
| `src/ui/`                   | React 游戏界面，组件与样式按功能就近维护   |
| `native/engine/`            | Rust 规则实现、动作树、编码与原生采样协议  |
| `training/haojie_training/` | BF16 模型、连续轨迹学习器和单 GPU 控制层   |
| `scripts/`                  | 构建、CLI、数据工具和原生差分验证          |
| `tests/`、`training/tests/` | 游戏行为、浏览器、TS / Rust 对应及训练验证 |
| `docs/`                     | 当前接口、规则与维护说明                   |

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

```sh
npm run format:check
npm run check
npm test
npm run check:structure
npm run train:rules:check
npm run train:teacher:check
npm run deploy:check
npx playwright install chromium
npm run test:browser
npm run test:pages
```

原生代码另运行 `npm run native:test`、`npm run native:check`，并使用已安装训练依赖的 Python 运行 `python -m unittest discover -s training/tests/online -p 'test_*.py'`；独立安装测试通过 `HAOJIE_NATIVE_LIB` 指定共享库。CI 验证两端规则、动作候选、公开编码、规范哈希和新训练会话恢复；CUDA 棋力和性能另做实测。

浏览器验收默认在网络离线的 `file://` 页面执行真实操作；特效开发预览使用 `npm run preview:vfx`。CLI 对局和回放使用 `npm run play:cli`，参数见[CLI 手册](docs/CLI.md)。

提交规则与数据兼容边界见 [AGENTS.md](AGENTS.md)及[维护约束](docs/maintenance/ENGINEERING-CONSTRAINTS.md)。
