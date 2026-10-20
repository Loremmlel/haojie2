# 浩劫 3.0 · 神龛 / haojie2

服务器与 Colab 集中训练见 [快速启动](training/COLAB.md) 和 [Notebook](notebooks/haojie_colab.ipynb)。

React + TypeScript 回合制战棋，支持经典 / 神龛模式、同屏双人和三档本地 AI。游戏发行物是一个可离线打开的 HTML，React、样式、棋子、特效和 AI Worker 全部内联。

仓库也包含 **Rust 规则引擎 + Python 训练控制台**：在本机页面管理自对弈、持续训练、暂停恢复、历史对手和真实三档 AI 评测。训练模型尚未接入网页发行版，网页继续使用现有手工搜索 AI。

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

先按[训练安装说明](training/NATIVE.md)安装 Python 3.12 环境、PyTorch 和训练包，并构建正式引擎：

```sh
npm run native:build
python -m haojie_training.console
```

使用已安装训练包的 Python。浏览器打开 `http://127.0.0.1:8765`，在页面中新建训练，然后开始 / 继续；默认受管根为 `~/.haojie-training`。可用 `--engine` 指定引擎，`--root` 指定整个训练根。

| 用途                                 | 入口                                       |
| ------------------------------------ | ------------------------------------------ |
| 日常训练、暂停恢复、评测与 20GB 配额 | [训练控制台](training/CONSOLE.md)          |
| 独立原生采样、记录审核、检查点恢复   | [Rust + Python CLI](training/NATIVE.md)    |
| 数据准备、模型训练、推理与导出       | [Python 训练包](training/README.md)        |
| 教师数据、已有搜索记录、模型对战     | [TypeScript 训练工具](docs/ai/TRAINING.md) |
| 模型输入和样本语义                   | [编码协议](training/ENCODING.md)           |
| 当前实现和维护导航                   | [AI / 训练状态](docs/ai/STATUS.md)         |

训练控制台使用终局收益更新 MC-Q 模型，原记录 CLI 继续保留其监督训练语义。自然棋力由冻结模型的独立评测检验；历史实验和测量见[历史索引](docs/ai/HISTORY.md)。

## 仓库结构

| 目录                        | 职责                                       |
| --------------------------- | ------------------------------------------ |
| `src/engine/`               | TypeScript 权威规则、公开视图、存档与回放  |
| `src/ai/`、`src/match/`     | 本地 AI 搜索、公开训练编码、对局环境       |
| `src/ui/`                   | React 游戏界面，组件与样式按功能就近维护   |
| `native/engine/`            | Rust 规则实现、动作树、编码与原生采样协议  |
| `training/haojie_training/` | Python 模型、训练器、原生宿主和 Web 控制台 |
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

原生代码另运行 `npm run native:test`、`npm run native:check`，并使用已安装训练依赖的 Python 运行 `python -m unittest discover -s training/tests -p 'test_*.py'`；原生测试通过 `HAOJIE_NATIVE` 指定已构建引擎。CI 还验证两端规则、动作候选、编码、规范哈希、MC-Q 选择和独立安装恢复。

浏览器验收默认在网络离线的 `file://` 页面执行真实操作；特效开发预览使用 `npm run preview:vfx`。CLI 对局和回放使用 `npm run play:cli`，参数见[CLI 手册](docs/CLI.md)。

提交规则与数据兼容边界见 [AGENTS.md](AGENTS.md)及[维护约束](docs/maintenance/ENGINEERING-CONSTRAINTS.md)。
