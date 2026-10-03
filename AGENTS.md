# 浩劫维护入口

本文件只列全局硬约束。详细且仍有效的工程与规则约束见[维护约束](docs/maintenance/ENGINEERING-CONSTRAINTS.md)；当前 AI/训练状态只读[状态入口](docs/ai/STATUS.md)，历史进度见[归档记录](https://github.com/Loremmlel/haojie2/blob/fcf37e73f91629965255b6fa4a5da1c80e7fa71a/docs/ai/history/TRAINING-PROGRESS.md)。实施规则以[现行规则](docs/RULES.md)和较新的作者反馈为准；来源优先级为作者回复 > 新增段落的明确修改 > 一般规则 > 旧版猜测。旧文中的“当前”“下一步”只对应其记录日期，不自动授权新工作。

## 游戏和公开边界

- 生产只生成一个可离线打开的 HTML；禁止运行时 CDN、图片/字体服务器、分包和隐式请求。`src/engine/index.ts` 不依赖 React、DOM、存储、音频、时钟或网络；`src/index.ts` 提供 React 组件。
- 所有游戏变化通过 `applyCommand`。输入不可变；非法命令不部分支付资源或消耗正式随机数；预览不消耗正式随机数。规则数字与图鉴共用 `catalog.ts`，状态词条共用 `engine/library/keywords.ts`。
- 真实规则版本、随机数边界、来源/权限与公开信息边界不得因工程整理变化。AI 只接收 `observe` 白名单，不能收到 seed/rng、整份 Session 或暗选私有信息；TS 与 Rust 共用同一规则及 AI 算法流程。Node 变慢不构成单端分叉理由；语言机制可不同，算法对应关系须记录并双端验证。
- 存档、训练记录与历史回放遵守原版本及指纹；旧档不伪造过程，旧未知结果不补胜负，不改写历史哈希。保持 `haojie.session.v2` 键、GameState v2 结构及离线发行、公开 `HaojieGame` / `HaojieOnlineGame` 接入契约；联网宿主的身份、房间与持久化不移入游戏规则。

## 工程和验证

- 生产代码、共享训练组件、研究入口和历史实验按实际依赖分开；共享组件不得依赖历史实验目录。不得为目录整理改游戏规则、AI 算法、训练方法或随机消耗。原始轨迹、模型、优化器、冻结源码、失败报告和缓存都是本地产物，未跟踪或忽略不代表可删除。
- 项目维护目录超过 10 个直属文件时按职责分组；不使用数字目录或 `misc` 凑数。移动文件要核对导入、脚本/CI、相对路径、工作目录、子进程路径、文档链接、复现命令和公开入口。
- 新增或修改注释用中文；作者原文、旧回放和第三方许可原样保留。页面源码变化时运行 `npm run deploy` 并提交根 `index.html`；纯工程整理仍须核对 `deploy:check`，不部署。
- 提交前运行格式、TypeScript、行为测试、构建、发行一致性、`check:structure` 和离线 `file://` 浏览器验收。Python/Rust 入口各自验收；环境缺失时明确报告，不把 `--help` 当功能通过，不绕过浏览器管理策略。

## 细则索引

- [规则与反馈](docs/RULES.md)、[2.5 合成](docs/changes/CHANGELOG-2.5.md)、[3.0 神龛](docs/changes/CHANGELOG-3.0.md)、[9 月 23 日反馈](docs/feedback/FEEDBACK-2026-09-23.md)：后续规则覆盖旧部署行快照，当前部署行从实时局面计算。
- [AI 算法](docs/ai/AI.md)、[训练导航](docs/ai/TRAINING.md)、[TS/Rust 状态分支对应](docs/ai/performance/kernel/ARCHITECTURE.md)、[Rust 接口](native/engine/README.md)。
- [在线接入](docs/ONLINE-ADAPTATION.md)、[存档与训练记录](docs/session/SAVES.md)、[特效](docs/VFX.md)。
