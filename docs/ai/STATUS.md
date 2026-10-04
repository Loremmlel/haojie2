# AI / 训练当前状态

2026-10-04 核对远端 `main=146ad89d274ec404cd16070a0e77b98703c7a8b1`（PR #11）。正式 Rust 引擎、Python/PyTorch 执行优化与常驻采样均已合并。

## 维护入口

- [无 Node 安装、采样与恢复](../../training/NATIVE.md)：完整记录研究模式；常驻工作池采样固定检查点，尚不自行更新学习器。
- [Python 训练](../../training/README.md)、[编码协议](../../training/ENCODING.md)、[TS 接口](TRAINING.md)。现有模型用已执行动作监督与真实终局价值，不是已验证的收益驱动策略改进。
- [原生引擎](../../native/engine/README.md)、[TS/Rust 共同算法](performance/kernel/ARCHITECTURE.md)。正式规则、公开观察、权威拒绝和随机数边界不变。
- [执行优化](performance/execution/README.md)、[常驻采样验收](performance/resident/README.md)：性能口径与失败证据；不能把更新速度、模仿 loss 或固定模型自对弈解释为棋力。
- [历史索引与本轮清理](HISTORY.md)。旧报告里的“下一步”不自动授权工作。

## 未解决的学习问题

组合模型与隔夜候选没有可靠棋力增益，未晋级发行 AI。自然终局稀疏、样本相关、行为模仿缺乏明确策略改进来源仍是限制。当前研究 CLI 的全量特征分片写入量不适合长期桌面训练；常驻报告的30天外推不是月级稳定性证据。网页游戏仍使用真实简单/中等/困难教师搜索。

历史文件、未跟踪模型与本地产物不因文档清理删除。既往回收记录见[本地产物清理记录](../maintenance/CLEANUP-2026-10-03.md)，该日期的占用不是今天的实时磁盘统计。
