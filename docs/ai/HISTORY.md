# 训练历史索引与维护边界

清理提交验收：420 TS、41 Python、10 Rust 测试通过；35 项离线 file:// 场景通过；类型、构建与目录检查通过。清理阶段全仓格式与发行一致性检出43个文件及根HTML行尾差异，未在清理提交中重建发行物。功能阶段格式化并重建后全通过，Git内容核对确认这些既有源码、规则夹具与根HTML没有内容变化；不是游戏行为修复。

本轮清理基线：[146ad89d 固定版本](https://github.com/Loremmlel/haojie2/tree/146ad89d274ec404cd16070a0e77b98703c7a8b1)。不复制旧文件到归档目录，不改原始轨迹或指纹。

- 删除八份已由正式内核架构、执行报告和运行说明替代的阶段文档：`TRAINING-NODE-2026-09-22.md`、`TRAINING-PYTORCH-2026-09-22.md`、`performance/{RUST-PROTOTYPE,TS-OPTIMIZATION,SHARED-QUERY-OPTIMIZATION-2026-09-27,STATE-SHARING-2026-09-27,ENCODING-REUSE-2026-09-29}.md`、`research/TS-ENGINE-PERF-2026-09-27.md`。原文均可从上述固定树按原路径读取；仍被当前说明引用的链接已改为固定提交。
- 删除 `scripts/training/performance/resident/{budget_repair,update_cost}.py`：前者只迁移已登记旧二进制的单次账本，后者只测本轮一次更新成本。没有运行、CI、打包或测试调用；通用测量、报告汇总、确定性恢复和权威拒绝复现继续维护。
- 保留 `src/ai/difficulty.ts`、`planning/`、`training/teacher.ts` 及全部真实教师依赖。保留 CI 的规则/编码/哈希差分、源码包、晚盘夹具、研究完整记录入口。历史搜索工具存在共享依赖或测试调用，不能按目录名整批删除。
- 保留作者反馈、规则、公开协议、历史指纹、关键失败报告及实际基准证据。未触碰未跟踪文件、模型、原始数据和外部导入件。

学习路线复核：[随机游戏研究原文](https://github.com/Loremmlel/haojie2/blob/0690c93fc1d8caeb5f533240ecdedb026942ad7f/docs/ai/research/STOCHASTIC-GAMES-2026-09-26.md)。其核心结论仍有效：真实终局收益可以提供学习信号，但自模仿、loss下降和少量幸运获胜都不足以证明棋力。对当前/历史对手自对弈与收益驱动策略更新必须分别验收。
