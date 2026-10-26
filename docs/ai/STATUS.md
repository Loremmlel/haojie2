# AI 与训练状态

网页继续使用三档手工 AI；训练模型尚未接入网页。训练主线为 C++ / LibTorch 宿主与 Rust 规则 actor，384 宽、6 层、6 头、FFN 1536，保留 BF16、公开信息和真实行为概率。

正式训练只学习完整终局 V-trace；`batch` 控制计算小批，不截断对局。模型、五组优化器、未完成局、未消费轨迹及两类 RNG 随检查点恢复；完整终局 v3 可续训，旧短块档只供测评。

MCTS 默认关闭，`--search-simulations 0` 直接使用策略网络，正数启用搜索。历史对手池独立保留，默认 80% 当前／20% 历史、8 份模型、每 64 次更新入池，历史对手整局固定。Colab 模拟数默认 0，续训也按当前选择覆盖；评测读取冻结检查点的模式和搜索配置。

长局、策略滞后及内存压力按整局淘汰，不补胜负或训练前缀；完整 ready 和学习中轨迹优先保留。已知空根异常按单局留档、淘汰并补局。TS/Rust 保留相同规则、合法支持集及精确空前缀过滤。

- [安装、Colab、运行与恢复](../../training/native/README.md)
- [模型、终局学习与调度约束](../../training/native/architecture/DECISION.md)
- [维护验证命令](../../training/native/VALIDATION.md)
- [TS/Rust 内核与搜索对应](RUNTIME.md)
- [记录审核与差分工具](TRAINING.md)
- [历史实测、失败记录和旧版本](HISTORY.md)
