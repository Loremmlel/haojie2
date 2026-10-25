# AI 与训练状态

训练使用独立 C++ / LibTorch 宿主与 Rust 规则 actor；运行不依赖 Python 或 Node。主干固定为 384 宽、6 层、6 头、FFN 1536，完整保留 BF16、公开信息、真实行为概率与检查点语义。

当前具备严格 K=64 策略滞后准入、学习输入预取、集中根服务、在线停放/恢复和原子保存。性能实验已停止，主线转入真实训练入口、硬件自动配置与 Colab 适配。尚未在 Colab 真机验收，也未启动本机正式训练。

- [训练入口](../../training/README.md)
- [运行与恢复](../../training/native/README.md)
- [架构约束](../../training/native/architecture/DECISION.md)
- [功能验证](../../training/native/VALIDATION.md)
- [TS/Rust 对应](RUNTIME.md)
