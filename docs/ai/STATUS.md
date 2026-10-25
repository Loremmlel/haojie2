# AI 与训练状态

训练使用独立 C++ / LibTorch 宿主与 Rust 规则 actor；运行不依赖 Python 或 Node。主干固定为 384 宽、6 层、6 头、FFN 1536，完整保留 BF16、公开信息、真实行为概率与检查点语义。

正式训练改为完整终局 V-trace：真实胜负沿整局反向递推，按显存计算小批汇总梯度，一次提交更新。修复 BF16 策略归一化的伪梯度与 tanh 输出舍入导致的过早零梯度。严格 K=64 通过限制新局与整轮自然终局排空维持；停放、保存和恢复保留未完成轨迹。原短轨迹检查点保留供测评，新训练从头初始化。

Colab 默认拉取 main、从头开始，按 CPU/GPU/空闲显存自动配置；保留 Web 面板、Drive 保存恢复、停止和手工／定期 AI 测评。本地已验证真实终局学习和数值修复，原始验收数见[功能验证](../../training/native/VALIDATION.md)。更新后的 L4 实际训练由 Colab 启动，性能实验未重启。

- [训练入口](../../training/README.md)
- [运行与恢复](../../training/native/README.md)
- [架构约束](../../training/native/architecture/DECISION.md)
- [功能验证](../../training/native/VALIDATION.md)
- [TS/Rust 对应](RUNTIME.md)
