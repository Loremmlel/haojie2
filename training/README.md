# 浩劫原生训练

唯一运行入口是独立 `haojie-train` C++ / LibTorch 可执行文件；Rust 保留规则。训练、推理、初始化、调度、优化器和保存恢复均不依赖 Python。

```powershell
pwsh -File training/native/build.ps1 -LibTorch E:/libtorch
pwsh -File training/native/run.ps1 run --directory artifacts/native/run --concurrency 64 --workers 8
pwsh -File training/native/run.ps1 status --directory artifacts/native/run
pwsh -File training/native/run.ps1 resize --directory artifacts/native/run --concurrency 256
pwsh -File training/native/run.ps1 stop --directory artifacts/native/run
pwsh -File training/native/run.ps1 run --directory artifacts/native/restored --resume artifacts/native/run/checkpoint.pt
```

构建可复用本机已有的 LibTorch。零并发会停放全部在途局；扩容优先恢复原局。停止与定时保存都保留进度和未消费轨迹，恢复使用新的空目录。

Linux 构建、模型、BF16 类型、轨迹、内存预算和验证命令见[运行说明](native/README.md)，本机结果与剩余瓶颈见[验收记录](native/VALIDATION.md)。旧模型、优化器与原始实验文件保留，旧 Python 包和训练格式不再提供入口。
