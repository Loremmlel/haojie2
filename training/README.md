# 浩劫原生训练

唯一运行入口是独立 `haojie-train` C++ / LibTorch 可执行文件；Rust 保留规则。训练、推理、初始化、调度、优化器和保存恢复均不依赖 Python。

```powershell
pwsh -File training/native/build.ps1 -LibTorch E:/libtorch
pwsh -File training/native/run.ps1 configure
pwsh -File training/native/run.ps1 run --directory E:/Training/haojie/run
pwsh -File training/native/run.ps1 status --directory E:/Training/haojie/run
pwsh -File training/native/run.ps1 resize --directory E:/Training/haojie/run --concurrency 8
pwsh -File training/native/run.ps1 stop --directory E:/Training/haojie/run
pwsh -File training/native/run.ps1 run --directory E:/Training/haojie/restored --resume E:/Training/haojie/run/checkpoint.pt
```

构建可复用本机已有的 LibTorch；正常训练根据 CPU、显卡和空闲显存自动选择配置。零并发会停放全部在途局；扩容优先恢复原局。停止与定时保存都保留进度和未消费轨迹，恢复使用新的空目录。

Linux 构建、自动配置、模型和恢复命令见[运行说明](native/README.md)，云端使用 [Colab 笔记本](colab/train.ipynb)，检查方法见[功能验证](native/VALIDATION.md)。旧 Python 包和训练格式不再提供入口。
