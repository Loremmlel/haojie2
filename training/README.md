# 浩劫原生 V-trace 训练

唯一训练入口为 `python -m haojie_training.online`，安装后的等价命令为 `haojie-train`。模型、轨迹和检查点使用新契约；旧 MC-Q、教师预训练、Web 控制台及旧模型兼容路径已移除。

本地验收检查正确性、控制恢复和新旧架构吞吐；棋力留到 Colab 运行几十小时后评估。当前实测见[验收记录](online/VALIDATION.md)，模型、运行时和控制命令见[运行说明](online/README.md)。

## 安装

需要 Python 3.12、Rust 1.98.1 和对应平台链接器。CUDA 训练要求原生 BF16 支持。

```sh
python -m venv training/.venv
# Windows 使用 training/.venv/Scripts/python.exe；Linux 使用 training/.venv/bin/python
python -m pip install ./training
cargo build --locked --release --manifest-path native/engine/Cargo.toml --target-dir artifacts/native-target
python -m haojie_training.online run --directory artifacts/root-vtrace/my-run --concurrency 16 --workers 2
```

在激活的新环境中执行以上 `python` 命令。共享库默认从仓库 `artifacts/native-target/release` 加载；独立安装包使用 `HAOJIE_NATIVE_LIB` 指向 `haojie_native.dll` 或 `libhaojie_native.so`。训练热路径只依赖 Python、PyTorch 和进程内 Rust 原生线程。

## 控制与恢复

```sh
python -m haojie_training.online status --directory artifacts/root-vtrace/my-run
python -m haojie_training.online resize --directory artifacts/root-vtrace/my-run --concurrency 64
python -m haojie_training.online save --directory artifacts/root-vtrace/my-run
python -m haojie_training.online stop --directory artifacts/root-vtrace/my-run
python -m haojie_training.online run --directory artifacts/root-vtrace/restored --resume artifacts/root-vtrace/my-run/checkpoint.pt
```

默认每 300 秒后台保存；`--save-seconds 0` 关闭定时保存，仍可手动保存。停止保存未完成局；缩容停止补局，让已接纳局自然完成。窗口指标在保存收尾前冻结。浏览器和控制客户端退出不会结束训练所有者进程；磁盘恢复只从已完成写入的检查点继续。

[Colab Notebook](notebooks/root-vtrace.ipynb)提供安装、Drive 持久化、控制与恢复入口，L4 实机验证尚未执行。已有模型和原始实验目录保留，旧权重不转换为新架构。
