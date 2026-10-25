# 独立 C++ / LibTorch 训练宿主

`haojie-train` 负责模型、推理、学习、调度与检查点；Rust lane 负责规则和逐局 actor。训练运行不启动 Python 或 Node。

## 构建与启动

需要 C++20、CMake 3.24+、Rust 1.98.1 和 LibTorch。Windows 使用 MSVC，构建入口可复用 `training/.venv/Lib/site-packages/torch`。CUDA Runtime/CRT 头包由入口从 NVIDIA 下载并校验；运行时使用 LibTorch 和 NVRTC。

```powershell
pwsh -File training/native/build.ps1
pwsh -File training/native/run.ps1 configure
pwsh -File training/native/run.ps1 run --directory E:/Training/haojie/run
```

```sh
export LIBTORCH=/opt/libtorch
bash training/native/build.sh
bash training/native/run.sh configure
bash training/native/run.sh run --directory /data/haojie/run
```

运行目录必须不存在或为空。正常训练默认 `--device auto`；按进程可用 CPU 核数、GPU 的 SM 数及空闲显存选配置。`configure` 只打印 JSON，不创建模型或开始训练；`run` 启动时再次检测并写入 `configuration.json`。可用 `--device`、`--batch`、`--concurrency`、`--workers`、`--inference-batch`、`--threads`、`--memory-mb` 覆盖对应项。

GPU 空闲显存达到 4/6/10/18 GiB 时，起始学习 batch 分别为 64/128/256/512，推理上限为 8/16/32/64；低于 4 GiB 使用 batch 32。lane 至多 16 个且不超过物理核数，并发同时受 lane 数、SM 数和推理上限约束。宿主内存预算取可用内存的三分之一，最多 8 GiB。CPU 模式用 batch 32，线程数至多 8。它们是无需试训的起始档位；实际 token 长度及其他进程占用变化仍会影响峰值显存。

轨迹块默认 16、发布周期默认 8 次更新。`--samples`、`--updates`、`--seconds` 分别限制实际消费、更新次数和运行时间；不指定上限则持续运行。

## Colab

打开 [训练笔记本](../colab/train.ipynb)，选择 L4、A100 或更新的原生 BF16 GPU，依次运行安装与配置单元。安装固定使用 PyTorch/LibTorch 2.14.0 cu126 及匹配的 CUDA 头文件；Python 仅用于安装和笔记本控制。T4/P100 会明确拒绝启动，保持 BF16 全链路。

检查配置后把 `START_TRAINING` 改为 `True` 才会启动。训练与检查点直接写入 Google Drive，控制单元支持状态、保存、停止和调整并发。重连后填入 `RESUME`，恢复到新目录。源码和编译缓存留在 `/content`，运行时回收后可重新安装。`REVISION` 必须是已推送到远端的分支或提交。Colab 的 GPU 分配和运行时寿命由服务决定，见 [Colab FAQ](https://research.google.com/colaboratory/faq.html)。

## 控制与恢复

```powershell
pwsh -File training/native/run.ps1 status --directory E:/Training/haojie/run
pwsh -File training/native/run.ps1 resize --directory E:/Training/haojie/run --concurrency 8
pwsh -File training/native/run.ps1 save --directory E:/Training/haojie/run
pwsh -File training/native/run.ps1 stop --directory E:/Training/haojie/run
pwsh -File training/native/run.ps1 run --directory E:/Training/haojie/restored --resume E:/Training/haojie/run/checkpoint.pt
```

默认每 300 秒保存，`--save-seconds 0` 关闭定时保存。保存先停在一致边界，冻结模型、五组 BF16 优化器状态、权威局面、两类 RNG、路径、排除 frontier、未决转换、部分轨迹和 ready 块，再后台原子写入。恢复沿用档内训练语义；并发、worker、推理上限、设备和宿主内存预算可以覆盖。

缩容在当前规则操作完成后停放并释放规则树与缓存；扩容先恢复旧局。并发可调到零，停放局仍可保存。恢复保留档内学习 batch 和并发（包括零并发），设备、lane、线程、推理上限与宿主内存预算重新检测。保留并发可避免自动停放尚有未决后继的旧局；需要调整时显式指定 `--concurrency`。

## 训练与资源边界

模型固定为 384 宽、6 层、6 头、FFN 1536，共 15,184,130 参数。模型只接收公开类型化输入。模型、激活、梯度、保存的统计和五组优化器状态为 BF16；算子内部累加可使用 FP32。完整约束见[架构决策](architecture/DECISION.md)。

同席位连续短轨迹使用真实后继 bootstrap 和 V-trace，折扣为 1，只有真实终局产生奖励。行为概率来自实际 BF16 质量的规范分布。正式路径不使用参考行为或诊断结果。

严格策略滞后上限 K=64：行为开始前预留容量，未决、部分、ready、预取和在途合计至多八个 batch；更新 48 次后排空并发布。消费前校验版本差。停放局的真实未决项不会伪造后继；排空可能等待该局恢复。历史积压无法满足 K 时拒绝启动并保留原档。

CPU lane 公平轮转，根请求由两个 GPU 槽集中处理；缓存、迟到结果及 CPU 头绑定实际快照版本。图按配置捕获，最多 128 根/8192 token，超容量执行完整普通前向。回退扩容不会替换图的捕获输出地址。学习预取下一批输入，只有一个优化更新在途；设备完整验证后原子提交。

`--memory-mb` 管理宿主轨迹与缓存，CUDA 分配器另使用物理显存的 90% 预算。固定转换数不等于固定 token 数；不得把本机容量经验当作所有训练卡的性能结论。

`status.json` 为最新快照；`window.json` 在退出排空前冻结；`saved.json` 为保存后的累计状态。`consumed_within_lag` 与 `consumed_lag_max` 报告 K 内消费及实际最大滞后。恢复后的速率重新计时。消费守恒为 `produced = consumed + ready + partial + learning`。

功能检查见[验证入口](VALIDATION.md)。独立的固定工作集与分解测量工具见[诊断入口](diagnostics/README.md)，不作为正常训练入口。
