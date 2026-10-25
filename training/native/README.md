# 独立 C++ / LibTorch 训练宿主

`haojie-train` 负责模型、推理、学习、调度与检查点；Rust lane 负责规则和逐局 actor。训练运行不启动 Python 或 Node。

## 构建与启动

需要 C++20、CMake 3.24+、Rust 1.98.1 和 LibTorch。Windows 使用 MSVC，构建入口可复用 `training/.venv/Lib/site-packages/torch`。CUDA Runtime/CRT 头包由入口从 NVIDIA 下载并校验；运行时使用 LibTorch 和 NVRTC。

```powershell
pwsh -File training/native/build.ps1
pwsh -File training/native/run.ps1 run --directory E:/Training/haojie/run
```

```sh
export LIBTORCH=/opt/libtorch
bash training/native/build.sh
artifacts/cpp-training-build/haojie-train run --directory /data/haojie/run
```

运行目录必须不存在或为空。默认学习 batch 512、轨迹块 16、发布周期 8 次更新；可用 `--batch`、`--concurrency`、`--workers`、`--inference-batch`、`--memory-mb` 覆盖配置。`--samples`、`--updates`、`--seconds` 分别限制实际消费、更新次数和运行时间；不指定上限则持续运行。

## 控制与恢复

```powershell
pwsh -File training/native/run.ps1 status --directory E:/Training/haojie/run
pwsh -File training/native/run.ps1 resize --directory E:/Training/haojie/run --concurrency 8
pwsh -File training/native/run.ps1 save --directory E:/Training/haojie/run
pwsh -File training/native/run.ps1 stop --directory E:/Training/haojie/run
pwsh -File training/native/run.ps1 run --directory E:/Training/haojie/restored --resume E:/Training/haojie/run/checkpoint.pt
```

默认每 300 秒保存，`--save-seconds 0` 关闭定时保存。保存先停在一致边界，冻结模型、五组 BF16 优化器状态、权威局面、两类 RNG、路径、排除 frontier、未决转换、部分轨迹和 ready 块，再后台原子写入。恢复沿用档内训练语义；并发、worker、推理上限、设备和宿主内存预算可以覆盖。

缩容在当前规则操作完成后停放并释放规则树与缓存；扩容先恢复旧局。并发可调到零，停放局仍可保存。旧档的学习 batch 不随恢复硬件自动更改。

## 训练与资源边界

模型固定为 384 宽、6 层、6 头、FFN 1536，共 15,184,130 参数。模型只接收公开类型化输入。模型、激活、梯度、保存的统计和五组优化器状态为 BF16；算子内部累加可使用 FP32。完整约束见[架构决策](architecture/DECISION.md)。

同席位连续短轨迹使用真实后继 bootstrap 和 V-trace，折扣为 1，只有真实终局产生奖励。行为概率来自实际 BF16 质量的规范分布。正式路径不使用参考行为或诊断结果。

严格策略滞后上限 K=64：行为开始前预留容量，未决、部分、ready、预取和在途合计至多八个 batch；更新 48 次后排空并发布。消费前校验版本差。停放局的真实未决项不会伪造后继；排空可能等待该局恢复。历史积压无法满足 K 时拒绝启动并保留原档。

CPU lane 公平轮转，根请求由两个 GPU 槽集中处理；缓存、迟到结果及 CPU 头绑定实际快照版本。图按配置捕获，最多 128 根/8192 token，超容量执行完整普通前向。回退扩容不会替换图的捕获输出地址。学习预取下一批输入，只有一个优化更新在途；设备完整验证后原子提交。

`--memory-mb` 管理宿主轨迹与缓存，CUDA 分配器另使用物理显存的 90% 预算。固定转换数不等于固定 token 数；不得把本机容量经验当作所有训练卡的性能结论。

`status.json` 为最新快照；`window.json` 在退出排空前冻结；`saved.json` 为保存后的累计状态。`consumed_within_lag` 与 `consumed_lag_max` 报告 K 内消费及实际最大滞后。恢复后的速率重新计时。消费守恒为 `produced = consumed + ready + partial + learning`。

功能检查见[验证入口](VALIDATION.md)。独立的固定工作集与分解测量工具见[诊断入口](diagnostics/README.md)，不作为正常训练入口。
