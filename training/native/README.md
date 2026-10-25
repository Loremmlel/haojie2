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

正式训练只消费完整终局，`batch` 是计算小批的转换数，不是轨迹截断长度。一个完整对局批汇总梯度后提交一次更新并发布；`chunk=0` 表示保留整局。`--samples`、`--updates`、`--seconds` 分别限制实际消费、更新次数和运行时间；消费上限在整局更新后检查，可能越过样本配额，不截断对局。不指定上限则持续运行。

## Colab

打开 [训练笔记本](https://colab.research.google.com/github/Loremmlel/haojie2/blob/main/training/colab/train.ipynb)，选择 L4、A100 或更新的原生 BF16 GPU，依次点击三个单元的运行按钮，也可选择“全部运行”。安装默认拉取 `main` 最新版，固定使用 PyTorch/LibTorch 2.14.0 cu126 及匹配的 CUDA 头文件；Python 仅用于安装和笔记本控制。T4/P100 会明确拒绝启动，保持 BF16 全链路。

无需编辑代码：默认“从头开始完整终局训练”，下拉框同时列出 Drive 中的检查点。后续可选择本版本的完整终局检查点续训；旧短轨迹档保留供检查与测评，不混入新训练。硬件配置自动生成，点击“启动训练”才创建进程，并自动展示 Web 控制台；“全部运行”只准备界面。训练与检查点写入新 Google Drive 目录，原档保留。源码和编译缓存留在 `/content`，运行时回收后可重新安装。Colab 的 GPU 分配和运行时寿命由服务决定，见 [Colab FAQ](https://research.google.com/colaboratory/faq.html)。

最后一个单元展示嵌入式 Web 统计面板，通过 Colab 官方回调读取原生状态，提供吞吐、最近总损失、积压、滞后和 CPU/GPU/内存曲线，以及保存、停止和并发控制。每 5 秒采样，保留最近 360 点；损失只来自已提交的真实学习步，恢复后首个新更新前留空。界面关闭或暂停刷新不影响训练；重新展示同一运行保留观测历史，旧面板失效。保存完成以 `saved.json` 更新为准，连接中断保留旧读数并显示状态未知。

面板提供立即测评、取消测评和后台定期测评，默认间隔 30 分钟，0 关闭。每组用固定种子对战现有简单／普通／困难 AI，经典／神龛分别交换先后手，共 12 局；默认每局 120 秒，可调整组数与时限。先等待完整检查点，再由独立原生进程一次性加载已提交权重，整轮固定模型版本；测评没有优化器，也不向训练写入轨迹。Python 调度及嵌入式 V8 只负责运行现有 TS 对手，Rust 执行正式规则，模型继续使用原生 BF16 推理。测评与训练共享 CPU/GPU，每次最多一轮，不积压定期任务。

测评显示模型更新号、逐局进度、胜负平、完成数、未完成／错误和真实终局胜率，记录保存到运行目录的 `evaluations/`。截断、超时、取消保持未完成；失败单列。关闭页面后定期调度继续，Colab 内核回收后停止；“保存并停止”同时取消测评和关闭定期计划。重连后从下拉框选择检查点续训到新目录。原生 `evaluate --checkpoint PATH --device cuda` 是由测评执行器使用的行协议入口。

更新正在运行的 Colab 时，先点“保存并停止”，等进程正常退出，再依次重新运行三个单元并点“启动训练”；无需重启 Colab 运行时。本次迁移保持默认从头开始；之后续训时选择刚保存的完整终局检查点。仅重开面板时运行最后一个单元，不会再创建训练进程。刷新页面不会拉取代码，也不会替换正在运行的程序。

## 控制与恢复

```powershell
pwsh -File training/native/run.ps1 status --directory E:/Training/haojie/run
pwsh -File training/native/run.ps1 resize --directory E:/Training/haojie/run --concurrency 8
pwsh -File training/native/run.ps1 save --directory E:/Training/haojie/run
pwsh -File training/native/run.ps1 stop --directory E:/Training/haojie/run
pwsh -File training/native/run.ps1 run --directory E:/Training/haojie/restored --resume E:/Training/haojie/run/checkpoint.pt
```

默认每 300 秒保存，`--save-seconds 0` 关闭定时保存。保存先停在一致边界，冻结模型、五组 BF16 优化器状态、权威局面、两类 RNG、路径、排除 frontier、未决转换、部分轨迹和 ready 块，再后台原子写入。恢复沿用档内训练语义；并发、worker、推理上限、设备和宿主内存预算可以覆盖。

每次 `checkpoint.pt` 原子落盘后立即发布对应冻结状态的 `saved.json`，作为面板和后台测评的保存回执；运行中手动保存、定时保存与停止保存共用同一路径。

训练恢复与评测均不比较构建指纹。v3 共享表格式保留；正式恢复要求 `chunk=0` 的完整终局轨迹，旧短轨迹档明确提示从头开始，仍可用于冻结测评。源检查点只读，新保存写入新运行目录。模型尺寸和 BF16 类型检查保留。

缩容在当前规则操作完成后停放并释放规则树与缓存；扩容先恢复旧局。并发可调到零，停放局仍可保存，ready 对局继续学习。恢复保留档内学习 batch 和并发（包括零并发），设备、lane、线程、推理上限与宿主内存预算重新检测；需要调整并发时显式指定 `--concurrency`。

## 训练与资源边界

模型固定为 384 宽、6 层、6 头、FFN 1536，共 15,184,130 参数。模型只接收公开类型化输入。模型、激活、梯度、保存的统计和五组优化器状态为 BF16；算子内部累加可使用 FP32。完整约束见[架构决策](architecture/DECISION.md)。

同席位的全部决策保留到真实终局，折扣为 1，只有胜/负/平产生 +1/-1/0 奖励。未终局、停放与中断轨迹保留原状态，不进入学习。冻结当前 learner 参数计算整局 V-trace 目标，反向递推跨越计算小批；随后逐小批反向并用 BF16 补偿汇总梯度，完整验证后只提交一次 AdamW 更新。行为概率来自实际 BF16 质量的规范分布。算法依据与边界见[架构决策](architecture/DECISION.md)。

完整终局按完成顺序异步进入学习，不等待长局或停放局。每次取完整对局直到转换数达到 `batch`，不足也立即提交学习；单局超过 `batch` 时按计算小批汇总一次梯度更新。学习期间照常补局，ready 对局数达到目标并发时才限制补局，已有局继续行动。

实际消费仍检查策略滞后上限 K=64。完整对局中任一行为版本超过上限，则整局跳过学习，记录为 `stale_games/stale_steps`，继续消费后面的新鲜对局。没有按更新次数触发的全局等待，没有长局计时截断；停放局恢复后继续真实对局，最终是否可学习按实际版本差决定。

CPU lane 公平轮转，根请求由两个 GPU 槽集中处理；缓存、迟到结果及 CPU 头绑定实际快照版本。推理图按配置捕获，最多 128 根/8192 token，超容量执行完整普通前向。完整终局学习按计算小批执行普通前向/反向，只有一个优化更新在途；旧短块学习图只供固定工作集诊断。

`--memory-mb` 管理宿主轨迹与缓存，CUDA 分配器另使用物理显存的 90% 预算。固定转换数不等于固定 token 数；不得把本机容量经验当作所有训练卡的性能结论。

`status.json` 为最新快照；`window.json` 在退出排空前冻结；`saved.json` 为保存后的累计状态。`consumed_within_lag` 与 `consumed_lag_max` 报告 K 内消费及实际最大滞后；`stale_games/stale_steps` 单列过期未学习的数据。恢复后的速率与窗口计数重新计时，检查点保留累计过期计数。累计守恒为 `produced = consumed + stale_steps + ready + partial + learning`。

`training_method=terminal-vtrace` 标识正式方法；`terminal_games_consumed` 是本次进程已学习的终局数。`latest_value_mse` 对照真实终局结果，`latest_policy_entropy` 和 `latest_gradient_norm` 用于检查退化；这些原始指标可在控制台 JSON 中查看。首次终局前 updates/consumed 为零属于正常采集阶段。完整更新期间保存/停止等待原子边界；宿主轨迹超预算时先保存并退出，保留未结束局。

功能检查见[验证入口](VALIDATION.md)。独立的固定工作集与分解测量工具见[诊断入口](diagnostics/README.md)，不作为正常训练入口。
