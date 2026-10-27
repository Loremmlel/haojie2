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

运行目录必须不存在或为空。正常训练默认 `--device auto`；按进程可用 CPU 核数、GPU 的 SM 数及空闲显存选配置。`configure` 只打印 JSON，不创建模型或开始训练；`run` 启动时再次检测并写入 `configuration.json`。可用 `--device`、`--batch`、`--concurrency`、`--workers`、`--inference-batch`、`--threads`、`--memory-mb`、`--max-game-steps` 覆盖对应项。

GPU 空闲显存达到 4/6/10/18 GiB 时，起始学习 batch 分别为 64/128/256/512，推理上限为 8/16/32/64；低于 4 GiB 使用 batch 32。lane 至多 16 个且不超过物理核数，并发同时受 lane 数、SM 数和推理上限约束。宿主内存预算取进程实际可用内存的三分之一，不再封顶 8 GiB；容器仍受 cgroup 配额约束，余量留给系统、模型和检查点序列化。CPU 模式用 batch 32，线程数至多 8。它们是无需试训的起始档位；实际 token 长度及其他进程占用变化仍会影响峰值显存。

正式训练只消费完整终局，`batch` 是计算小批的转换数，不是轨迹截断长度。一个完整对局批汇总梯度后提交一次更新并发布；`chunk=0` 表示保留整局。`--samples`、`--updates`、`--seconds` 分别限制实际消费、更新次数和运行时间；消费上限在整局更新后检查，可能越过样本配额，不截断对局。不指定上限则持续运行。

`--max-game-steps` 默认 262144，按己方已完成转换计数；达到上限仍未终局就整局淘汰，完整终局超过该上限也不学习。未结束局最早行为版本超过 K=64 时提前淘汰。宿主管理内存达到预算 75% 后，在一致边界优先回收最长的未结束局，回到半预算后恢复准入；停放局同样参与回收。完整终局队列和在学习数据保留，若它们暂时独占预算则暂停采样，等学习释放内存再恢复。淘汰局不补胜负、不截取前后缀学习，后续新局正常补入。

状态 JSON 的 `discarded_games` / `discarded_steps` 与 `discarded_long_games` / `discarded_stale_games` / `discarded_memory_games` 单独记录淘汰总量及原因；不计入 `completed_games` 或 `terminal_games_consumed`，已经真实终局但因长度上限被拒收的局仍属于 `completed_games`。`memory_backpressure` 标识完整终局占用导致的采样暂停。`produced = consumed + stale_steps + discarded_steps + ready_steps + partial_steps + learning_steps`，未决转换另计。

## 搜索与对手

正式训练默认直接使用策略网络，`--search-simulations 0` 关闭 MCTS；设为正整数（例如 `--search-simulations 64`）开启搜索。关闭时只计算当前公开节点的原策略，不构造模拟分支或生成搜索访问标签，历史对手池照常使用。冻结评测读取检查点的模拟数，缺失时默认关闭；`evaluate --search-simulations 0` 可显式关闭旧档的搜索。

开启后使用完整命令级随机 MCTS：网络提供参数先验和价值，PUCT 选择命令，执行公开规则后继续搜索本方连续操作、对手行动和反应。最深默认 64 个决策边；参数补全不冒充对抗搜索深度。渐进扩宽从未裁剪的动作树采样完整命令，机会后继使用独立 xorshift 场景，不读取正式 RNG。每个参数节点使用 95% 条件访问分布加 5% 原策略分布，再按真实 BF16 质量采样并保存实际行为概率。

同一命令的参数选择复用访问数，并固定该次搜索的模型；探索到尚未访问的前缀时再搜索。非终局叶使用网络价值估计，搜索截止不会写成平局或终局奖励。神龛暗选阶段保留公开策略采样，不补造对手的秘密选择；揭示后的神龛对局照常搜索。单次命令补全最多请求 4096 次非强制参数采样，超过时退回完整原策略，计入 `search_parameter_fallbacks`；假想分支遇到空根同样回退原策略，计入 `search_empty_fallbacks`，两者均不丢弃实局。

搜索访问分布作为额外策略交叉熵，随完整终局一起学习；价值和策略梯度仍按真实终局做整局 V-trace，只有真实终局才能入队。关闭时 `training_method=terminal-vtrace`，启用时为 `terminal-vtrace-mcts`；恢复后已有轨迹的搜索标签原样保留并参与原定学习。原模型结构、15,184,130 个参数及 BF16 优化器保持不变。TS 参照入口为 `src/ai/training/command-search.ts`，原生入口为 `src/actor/search.cpp`，共用 PUCT、渐进扩宽、随机后继、同方连续行动符号及条件访问分布流程。

对手默认按新局编号分层安排 80% 当前自对弈、20% 历史对手；没有更早模型时先用当前策略。历史池默认保留 8 份、每 64 次更新加入一份，从比当前版本更早的池成员中均匀选择。历史对手开局选定后整局固定；当前策略在下一条完整命令更新版本。训练席位继续交替先后手，仅该席位的轨迹参与学习。80/20 参考 [OpenAI Five 论文附录 N](https://cdn.openai.com/dota-2.pdf)，是实验起点。

| 参数                   | 默认 | 含义                                 |
| ---------------------- | ---- | ------------------------------------ |
| `--search-simulations` | 0    | 0 关闭搜索，正整数指定每根模拟次数   |
| `--search-depth`       | 64   | 完整命令／可选中断决策边的深度上限   |
| `--history-size`       | 8    | 池内历史模型数上限                   |
| `--history-interval`   | 64   | 历史模型发布间隔，按学习更新计       |
| `--history-percent`    | 20   | 新局使用历史对手的目标百分比，0 关闭 |

这些参数写入配置与检查点，恢复时默认沿用，也可显式覆盖。缩小历史池不会更换已有对局的对手。`search_simulations`、`search_expanded`、`search_max_depth`、`search_opponent_nodes`、`search_model_queries` 和 `historical_games_admitted` 记录本次运行的实际搜索及历史对手准入；`history_pool_size` 是当前池规模。搜索推理的编码图、token 与 CPU 解码时间计入原有资源指标。

已知的 `no legal action at root` 只隔离故障局，保存到运行目录 `failed-games/game-ID.json`，整局轨迹淘汰并计入 `discarded_error_games`，随后补开新局。故障局不计胜负、不补奖励；设备、协议及其他未识别错误仍按原有失败保存流程上报。诊断文件包含规则恢复状态，只保存在运行目录，不作为模型输入。

## Colab

打开 [训练笔记本](https://colab.research.google.com/github/Loremmlel/haojie2/blob/main/training/colab/train.ipynb)，选择 L4、A100 或更新的原生 BF16 GPU，依次点击三个单元的运行按钮，也可选择“全部运行”。安装默认拉取 `main` 最新版，固定使用 PyTorch/LibTorch 2.14.0 cu126 及匹配的 CUDA 头文件；Python 仅用于安装和笔记本控制。T4/P100 会明确拒绝启动，保持 BF16 全链路。

无需编辑代码：默认“从头开始完整终局训练”，下拉框同时列出 Drive 中的检查点。“MCTS 模拟数”默认 0，恢复旧档也会显式覆盖为当前选择；要启用可填 64 等正整数。命令行恢复默认沿用档内搜索预算，关闭旧档搜索须传 `--search-simulations 0`。后续可选择本版本的完整终局检查点续训；旧短轨迹档保留供检查与测评，不混入新训练。硬件配置自动生成，点击“启动训练”才创建进程，并自动展示 Web 控制台；“全部运行”只准备界面。训练与检查点写入新 Google Drive 目录，原档保留。源码和编译缓存留在 `/content`，运行时回收后可重新安装。Colab 的 GPU 分配和运行时寿命由服务决定，见 [Colab FAQ](https://research.google.com/colaboratory/faq.html)。

最后一个单元展示嵌入式 Web 统计面板，通过 Colab 官方回调读取原生状态，提供吞吐、最近总损失、积压、运行最大消费滞后和 CPU/GPU/内存曲线，以及保存、停止和并发控制。后台每秒采集资源，每 5 秒记录观测，最多保留 720 点；页面每 5 秒读取历史，关闭或暂停页面后仍持续采样。吞吐与利用率默认显示最近 30 秒均值，可叠加原始波动，时间范围可选最近 10/30/60 分钟；损失按观测到的更新去重，叠加 EMA 趋势与原始线，恢复后首个新更新前留空。Ready 使用阶梯线，内存保留原始折线。缺测留白、标示原因和时长，恢复后重建速率基线；训练状态延迟期间资源继续采集。

历史仅存于当前 Colab 内核内存，内核回收后清空；重新展示同一运行保留曲线，旧面板失效。保存完成以 `saved.json` 更新为准，连接中断保留旧读数并显示状态未知。页面展示的状态文件年龄与后台采样时间独立，避免将页面读取时间当成数据更新时间。

面板提供立即测评、取消测评和后台定期测评，默认间隔 30 分钟，0 关闭。手动与定期测评均从冻结检查点读取训练模式：经典档只测经典，神龛档只测神龛。每轮重新生成随机种子，每组用同一个对局种子对战现有简单／普通／困难 AI，交换先后手，共 6 局；默认每局 120 秒，可调整组数与时限。测评温度默认 0.5，对每个参数节点的最终动作分布按 `p^(1/T)` 重新归一化；0 为逐节点贪心选招，1 为原概率采样，面板支持 0–2。温度只影响测评执行，不改变训练或搜索先验，设置从下一轮生效。先等待完整检查点，再由独立原生进程一次性加载已提交权重与训练配置，整轮固定模型版本和规则模式；测评没有优化器，也不向训练写入轨迹。Python 调度及嵌入式 V8 只负责运行现有 TS 对手，Rust 执行正式规则，模型继续使用原生 BF16 推理。测评与训练共享 CPU/GPU，每次最多一轮，不积压定期任务。

测评显示模型更新号、逐局进度、胜负平、完成数、未完成／错误和真实终局胜率，记录保存到运行目录的 `evaluations/`，包含本轮主种子、逐局种子和实际温度。命令行测评执行器 `python -B -m training.evaluation.run` 支持 `--seed` 重放同一测评安排及 `--temperature` 覆盖默认温度。截断、超时、取消保持未完成；失败单列。关闭页面后定期调度继续，Colab 内核回收后停止；“保存并停止”同时取消测评和关闭定期计划。重连后从下拉框选择检查点续训到新目录。原生 `evaluate --checkpoint PATH --device cuda --temperature 0.5` 是由测评执行器使用的行协议入口。

更新正在运行的 Colab 时，先点“保存并停止”，等进程正常退出，再依次重新运行三个单元并点“启动训练”；无需重启 Colab 运行时。本次迁移保持默认从头开始；之后续训时选择刚保存的完整终局检查点。仅重开面板时运行最后一个单元，不会再创建训练进程。刷新页面不会拉取代码，也不会替换正在运行的程序。

## 控制与恢复

```powershell
pwsh -File training/native/run.ps1 status --directory E:/Training/haojie/run
pwsh -File training/native/run.ps1 resize --directory E:/Training/haojie/run --concurrency 8
pwsh -File training/native/run.ps1 save --directory E:/Training/haojie/run
pwsh -File training/native/run.ps1 stop --directory E:/Training/haojie/run
pwsh -File training/native/run.ps1 run --directory E:/Training/haojie/restored --resume E:/Training/haojie/run/checkpoint.pt
```

默认每 300 秒保存，`--save-seconds 0` 关闭定时保存。保存先停在一致边界，冻结模型、五组 BF16 优化器状态、权威局面、两类 RNG、路径、排除 frontier、未决转换、部分轨迹和 ready 块，再后台原子写入。历史池、仍被对局引用的已退池模型、对手绑定和搜索访问标签一并保存。保存边界两侧都清空临时搜索缓存，从同一公开前缀和冻结参数重建；不改变正式 RNG 或已记录的行为概率。

`v3` 指检查点及状态 JSON 的数据格式，不是 main 提交版本或训练方法。原始状态和 13 项配置布局保持不变，新元数据使用可选字段。此前 `chunk=0` 的完整终局 v3 可直接恢复模型、优化器、有效未结束轨迹及 ready；旧样本没有搜索标签时继续使用原 V-trace 项，新动作按本轮搜索配置采样。短块 v3 仍拒绝正式续训。无需删除已有完整终局检查点。并发、worker、推理上限、设备、内存预算、`--max-game-steps` 和上述搜索／历史池参数可以显式覆盖。

每次 `checkpoint.pt` 原子落盘后立即发布对应冻结状态的 `saved.json`，作为面板和后台测评的保存回执；运行中手动保存、定时保存与停止保存共用同一路径。

停止时先等待检查点与回执写完，再退出学习／规则线程和推理服务，最后在命令返回后显式释放 CUDA 内核缓存及其锁页内存，避免拖到进程全局析构。普通输出即时刷新，退出日志记录清理阶段；Linux 通过 LibTorch 打印致命信号栈。`wall.json` 在停止保存流程返回后写入，之后的异常属于退出阶段；面板仍按实际退出码报告。

执行异常在等待线程之前写入日志和 `failure.txt`，退出日志分别记录 learner、各 lane 和推理服务，保留触发清理的原始原因。

训练恢复与评测均不比较构建指纹。v3 共享表格式保留；正式恢复要求 `chunk=0` 的完整终局轨迹，旧短轨迹档明确提示从头开始，仍可用于冻结测评。源检查点只读，新保存写入新运行目录。模型尺寸和 BF16 类型检查保留。

缩容在当前规则操作完成后停放并释放规则树与缓存；扩容先恢复旧局。并发可调到零，停放局仍可保存，ready 对局继续学习。恢复保留档内学习 batch 和并发（包括零并发），设备、lane、线程、推理上限与宿主内存预算重新检测；需要调整并发时显式指定 `--concurrency`。

Colab 启动配置中的 Worker 数为 0 时自动选择，正整数覆盖；控制台可独立调整并发和 1–128 个 Worker。命令行为 `workers --directory PATH --workers N`。运行中调整先停在一致边界，在原 lane 序列化并释放 Rust 状态，再由新 lane 恢复；保留对局、RNG、未决转换、完整及部分轨迹和历史对手绑定。推理服务按 `inference_batch` 分配批上限，不再固定限制为启动时的 worker 数。状态中的 `workers` 为已应用数量。

## 训练与资源边界

模型固定为 384 宽、6 层、6 头、FFN 1536，共 15,184,130 参数。模型只接收公开类型化输入。模型、激活、梯度、保存的统计和五组优化器状态为 BF16；算子内部累加可使用 FP32。完整约束见[架构决策](architecture/DECISION.md)。

同席位的全部决策保留到真实终局，折扣为 1，只有胜/负/平产生 +1/-1/0 奖励。未终局、停放与中断轨迹保留原状态，不进入学习。冻结当前 learner 参数计算整局 V-trace 目标，反向递推跨越计算小批；随后逐小批反向并用 BF16 补偿汇总梯度，完整验证后只提交一次 AdamW 更新。行为概率来自实际 BF16 质量的规范分布。算法依据与边界见[架构决策](architecture/DECISION.md)。

完整终局按完成顺序异步进入学习，不等待长局或停放局。`--update-games` 默认 4，控制每次更新最多合并的完整终局数；目标并发不足时取较小值。`--update-wait-seconds` 默认 30，从首个 ready 开始计时，到时即提交已有完整局；并发为 0 或内存背压时立即消费。`batch` 只控制前后向计算小批，同批各局分别计算完整 V-trace，汇总梯度后仅提交一次更新和发布。学习期间照常补局，ready 对局数达到目标并发时才限制补局，已有局继续行动。保存与停止保留尚未消费的 ready，恢复后重新计等待时间。合批配置随检查点恢复，旧档缺失时采用新默认值，命令行可显式覆盖。

实际消费仍检查策略滞后上限 K=64。完整对局中任一行为版本超过上限，则整局跳过学习，记录为 `stale_games/stale_steps`，继续消费后面的新鲜对局。没有按更新次数触发的全局等待，没有长局计时截断；停放局恢复后继续真实对局，最终是否可学习按实际版本差决定。

CPU lane 公平轮转，各 lane 并行计算策略头；根编码缓存命中不进入设备队列，关闭搜索时不计算价值头。策略服务使用两个有界编码槽，CPU 解码与 GPU 编码重叠；切换冻结模型前先排空旧槽，队首历史模型不会被后续当前模型请求饿死。缓存、迟到结果及 CPU 头绑定实际快照版本。推理图按配置捕获，超容量执行完整普通前向。完整终局学习按计算小批执行普通前向/反向，只有一个优化更新在途。

`--memory-mb` 管理宿主轨迹与缓存，CUDA 分配器另使用物理显存的 90% 预算。固定转换数不等于固定 token 数；不得把本机容量经验当作所有训练卡的性能结论。

`status.json` 为最新快照；`window.json` 在退出排空前冻结；`saved.json` 为保存后的累计状态。`inference_waiting/inference_running` 分别统计设备编码队列与在途根请求，lane 等待编码期间仍包含在 `busy` 中；`policy_cpu_seconds` 累加各 lane 的解码时间，可超过墙钟。`consumed_within_lag` 与 `consumed_lag_max` 报告 K 内消费及实际最大滞后；`stale_games/stale_steps` 单列过期未学习的数据。恢复后的速率与窗口计数重新计时，检查点保留累计过期计数。累计守恒为 `produced = consumed + stale_steps + discarded_steps + ready + partial + learning`。

`training_method=terminal-vtrace` 标识默认直接策略，开启搜索时为 `terminal-vtrace-mcts`；`terminal_games_consumed` 是本次进程已学习的终局数。`latest_value_mse` 对照真实终局结果，`latest_policy_entropy` 和 `latest_gradient_norm` 用于检查退化；这些原始指标可在控制台 JSON 中查看。首次终局前 updates/consumed 为零属于正常采集阶段。完整更新期间保存/停止等待原子边界；宿主轨迹接近预算时按前述内存回收规则整局淘汰或暂停采样。

功能检查见[验证入口](VALIDATION.md)。旧短块学习器及其固定工作集、性能诊断入口已移除，历史源码与测量资料见[历史索引](../../docs/ai/HISTORY.md)。v3 检查点布局保留，现有完整终局训练可继续恢复。
