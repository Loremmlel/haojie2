# 独立 C++ / LibTorch 宿主

`haojie-train` 独占模型、推理、学习、优化器、调度与检查点。Rust 共享库只拥有规则和逐局 actor。运行时不启动 Python、Node 或其他训练子进程。

## 构建

需要 C++20 编译器、CMake 3.24+、Rust 1.98.1，以及相应平台的 LibTorch。Windows 使用 MSVC；Linux 使用 GCC。CUDA 计算使用 LibTorch，固定输入图使用 NVRTC 编译的整数/BF16 投影与搬运内核。Windows cu130 构建入口会下载并校验 NVIDIA 官方 Runtime/CRT 头包，缓存于 `artifacts/native-cuda-runtime-13/`，无需调用 nvcc；也可显式传入 `-CudaRuntime`、`-CudaCompilerHeaders`。

```powershell
pwsh -File training/native/build.ps1 -LibTorch E:/libtorch
pwsh -File training/native/run.ps1 run --directory artifacts/native/run --concurrency 12 --workers 8 --batch 32 --inference-batch 12
```

已有环境中的 `torch` 目录也可以作为 LibTorch 根目录；构建与运行只链接其 C++ 库。构建脚本在本机可自动找到 `training/.venv/Lib/site-packages/torch`，运行脚本使用构建时记录的路径。没有 `torch_python` 链接和 Python 初始化。

```sh
export LIBTORCH=/opt/libtorch
bash training/native/build.sh
artifacts/cpp-training-build/haojie-train run --directory artifacts/native/run --device cuda --concurrency 12 --workers 8 --batch 32 --inference-batch 12
```

Linux 可执行文件记录 LibTorch 和规则库的构建路径；搬运时将相应库目录放入 `LD_LIBRARY_PATH`。Windows 的 `run.ps1` 只为子进程配置 DLL 搜索目录；也可直接配置 `PATH` 后运行可执行文件。

## 控制与保存

```powershell
pwsh -File training/native/run.ps1 status --directory artifacts/native/run
pwsh -File training/native/run.ps1 resize --directory artifacts/native/run --concurrency 8
pwsh -File training/native/run.ps1 resize --directory artifacts/native/run --concurrency 0
pwsh -File training/native/run.ps1 budget --directory artifacts/native/run --memory-mb 4096
pwsh -File training/native/run.ps1 save --directory artifacts/native/run
pwsh -File training/native/run.ps1 stop --directory artifacts/native/run
pwsh -File training/native/run.ps1 run --directory artifacts/native/restored --resume artifacts/native/run/checkpoint.pt --concurrency 12 --workers 8
```

`resize` 缩小时，在当前规则操作完成后将整局序列化并从 Rust lane 移除，释放动作树和根推理缓存；扩容先恢复停放局，再开新局。零并发可以保存全部进度。每个 CPU lane 固定拥有自己的目录和 actor，规则对象及 `Rc` 不跨线程传递。

默认每 300 秒保存；`--save-seconds 0` 关闭定时保存。保存先处理在途 CPU、推理和已经提交的唯一学习工作，在一致边界冻结 CPU 副本，随后后台写临时文件并原子替换 `checkpoint.pt`。磁盘恢复只使用写入完成的检查点。运行目录必须新建或为空，防止覆盖已有实验。

检查点包括运行配置、学习参数、发布的 actor 参数、五组 BF16 优化器状态及整数步数、正式状态、规则 RNG、采样 RNG、参数路径、完整排除 frontier、未决同席位转换、部分轨迹、待学习块和累计消费。v3 格式把唯一行、候选、根和节点各写一次，全局行也使用同一类型化行表。恢复默认沿用保存的模型、规则模式、设备、并发、worker、批量、发布周期和内存预算；可显式覆盖设备、并发、worker 和内存预算。归档只加载一次，根推理缓存恢复时重建。原生构建身份和参数布局必须相符。

## 模型与轨迹

默认网络使用宽 384、6 层、6 个注意力头、FFN 1536 的 Pre-LN Transformer，共 15,184,130 参数。公开实体及全局 token 完整参与六层自注意力交互；节点上下文加入有序参数行、当前路径和各层精确排除事实。节点 query 与候选、源/目标实体绑定的 key 分别通过 GELU MLP，再用缩放点积和可学习先验评分。价值头也使用非线性隐藏层。

根、节点和候选采用张量表，完整类型化行及候选按批去重。CUDA 使用 LibTorch 自带的变长 FlashAttention/efficient attention 前后向；CPU 使用 BF16 注意力矩阵。普通学习按实际长度计算，训练图按有限容量族填充，并用独立查询/key 累计长度隔离填充位置及梯度。前缀行先非线性映射再聚合，顺序和绑定保留在整数特征中。CPU 缓存同一根绑定的候选 key，学习按唯一绑定计算；被排除的动作保留原支持集索引，省去宽候选 MLP。重复引用及前缀均值采用原生 CSR 归约。

公开输入版本为 `haojie-public-tables-v3`。实体与全局行使用 int64 身份/整数、布尔掩码及独立 BF16 连续槽，候选直接保存 int64。完整 64 位整数经过可逆 ZigZag 映射后投影；CPU 稀疏求和，训练图在设备展开精确零一矩阵后使用张量核心。Rust 不生成旧浮点副本。参数、激活、梯度、保存的归一化统计及五组补偿 AdamW 状态均为 BF16；算子内部寄存器和工作区负责累加。统计与规范行为概率的标量计算使用双精度。

不保存或重放完整尝试历史，不使用 GRU，也不持有长根模型租约。当前公开状态、参数路径与 frontier 是决策输入；规则状态、正式随机数和暗选私有字段只留在规则侧。根缓存绑定当前发布版本；发布时递增版本并按需重建缓存，不按角色或根版本切分推理批。换根只传一次根表。

每局保留一个学习席位；对手使用同一个当前发布 actor。学习连接同席位的随机微步；对手行动期间保留未决转换，真实终局奖励为胜 `+1`、负 `-1`、平 `0`，其他转换奖励为零。短轨迹使用真实后继 bootstrap、折扣 1 和 V-trace，失败与回溯更新完整候选掩码和 frontier。训练密度与方法的论文依据见[架构决策](architecture/DECISION.md)。

actor 和 learner 共用分段 BF16 softmax。Rust 按实际 BF16 质量精确归一化、抽样并记录行为对数概率，learner 使用相同规范定义计算重要性比。CUDA 在设备递推 V-trace，零质量旧动作的修正和策略项严格为零。每批只有一次前向与反向，bootstrap 保留完整编码和价值。训练图统一执行类型化投影、模型、目标、损失、反向及更新；参数、梯度和优化器保持稳定地址。全部提案及损失验证后共同提交参数和五组状态，宿主读取一次回执。超出图容量时执行同一完整普通学习器。

## 调度、内存与统计

所属 CPU lane 在 Rust 返回视图上直接解码，持有未决转换、部分轨迹和已完成块；热根上的评分、规则推进及轨迹切分连续执行，各局公平轮转。独立根服务直接接收缺根请求，集中使用两个 GPU 槽，完整 Transformer 编码后将 BF16 实体仿射项、上下文及匹配的不可变头部快照直接交回所属 lane。控制线程处理完成块和控制事件。发布等待已提交 GPU 读取，再整体复制权重并生成不可变快照；已开始的 CPU 决策可持旧快照完成。图在并行执行器启动前捕获，迟到编码仍绑定实际旧版本。默认每 8 次优化步发布。

默认并发 **12**、轨迹块 16 微步、学习批 512 微步、推理最多 64 节点；后续性能对照统一使用 12 并发。原始公开输入、未决转换、部分块、完成块、学习在途块、CPU 收发队列、Rust 活动分配和根缓存一起计入 `managed_bytes`。`--memory-mb` 约束这部分数据预算；固定模型、优化器、图槽及算子临时工作区另外占用内存。推理按未命中根、前缀和候选的实际行数限制工作量，命中缓存不再收取完整根编码费用。

类型化行、候选、根和节点按存活期记账，共享相同值并在最后一个引用释放时归还预算。每个设备槽整包回送根输出，完成后交给根拥有的 CPU 缓存并复用槽；CPU 根结果与投影缓存采用同样的记账。没有活动局的 lane 释放投影缓存及模型快照。控制线程不反复遍历整张轨迹引用图。

达到背压水位时，可获得真实后继的部分块直接切出学习；完成块只是转移已有数据的归属，不会在多局同时完成时重新分配整段轨迹。没有可消费块时继续推进获得同席位后继所需的环境。超过数据预算时先保存检查点再明确退出，可提高预算后恢复。

`status.json` 是最新完整统计快照；`window.json` 在退出排空前冻结；`saved.json` 单列保存后的累计状态；`wall.json` 包含初始化、窗口、保存和收尾的整条管线耗时。恢复后窗口计数从恢复点重新起算。消费、产生、ready、partial、pending、learning 和活跃/停放局分别列出；`active_game_seconds` 与 `mean_active` 记录本进程非停放环境的时间积分和平均数，显示有限工作集的并发回落；`advancing_1s` 单列最近一秒实际推进的局。检查 `累计产生 = 累计消费 + ready + partial + learning`。

`--samples 4096` 精确限制本轮实际消费；需要时在真实后继处切分块。`--updates` 限制优化步，`--seconds` 限制测量窗口；排空不计入窗口消费。固定工作量短基准、main 对照与剩余瓶颈见[最新实测](architecture/MEASUREMENTS.md)。

## 验证

```powershell
pwsh -File training/native/run.ps1 check --device cpu
pwsh -File training/native/run.ps1 check --device cuda
pwsh -File training/native/validate.ps1 -Output artifacts/native/control-check -Device cuda
pwsh -File training/native/benchmark.ps1 -Output artifacts/native/fixed-work -Samples 4096
```

测试入口默认使用 12 并发、推理批量上限 12、学习批量 32；固定消费短测默认 1,024 个转换、一次重复，main 对照两边各 20 秒。main 保留其原有较小学习批量。生产宿主默认学习批量仍为 512，测试参数通过命令行覆盖；结果单列批量、实际消费和更新数。

检查覆盖 BF16 前后向和优化器、非有限更新拒绝、模型/优化器保存后继续更新、支持集与行为概率、类型化节点序列化、停放和正式 RNG 连续性，以及真实在线缩放、停止、磁盘恢复和继续训练。

固定对局工作集诊断使用同一真实状态、实际行为概率及训练配额；main 对照只测 12 并发，额外短曲线经作者授权使用 1/2/4/8/12。任务为同一组 16 个游戏种子：

```powershell
& training/native/benchmark.ps1 -Output artifacts/native/workset12 -Games 16 -GameSamples 256 -Concurrency @(12) -Repeats 1 -Shrine
```

在 PowerShell 中直接调用脚本并传入数组参数。首轮生成参考；可用 `-BehaviorReference` 指定已有参考，比较不同实现。重放时照常执行模型与学习，并使用参考 BF16 行为分布推进相同真实规则，逐局行为和正式停放状态必须完全相同。这个独立诊断入口保留所有在途局；普通 `run` 始终使用当前 actor 的实际输出。
