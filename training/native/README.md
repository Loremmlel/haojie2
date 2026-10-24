# 独立 C++ / LibTorch 宿主

`haojie-train` 独占模型、推理、学习、优化器、调度与检查点。Rust 共享库只拥有规则和逐局 actor。运行时不启动 Python、Node 或其他训练子进程。

## 构建

需要 C++20 编译器、CMake 3.24+、Rust 1.98.1，以及相应平台的 LibTorch。Windows 使用 MSVC；Linux 使用 GCC。CUDA 由预编译 LibTorch 提供，无自定义 CUDA kernel，因此 Windows 构建无需匹配版本的 CUDA 编译器。

```powershell
pwsh -File training/native/build.ps1 -LibTorch E:/libtorch
pwsh -File training/native/run.ps1 run --directory artifacts/native/run --concurrency 64 --workers 8
```

已有环境中的 `torch` 目录也可以作为 LibTorch 根目录；构建与运行只链接其 C++ 库。构建脚本在本机可自动找到 `training/.venv/Lib/site-packages/torch`，运行脚本使用构建时记录的路径。没有 `torch_python` 链接和 Python 初始化。

```sh
export LIBTORCH=/opt/libtorch
bash training/native/build.sh
artifacts/cpp-training-build/haojie-train run --directory artifacts/native/run --device cuda --concurrency 64 --workers 8
```

Linux 可执行文件记录 LibTorch 和规则库的构建路径；搬运时将相应库目录放入 `LD_LIBRARY_PATH`。Windows 的 `run.ps1` 只为子进程配置 DLL 搜索目录；也可直接配置 `PATH` 后运行可执行文件。

## 控制与保存

```powershell
pwsh -File training/native/run.ps1 status --directory artifacts/native/run
pwsh -File training/native/run.ps1 resize --directory artifacts/native/run --concurrency 256
pwsh -File training/native/run.ps1 resize --directory artifacts/native/run --concurrency 0
pwsh -File training/native/run.ps1 budget --directory artifacts/native/run --memory-mb 4096
pwsh -File training/native/run.ps1 save --directory artifacts/native/run
pwsh -File training/native/run.ps1 stop --directory artifacts/native/run
pwsh -File training/native/run.ps1 run --directory artifacts/native/restored --resume artifacts/native/run/checkpoint.pt --concurrency 64 --workers 8
```

`resize` 缩小时，在当前规则操作完成后将整局序列化并从 Rust lane 移除，释放动作树和根推理缓存；扩容先恢复停放局，再开新局。零并发可以保存全部进度。每个 CPU lane 固定拥有自己的目录和 actor，规则对象及 `Rc` 不跨线程传递。

默认每 300 秒保存；`--save-seconds 0` 关闭定时保存。保存先处理在途 CPU 操作和已经提交的唯一学习工作，在一致边界冻结 CPU 副本，随后后台写临时文件并原子替换 `checkpoint.pt`。磁盘恢复只使用写入完成的检查点。运行目录必须新建或为空，防止覆盖已有实验。

检查点包括运行配置、学习参数、发布的 actor 参数、五组 BF16 优化器状态及整数步数、正式状态、规则 RNG、采样 RNG、参数路径、完整排除 frontier、未决同席位转换、部分轨迹、待学习块和累计消费。恢复默认沿用保存的模型、规则模式、设备、并发、worker、批量、发布周期和内存预算；可显式覆盖设备、并发、worker 和内存预算。根推理缓存恢复时重建。仅加载自己生成且可信的检查点；原生构建身份和参数布局必须相符。

## 模型与轨迹

默认网络使用宽 384、6 层、6 个注意力头、FFN 1536 的 Pre-LN Transformer，共 15,048,578 参数。公开实体及全局 token 完整参与六层自注意力交互；节点上下文加入有序参数行、当前路径和各层精确排除事实。策略头结合上下文、候选及源/目标实体，经过两层 GELU 隐藏层后评分，使上下文能改变候选的相对偏好。价值头也使用非线性隐藏层。

根、节点和候选采用 ragged 表，根和候选投影按批去重，候选无宽度补零。根注意力按长度分桶，仅在桶内补齐并遮罩 padding，每个根独立进行注意力计算；没有实体截断。前缀行在聚合前经过非线性映射，字段顺序、参数所属关系和路径序号保留在整数特征中。推理缓存复用完整六层 Transformer 的根编码。

身份、索引、顺序、整数参数、类型位和存在位以整数或布尔形式保存；送入网络前在设备上展开为精确的 0/1 位特征。连续字段、参数、激活、梯度和五组补偿 AdamW 状态均为 BF16。统计与宿主分类概率的标量计算使用双精度，不建立 FP32/FP64 模型或优化器张量。

不保存或重放完整尝试历史，不使用 GRU，也不持有长根模型租约。当前公开状态、参数路径与 frontier 是决策输入；规则状态、正式随机数和暗选私有字段只留在规则侧。根缓存绑定当前发布版本；发布时递增版本并按需重建缓存，不按角色或根版本切分推理批。换根只传一次根表。

每局保留一个学习席位；对手使用同一个当前发布 actor。学习连接同席位的随机微步；对手行动期间保留未决转换，真实终局奖励为胜 `+1`、负 `-1`、平 `0`，其他转换奖励为零。短轨迹使用真实后继 bootstrap，失败与回溯更新完整候选掩码和 frontier。

actor 和 learner 共用分段 BF16 softmax。Rust 按实际 BF16 概率质量精确归一化并抽样，记录实际行为对数概率；learner 使用相同宿主定义计算重要性比，并将规范对数概率作为 BF16 前向值，梯度通过同一分段策略计算。每个学习批只执行一次前向与一次反向；根与候选投影按表去重，不做激活 CPU offload 或第二次解码。所有优化器更新预计算并检查有限性后才一起提交。

## 调度、内存与统计

每个就绪 actor 独立进入 FIFO，不等待同 worker 的其他局。一个全局 GPU 提交锁协调推理和唯一在途学习工作；推理结果释放 GPU 锁后直接回送 CPU lane。学习在独立线程执行，CPU 规则可以同时推进。默认每 8 次优化步发布一次权重，不按单局发布或推进样本代际。

默认轨迹块 16 微步、学习批 512 微步、推理最多 256 节点。原始公开输入、未决转换、部分块、完成块、学习在途块、CPU 收发队列、Rust 活动分配和根缓存一起计入 `managed_bytes`。`--memory-mb` 约束这部分数据预算；固定模型、优化器及算子临时工作区另外占用内存。推理按未命中根、前缀和候选的实际行数限制工作量，命中缓存不再收取完整根编码费用。

达到背压水位时，可获得真实后继的部分块直接切出学习；完成块只是转移已有数据的归属，不会在多局同时完成时重新分配整段轨迹。没有可消费块时继续推进获得同席位后继所需的环境。超过数据预算时先保存检查点再明确退出，可提高预算后恢复。

`status.json` 是最新完整统计快照；`window.json` 在退出排空前冻结；`saved.json` 单列保存后的累计状态；`wall.json` 包含初始化、窗口、保存和收尾的整条管线耗时。恢复后窗口计数从恢复点重新起算。消费、产生、ready、partial、pending、learning 和活跃/停放局分别列出，`advancing_1s` 单列最近一秒实际推进的局，并检查 `累计产生 = 累计消费 + ready + partial + learning`。

`--samples 4096` 精确限制本轮实际消费；需要时在真实后继处切分块。`--updates` 限制优化步，`--seconds` 限制测量窗口；排空不计入窗口消费。固定工作量短基准与剩余瓶颈见[本地验收](VALIDATION.md)。

## 验证

```powershell
pwsh -File training/native/run.ps1 check --device cpu
pwsh -File training/native/run.ps1 check --device cuda
pwsh -File training/native/validate.ps1 -Output artifacts/native/control-check -Device cuda
pwsh -File training/native/benchmark.ps1 -Output artifacts/native/fixed-work -Samples 4096
```

检查覆盖 BF16 前后向和优化器、非有限更新拒绝、模型/优化器保存后继续更新、支持集与行为概率、类型化节点序列化、停放和正式 RNG 连续性，以及真实在线缩放、停止、磁盘恢复和继续训练。
