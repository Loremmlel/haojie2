# 普通局训练

新入口采用单进程、Rust 固定工作线程池和 PyTorch。Rust 负责规则、三档手工 AI、搜索与请求组织，Python 主线程负责 GPU 推理与学习，协调/I/O 线程负责回放、阶段和保存。只接受普通局；普通局的终极召唤、合成、老千自选、死亡反应和随机机制均保留。分层解码 v2 模型为 5,503,366 参数。

## 安装

在仓库根目录执行。需要 Python 3.12、Rust 1.98.1 和平台链接器。安装依赖后，正式训练不需要 Node、V8、C++ 宿主或 LibTorch 开发包。

```powershell
python -m venv training/.venv
training/.venv/Scripts/python.exe -m pip install -r training/requirements.txt
cargo build --locked --release --manifest-path native/engine/Cargo.toml --target-dir artifacts/native-target
```

PyTorch 请使用与本机驱动匹配的 [官方安装命令](https://pytorch.org/get-started/locally/)。安装后用以下命令确认实际设备：

```powershell
training/.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')"
```

Linux 对应使用 `training/.venv/bin/python`。规则库默认为 `artifacts/native-target/release/haojie_native.dll` 或 `libhaojie_native.so`，自定义路径用 `HAOJIE_ENGINE`。CPU 功能验收可显式选择 `--device cpu`。

NumPy 固定为 **2.5.3**。原生搜索直接使用 `_generator` 导出的分布函数及 `numpy_1.24_ufunc_call_info` capsule 提供的无 GIL f32 exp 内核，以保持原来的随机消耗和浮点舍入。这些属于 NumPy 私有 C ABI，启动时检查精确版本，不匹配就报错。此次功能验收使用 NumPy 2.5.3、PyTorch 2.14.0+cu130。

## 运行与恢复

### Web 控制台

```powershell
training/.venv/Scripts/python.exe -m training.haojie web --port 8765
```

打开 `http://127.0.0.1:8765`。默认管理 `artifacts/training/` 下的运行，`--root` 可指定其他根目录。页面提供新建、选择已保存运行、保存停止、继续和运行中应用参数；一个服务管理一个训练进程，刷新/关闭页面不影响训练，服务退出时请求保存停止。

支持调整搜索线程、每线程并行局数、训练/推理批量、搜索预算、学习率、对手比例、回放内存和保存/停止条件。运行中点击“保存并应用参数”先正常保存，再以新配置继续；缩容时多出的原局排队，扩大时恢复原局并补充新局。种子、设备和预训练配置只在新建时设置，网络保持 5.50M、原生 BF16。

图表默认 60 秒时间加权 EMA 和保形平滑曲线，可切换 15/30/60/120 秒、原始线及 10 分钟至 24 小时时间范围。后台最多保存 17,280 个观测点和 200 条日志，均在服务内存；服务重启后曲线重新采集。资源读数区分训练进程 CPU/RAM 与整卡 GPU/显存，缺测留空，保存/重启区间断线。

前端已打包为单个 HTML，运行不需要 Node。修改页面后执行 `npm run training:web:build`；`npm run training:web:check` 核对产物，`npm run training:web:test` 运行本机 GPU 浏览器功能验收，`npm run training:web:views` 使用明确标注的合成数据检查图表与草稿交互。设计与接口边界见[控制台说明](../docs/ai/TRAINING-WEB.md)。

### 命令行

```powershell
training/.venv/Scripts/python.exe -m training.haojie run --directory artifacts/classic-main
```

默认先采集 8 个自然结束的简单档示范局，再完成 512 次独立模仿更新，发布模型后才启动自对弈。正常阶段使用 70% 当前自对弈、20% 历史模型、10% 手工对手；历史为空或选中当前版本时转为自对弈。默认 8 个 Rust 工作线程，每个推进 8 局，总在途仍为 64 局；一局等待神经响应或记录确认时，线程继续推进其他就绪局。`--workers`、`--games-per-worker` 分别调整原生线程数和每线程在途局数。`--warmup-games`、`--warmup-updates` 调整两个预训练阶段；前者为 0 时明确跳过预训练。

网络为 5,503,366 参数，宽度 256、6 层。主干只处理实体 token，效果/死亡等原有记录在局部编码；加入规则几何注意力偏置。训练、评测、历史模型推理和截断校准统一原生 BF16：权重、梯度、AdamW 一阶/二阶动量、神经浮点输入、输出与损失均为 BF16，没有 FP32 主权重或 autocast。步数、实体引用和类别索引使用整数。GPU 输出按 BF16 原始 16 位经进程内批量 FFI 回填；CPU 规则、搜索统计和落盘事实使用各自的数据类型。CUDA 使用 PyTorch 标准 BF16 算子及其内部累加实现。

默认学习批量为 **16**（`--batch`），推理批量上限为 **32**（`--inference-batch`）。状态记录 PyTorch 已分配峰值 `cuda_peak_bytes`、预留峰值 `cuda_peak_reserved_bytes` 及锁页主机内存当前/峰值 `pinned_host_bytes` / `pinned_host_peak_bytes`（含缓存）；不支持锁页统计的 PyTorch 版本记 null。整卡显存和 Windows 共享 GPU 内存占用在性能采样中另记。

来源 → 动作 → 参数按网络条件分布解码，共用一次状态主干；搜索的一条边是一条完整命令。候选随访问量渐进扩宽，无固定 32 命令上限，同次搜索内复用转置局面。默认 25% 决策搜索 64 次，其余 16 次，最大深度 16 条完整命令；唯一合法命令直接执行。仅深搜步和示范步形成策略监督。`--simulations`、`--fast-simulations`、`--full-fraction`、`--depth` 可调整；主干评估与条件头解码分别计数。

`--seconds 600`、`--updates 100`、`--games 10` 分别限制本次启动的秒数、更新数、自然终局数；0 表示不限。按 Ctrl+C 或在运行目录创建 `STOP` 文件会保存并退出。续训前移除自己创建的 `STOP` 文件：

```powershell
training/.venv/Scripts/python.exe -m training.haojie run --directory artifacts/classic-main --resume
```

每 300 秒自动保存，`--save-seconds` 可调整。保存先暂停原生推进，收回已发出的神经响应并排空正式记录，再通过屏障取得恢复状态。恢复模型、AdamW、随机数、预训练阶段/进度、校准器、历史池、数据进度、手工 AI 预算与缓存计划，以及所有尚未完成的对局。未完成搜索按原命令起点和固定模型重新计算。改变线程或并行局数时重分配原局，多出的局保存在待恢复队列中；学习率同时更新优化器参数组。

续训、评测和基准只接受 `haojie-classic-training-v3` 检查点，评测也可读取历史池权重。没有旧格式导入或迁移；新训练使用新目录。网络和监督仍为当前的分层解码 v2。

回放在内存中完成，默认最多 **128 局、4096 MiB 张量**（`--replay-games`、`--replay-mib`），超限按最早入池整局淘汰。元数据和终局标签直接更新共享内存对象，训练抽样不读取文件。预算不含原生环境、在途待打包样本、命令和时间线元数据；单个片段超过预算会报错，需增大预算。状态输出提供 `replay_bytes`、`replay_games` 和 `replay_limit_bytes`。

普通发布权重只在内存：当前版本和在途搜索引用版本存活，其余临时版本回收。历史权重有 **256 MiB CPU 缓存**和最多 **3 个 GPU 模型缓存**；活跃版本可超过 CPU 缓存预算，以保证搜索不中途换权重。只有进入历史池的模型单独落盘，历史池仍按原来的跨时期策略保存和抽取。

每次保存只原子替换一个 `checkpoint.pt`，包含当前模型/优化器、必要临时权重、有限回放、在途对局和随机数等完整恢复状态。运行时不再写逐局文件或追加结果日志；异常中断从最近成功保存点恢复。保存期间磁盘会暂时同时存在旧检查点和新临时文件。

每局每 64 条命令将样本打包送入内存回放。正常采集默认完成 **50 个单方回合**后截断；开局随机保留 **25% 完整局**免除此回合上限，示范局全部保留。`--max-ply` 和 `--full-game-fraction` 可调整，前者为 0 时关闭回合截断。完整局分配随每局保存，恢复不重抽；调整默认值只影响新分配对局。全部对局仍受 20,000 条命令硬上限保护。

截断边界局面直接在内存中完成自举，分别记录 `ply_limit` / `command_limit` 原因，胜负为 unknown。自对弈累计至少 16 个真实终局且校准器通过留出检验后，可附加低权重截断软价值；到 128 个真实终局后改用冻结发布模型自举。软值限制 ±0.98，权重 0.25，与真实胜负分字段记录。此前截断局屏蔽价值损失。正常暂停的在途局随 `--resume` 继续；自然终局通过同一局元数据关联真实标签。详见[目标定义](../docs/ai/TRAINING.md#奖励与截断)。

## 结果与评测

运行目录包含：

- `checkpoint.pt`：唯一滚动恢复点，包含有限内存回放和在途状态；`models/`：仅历史池权重。
- `status.json`：阶段、更新、自然终局、两类截断、真实/软价值采样、主干/解码次数、推理/学习耗时与显存峰值。
- `failed-native.json`：运行失败时覆盖保存的诊断。日常不生成逐局目录和累计结果文件。

```powershell
training/.venv/Scripts/python.exe -m training.haojie evaluate --checkpoint artifacts/classic-main/checkpoint.pt --opponent easy --output artifacts/classic-main/eval-easy.jsonl --games 20
```

对手可选 `easy`、`medium`、`hard` 或另一模型文件。相同种子配对换边，冻结双方权重与搜索预算；unknown 单列。输出旁保存 `.model.pt` 及必要的 `.opponent.pt`，逐局记录引用这些冻结权重，续训不会改变评测依据。输出文件须为新文件，避免覆盖之前的评测。

## 验证和性能

按[测试范围](../docs/maintenance/TESTING.md)选择模块；以下 discover 用于训练整体回归，日常只运行受影响的测试。Web 协议使用 `training.tests.test_web`，图表使用 `npm run test:training-ui` 和 `npm run training:web:views`，均不需要启动 GPU。

```powershell
training/.venv/Scripts/python.exe -B -m unittest discover -s training/tests -v
training/.venv/Scripts/python.exe -B -m training.tests.bench_pipeline --checkpoint artifacts/classic-main/checkpoint.pt --output artifacts/classic-memory-benchmark --tasks 256 --commands 8 --trials 3
node --import tsx scripts/native/learning/validate.ts artifacts/native-target/release/haojie-engine.exe --evaluation --candidates --search --decoder
node --import tsx scripts/native/learning/teacher-validation.ts artifacts/native-target/release/haojie-engine.exe 64
```

正确性测试从 Git 提交 `7cc4ba9` 读取冻结的 Python 搜索，核对原生搜索的请求、命令、访问统计、监督和 RNG，并验证暂停恢复、阶段和 BF16。

统一基准从 v3 检查点的四局回放中各抽取四个中后期起点；默认连续分配 256 个窗口，每个推进 8 条命令，保持 64 局在途，执行正式 64/16 搜索和同量 BF16 学习。使用冻结反馈核对工作量，支持当前运行时的线程数或插桩开关对照；已删除旧文件及旧多进程架构适配。外部 CPU/GPU 采样另需 `psutil` 和 `nvidia-smi`；利用率只统计预热后仍有待分配窗口的满供给区间。`--output` 必须是新目录，保留报告和采样；运行用临时目录自动清理，源检查点只读。线程职责、数组生命周期和验收顺序见[运行时架构](../docs/ai/TRAINING-RUNTIME.md)。

比较当前单进程的线程数时使用 `--compare-workers 4 8`，固定总在途为 64 局，即 4×16 与 8×8；`--batch 16` 固定学习批量。可用已有 `--feedback` 跳过重新录制。Windows 采样另需 `pwsh.exe`，记录受测 PID 的共享/专用 GPU 内存，保留各适配器实例；无有效计数时记 null。本机复现命令：

```powershell
training/.venv/Scripts/python.exe -B -m training.tests.bench_pipeline --checkpoint artifacts/classic-main/checkpoint.pt --compare-workers 4 8 --batch 16 --output artifacts/classic-workers-benchmark --tasks 256 --commands 8 --trials 3
```

分段诊断默认关闭。运行前设置 `$env:HAOJIE_PROFILE = '1'`，`status.json` 会增加 `profile_native`、`profile_gpu` 和 `profile_coordinator`。Rust 按线程记录搜索控制、规则推进、合法动作解码、几何与状态编码等阶段；Python 分开记录组批、主机提交、传输等待、学习、回放与保存。嵌套阶段同时保留包含子段和排除子段的时间，比较占比使用后者。Windows 原生 CPU 周期只用于阶段占比；CUDA 抽样流区间包含提交间隙，实际 kernel 执行时间另用短 trace 确认。诊断后用 `Remove-Item Env:HAOJIE_PROFILE` 关闭。

以下命令固定 8×8、batch 16，复用冻结反馈，以 OFF/ON、ON/OFF 两组比较插桩开销：

```powershell
training/.venv/Scripts/python.exe -B -m training.tests.bench_pipeline --checkpoint artifacts/classic-main/checkpoint.pt --compare-profile --batch 16 --feedback artifacts/classic-memory-benchmark/feedback.pkl --output artifacts/classic-profile-benchmark --tasks 256 --commands 8
```

独立短 trace 使用 `$env:HAOJIE_TRACE = 'artifacts/profile/gpu-trace.json'`；默认 GPU 循环第 64–95 次采集 CPU/CUDA 活动及阶段标签。也可设置 `$env:HAOJIE_TRACE_START_SECONDS = '20'`，运行满 20 秒后再采集 32 个完整循环，避开启动阶段。输出 Chrome trace 和同名 `.meta.json` 开始/结束计数，CUDA API 的提交、拷贝和同步调用可分别归因。需要支持 CUDA profiler 的 PyTorch 环境。这个窗口含 profiler 开销，单独用于归因；上述 OFF/ON 基准会清除子进程的 trace 设置。使用后移除 `HAOJIE_TRACE` 和自行设置的 `HAOJIE_TRACE_START_SECONDS` 环境变量。具体计量口径见[分段诊断](../docs/ai/TRAINING-RUNTIME.md#分段诊断口径)，本机结论见[状态入口](../docs/ai/STATUS.md#当前-8-线程瓶颈诊断)。

开启 `HAOJIE_PROFILE=1` 时，`profile_native.batching` 记录原生取批大小、encode/decode 混合、互斥出队原因、队列最老请求年龄、锁和条件变量等待时间；`profile_gpu.observations` 记录输入批次、拆组数和实际神经批次，以及实体、记录、注意力配对和动作的有效／padding 数量。直方图小于等于 32 时为精确值，更大值向上归入二次幂桶；原生批次末桶为 32 及以上。

这些 Node 命令仅用于 TS/Rust 差分开发验收。训练和评测调用同一进程内 Rust 手工 AI。架构、目标定义和研究依据见[训练设计](../docs/ai/TRAINING.md)，实际完成的验收见[状态入口](../docs/ai/STATUS.md)。
