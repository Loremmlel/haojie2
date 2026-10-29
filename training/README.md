# 普通局训练

新入口采用 Python 多进程、进程内 Rust 规则/手工 AI、单 GPU PyTorch 推理与学习。只接受普通局；普通局的终极召唤、合成、老千自选、死亡反应和随机机制均保留。分层解码 v2 模型为 5,503,366 参数。

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

## 运行与恢复

```powershell
training/.venv/Scripts/python.exe -m training.haojie run --directory artifacts/classic-main
```

默认先采集 8 个自然结束的简单档示范局，再完成 512 次独立模仿更新，发布模型后才启动自对弈。正常阶段使用 70% 当前自对弈、20% 历史模型、10% 手工对手；历史为空或选中当前版本时转为自对弈。默认 4 个 CPU actor，每个推进 16 局；逐局请求直接送往独占 GPU 的执行进程，CPU 协调进程负责回放和学习批次。`--workers`、`--games-per-worker` 分别调整进程数和每进程在途局数。`--warmup-games`、`--warmup-updates` 调整两个预训练阶段；前者为 0 时明确跳过预训练。

网络为 5,503,366 参数，宽度 256、6 层。主干只处理实体 token，效果/死亡等原有记录在局部编码；加入规则几何注意力偏置。训练、评测、历史模型推理和截断校准统一原生 BF16：权重、梯度、AdamW 一阶/二阶动量、神经浮点输入、输出与损失均为 BF16，没有 FP32 主权重或 autocast。步数、实体引用和类别索引使用整数。GPU 输出按 BF16 原始 16 位经共享内存返回；CPU 规则、搜索统计和落盘事实使用各自的数据类型。CUDA 使用 PyTorch 标准 BF16 算子及其内部累加实现。

来源 → 动作 → 参数按网络条件分布解码，共用一次状态主干；搜索的一条边是一条完整命令。候选随访问量渐进扩宽，无固定 32 命令上限，同次搜索内复用转置局面。默认 25% 决策搜索 64 次，其余 16 次，最大深度 16 条完整命令；唯一合法命令直接执行。仅深搜步和示范步形成策略监督。`--simulations`、`--fast-simulations`、`--full-fraction`、`--depth` 可调整；主干评估与条件头解码分别计数。

`--seconds 600`、`--updates 100`、`--games 10` 分别限制本次启动的秒数、更新数、自然终局数；0 表示不限。按 Ctrl+C 或在运行目录创建 `STOP` 文件会保存并退出。续训前移除自己创建的 `STOP` 文件：

```powershell
training/.venv/Scripts/python.exe -m training.haojie run --directory artifacts/classic-main --resume
```

每 300 秒自动保存，`--save-seconds` 可调整。恢复模型、AdamW、随机数、预训练阶段/进度、校准器、历史池、数据进度、手工 AI 预算与缓存计划，以及所有尚未完成的对局。未完成搜索按原命令起点和固定模型重新计算。恢复保持原 worker 数、每 worker 局数。

当前检查点为 `haojie-classic-training-v2`，样本为 `autoregressive-v2`；启动训练须使用新目录，续训只接受该格式。

每局每 64 条命令写一个不可变片段。正常采集默认完成 **50 个单方回合**后截断；开局随机保留 **25% 完整局**免除此回合上限，示范局全部保留。`--max-ply` 和 `--full-game-fraction` 可调整，前者为 0 时关闭回合截断。完整局分配随每局保存，恢复不重抽；调整默认值只影响新分配对局。没有回合上限字段的现有 v2 在途局继续按原方式运行。全部对局仍受 20,000 条命令硬上限保护。

截断保存 `environment.json`，分别记录 `ply_limit` / `command_limit` 原因，胜负为 unknown。自对弈累计至少 16 个真实终局且校准器通过留出检验后，可附加低权重截断软价值；到 128 个真实终局后改用冻结发布模型自举。软值限制 ±0.98，权重 0.25，与真实胜负分字段记录。此前截断局屏蔽价值损失。正常暂停的在途局随 `--resume` 继续；自然终局通过同一局元数据关联真实标签。详见[目标定义](../docs/ai/TRAINING.md#奖励与截断)。

## 结果与评测

运行目录包含：

- `checkpoint.pt`：完整恢复点；`models/`：发布的推理模型与历史对手。
- `games/<id>/`：完整命令片段、实体/局部记录/几何张量、条件策略标签、元数据及必要的超限快照。
- `status.json`：阶段、更新、自然终局、两类截断、真实/软价值采样、主干/解码次数、推理/学习耗时与显存峰值。
- `results.jsonl`：逐局结果；失败时保留 `failed-worker-*.json`。

```powershell
training/.venv/Scripts/python.exe -m training.haojie evaluate --checkpoint artifacts/classic-main/checkpoint.pt --opponent easy --output artifacts/classic-main/eval-easy.jsonl --games 20
```

对手可选 `easy`、`medium`、`hard` 或另一模型文件。相同种子配对换边，冻结双方权重与搜索预算；unknown 单列。输出旁保存 `.model.pt` 及必要的 `.opponent.pt`，逐局记录引用这些冻结权重，续训不会改变评测依据。输出文件须为新文件，避免覆盖之前的评测。

## 验证和性能

```powershell
training/.venv/Scripts/python.exe -m unittest training.tests.test_training -v
training/.venv/Scripts/python.exe -m unittest training.tests.test_pipeline -v
training/.venv/Scripts/python.exe -m training.tests.bench_ipc --rounds 150
training/.venv/Scripts/python.exe -m training.tests.bench_pipeline --checkpoint artifacts/classic-main/checkpoint.pt --trials 3
node --import tsx scripts/native/learning/validate.ts artifacts/native-target/release/haojie-engine.exe --evaluation --candidates --search --decoder
node --import tsx scripts/native/learning/teacher-validation.ts artifacts/native-target/release/haojie-engine.exe 64
```

流水线基准需要四局已有轨迹，逐局抽取四个中后期起点；执行正式 64/16 搜索和同量 BF16 学习，分别测量 16、64 局供给。加 `--baseline <基线导出目录>` 可与提交 `d2eb7a4` 的旧运行时交错比较。基准目录自动清理，源检查点和轨迹只读。进程职责、协议和验收顺序见[运行时架构](../docs/ai/TRAINING-RUNTIME.md)。

这些 Node 命令仅用于 TS/Rust 差分开发验收。训练和评测调用同一进程内 Rust 手工 AI。架构、目标定义和研究依据见[训练设计](../docs/ai/TRAINING.md)，实际完成的验收见[状态入口](../docs/ai/STATUS.md)。
