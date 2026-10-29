# 普通局训练

新入口采用 Python 多进程、进程内 Rust 规则/手工 AI、单 GPU PyTorch 推理与学习。只接受普通局；普通局的终极召唤、合成、老千自选、死亡反应和随机机制均保留。模型为 5,338,950 参数。

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
training/.venv/Scripts/python.exe -m training.haojie run --directory artifacts/classic-main --workers 4 --games-per-worker 4
```

默认先生成 8 局手工 AI 示范；随后使用 70% 当前自对弈、20% 历史模型、10% 手工对手。历史池为空时相应份额转为当前自对弈。每个 worker 长期复用，一次交替推进多局；单局结束不重启进程。

网络为 5,338,950 参数。训练、评测和历史模型推理统一原生 BF16：权重、梯度、AdamW 一阶/二阶动量、输入浮点张量、输出与损失均为 BF16，没有 FP32 主权重或 autocast。AdamW 步数、动作来源/目标行号和类别嵌入索引使用 int64，避免 BF16 对大整数的舍入。GPU 输出按 BF16 原始 16 位数据经共享内存返回；CPU 规则编码、搜索统计和落盘原始事实独立于神经网络 dtype。CUDA 使用 PyTorch 标准 BF16 算子及其内部累加实现。

默认 32 条完整命令候选，25% 决策搜索 64 次，其余搜索 16 次；仅深搜步和示范步形成策略监督。`--simulations`、`--fast-simulations`、`--full-fraction` 可调整。搜索、推理、实际训练样本及自然终局分别计数。

`--seconds 600`、`--updates 100`、`--games 10` 分别限制本次启动的秒数、更新数、自然终局数；0 表示不限。按 Ctrl+C 或在运行目录创建 `STOP` 文件会保存并退出。续训前移除自己创建的 `STOP` 文件：

```powershell
training/.venv/Scripts/python.exe -m training.haojie run --directory artifacts/classic-main --resume
```

每 300 秒自动保存，`--save-seconds` 可调整。恢复模型、AdamW、随机数、历史池、数据进度、手工 AI 预算与缓存计划，以及所有尚未完成的对局。未完成搜索按原命令起点和固定模型重新计算。恢复保持原 worker 数、每 worker 局数与候选数。

每局每 64 条命令写一个不可变片段，默认最多 20,000 条命令。达到硬上限的局保存 `environment.json` 并标记 truncated，胜负为 unknown；不按基地血量判胜。正常暂停的在途局随 `--resume` 接着进行。终局到来后，片段通过同一局的元数据获得真实胜负标签。

## 结果与评测

运行目录包含：

- `checkpoint.pt`：完整恢复点；`models/`：发布的推理模型与历史对手。
- `games/<id>/`：命令片段、状态/候选/搜索分布张量、元数据及必要的超限快照。
- `status.json`：更新、自然终局、unknown、有效终局价值样本、推理/学习耗时、显存峰值。
- `results.jsonl`：逐局结果；失败时保留 `failed-worker-*.json`。

```powershell
training/.venv/Scripts/python.exe -m training.haojie evaluate --checkpoint artifacts/classic-main/checkpoint.pt --opponent easy --output artifacts/classic-main/eval-easy.jsonl --games 20
```

对手可选 `easy`、`medium`、`hard` 或另一模型文件。相同种子配对换边，冻结双方权重与搜索预算；unknown 单列。输出旁保存 `.model.pt` 及必要的 `.opponent.pt`，逐局记录引用这些冻结权重，续训不会改变评测依据。输出文件须为新文件，避免覆盖之前的评测。

## 验证和性能

```powershell
training/.venv/Scripts/python.exe -m unittest training.tests.test_training -v
training/.venv/Scripts/python.exe -m training.tests.bench_ipc --rounds 150
node --import tsx scripts/native/learning/validate.ts artifacts/native-target/release/haojie-engine.exe --evaluation --candidates --search
node --import tsx scripts/native/learning/teacher-validation.ts artifacts/native-target/release/haojie-engine.exe 64
```

这些 Node 命令仅用于 TS/Rust 差分开发验收。训练和评测调用同一进程内 Rust 手工 AI。架构、目标定义和研究依据见[训练设计](../docs/ai/TRAINING.md)，实际完成的验收见[状态入口](../docs/ai/STATUS.md)。
