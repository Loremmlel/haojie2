# Python 训练子项目

Colab / 多核服务器入口见 [COLAB.md](COLAB.md)，包含硬件初始配置、监控、完整恢复和周期备份。

日常训练从[本地训练控制台](CONSOLE.md)开始。它串接正式 Rust 引擎、内存自对弈样本、PyTorch 持续学习器、历史对手、冻结评测和恢复点；随包教师通过嵌入式 V8 运行，训练进程无需 Node。

## 选择入口

| 需求                           | 入口                                                   | 文档                          |
| ------------------------------ | ------------------------------------------------------ | ----------------------------- |
| GUI 自动训练、暂停、继续、评测 | `python -m haojie_training.console` / `haojie-console` | [控制台](CONSOLE.md)          |
| 固定检查点采样、记录审核、恢复 | `python -m haojie_training.native`                     | [Rust＋Python CLI](NATIVE.md) |
| 教师轨迹转张量                 | `python -m haojie_training.prepare`                    | 下文；需要仓库 Node 依赖      |
| 张量训练、续训、评估           | `haojie_training.train` / `haojie_training.evaluate`   | 下文                          |
| ONNX 导出、设备基准            | `haojie_training.export` / `haojie_training.benchmark` | 独立开发工具，按需运行        |

现行模型和功能状态见[状态入口](../docs/ai/STATUS.md)，公开输入与动作分解见[编码协议](ENCODING.md)，TS/Rust 对应见[运行结构](../docs/ai/RUNTIME.md)。旧阶段实验、固定硬件成绩和交付报告从[历史索引](../docs/ai/HISTORY.md)查看。

## 安装与启动

在仓库根目录操作，需要 Python 3.12 和 Rust 1.98.1。先建立独立环境，再选择与设备对应的 PyTorch。

Windows / NVIDIA：

```powershell
python -m venv training/.venv
training/.venv/Scripts/python.exe -m pip install -r training/requirements-win-cuda.txt
training/.venv/Scripts/python.exe -m pip install -e "training[dev]"
cargo build --release --locked --manifest-path native/engine/Cargo.toml --target-dir artifacts/native-target
training/.venv/Scripts/python.exe -m haojie_training.console --device cuda
```

Windows / Intel XPU 使用 `requirements-win-xpu.txt` 替换 CUDA 清单，并选择 `--device xpu`。两份清单保留已记录的设备依赖；随后安装训练包会补齐控制台需要的 `psutil`、`mini-racer` 等运行依赖。已有环境可直接更新训练包。

Linux / CPU：

```sh
python3 -m venv training/.venv
training/.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
training/.venv/bin/pip install -e 'training[dev]'
cargo build --release --locked --manifest-path native/engine/Cargo.toml --target-dir artifacts/native-target
training/.venv/bin/python -m haojie_training.console --device cpu
```

控制台默认打开 `http://127.0.0.1:8765`，训练目录为 `~/.haojie-training`。已有引擎用 `--engine PATH` 指定；首次使用在页面新建训练，然后开始。暂停、恢复、导出和配额操作见 [CONSOLE.md](CONSOLE.md)。

`.venv`、模型、缓存和本地产物不提交 Git。下文 `python` 表示已激活的训练环境，或替换为上述完整 Python 路径。

## 模型与维护职责

`entity-transformer-v2` 为 6 层 Pre-LN Transformer：width 384、6 个注意力头、FFN 1536、dropout 0，共 11,695,874 个参数。64 维公开实体特征、类别嵌入和 32 维全局特征进入主干；64 维候选结合全局、来源及目标实体，输出候选评分和观察方价值。实体和候选数量可变，填充由掩码表示。候选投影及评分保留 FP32，主干和价值头使用所选精度。

控制台以 `decomposed-mc-q-v2` 让真实终局收益监督实际分解选择；记录 CLI 保持 `sampled-action-imitation` 的动作监督与终局价值语义。两条路径共享模型和运行时，各自保存方法及来源。

| 模块                                     | 职责                                         |
| ---------------------------------------- | -------------------------------------------- |
| `console/controller.py`                  | 后台生命周期、阶段切换、控制意图及内存状态   |
| `console/config.py`                      | 设置默认值、字段和范围校验                   |
| `console/storage.py`                     | 配额、原子写入、实验目录及恢复点读取校验     |
| `console/sampling.py` / `memory.py`      | 任务分配、终局样本留存与消费                 |
| `console/evaluation.py` / `opponents.py` | 冻结评测、历史模型和正式教师                 |
| `native/`                                | 引擎进程、二进制帧、批量推理、记录采样及审核 |
| `model.py` / `runtime.py`                | 网络、优化器、设备精度、检查点加载和恢复     |
| `data.py` / `prepare.py`                 | 张量校验、组批、分片及轨迹转换               |

`runtime.read_checkpoint` 一次读取检查点并检查格式及网络版本，各入口继续核对自己的规则、数据或运行条件。`checkpoint_config` 保留为仅读取配置的兼容入口。

## 教师记录准备

TS 教师、旧记录转换与网络 CLI 对战需要仓库的 Node 依赖：

```sh
npm ci
npm run train:selfplay -- --games 4 --seed 2026092201 --rules classic --difficulty easy --nodes 40 --plies 80 --commands 1200 --output artifacts/training/pilot/teacher.jsonl.gz
python -m haojie_training.prepare artifacts/training/pilot/teacher.jsonl.gz --output artifacts/training/pilot/encoded
```

`haojie-training-record-v1` 保存开局身份、命令、操作者、指纹、教师统计和结果。`prepare` 经同一个 TS 重放入口核对规则、权限和指纹，通过管道接收公开编码，直接写二进制 `train.pt` / `validation.pt`。支持多个 `.jsonl` / `.jsonl.gz` 输入；损坏数据明确失败。训练与验证按完整种子族隔离，拒绝重复对局；缺少终局的完整前缀只有策略标签。

- `--workers N` 并行编码输入文件，汇总保持输入顺序。
- `--shard-size N` 输出分片索引，取样时再按当前批次填充。
- `--split-manifest 原manifest.json` 固定已有验证族，新增族进入训练。
- `--teacher-difficulty hard` 只保留指定教师实际动作。
- `--search-policy` 保留旧 PUCT 轨迹的软访问分布，相应记录读取入口继续维护。

教师换边、并行采样、轨迹审计和纠错见 [TS 训练工具](../docs/ai/TRAINING.md)。参数详情见对应 CLI 帮助。

## 训练、续训与初始化

```sh
python -m haojie_training.train --data artifacts/training/pilot/encoded/train.pt --validation artifacts/training/pilot/encoded/validation.pt --device cpu --batch-size 8 --steps 20 --checkpoint artifacts/training/pilot/model.pt
python -m haojie_training.train --data artifacts/training/pilot/encoded/train.pt --validation artifacts/training/pilot/encoded/validation.pt --resume artifacts/training/pilot/model.pt --checkpoint artifacts/training/pilot/model.pt --device cpu --batch-size 8 --steps 20
```

`--steps` 是本次新增步数。`--resume` 恢复模型、优化器、AMP、累计步骤、随机状态及来源，要求数据、结构、目标和精度兼容。新一轮数据用互斥的 `--initialize-from 父检查点`，只继承兼容权重并重建优化器；父文件只读，祖先训练族继续排除在新验证集之外。

CUDA 可用 `--precision bf16 --threads 1`，`--optimizer auto` 选择受支持的执行方式。`--length-bucket-size 256` 可减少组批填充，设置随检查点保存。设备及确定性恢复条件见 [NATIVE.md](NATIVE.md)。

功能烟雾检查可以使用小模型和合成数据，合成标记随检查点保留：

```sh
python -m haojie_training.train --synthetic --tiny --device cpu --steps 2 --checkpoint artifacts/training/smoke.pt
```

`evaluate` 对兼容数据报告策略拟合、真实终局价值误差及按局汇总；实际棋力由冻结评测和独立对战衡量。

## 输入与产物

训练输入接受张量字典或 `haojie-training-shards-v1` 索引。索引绑定各分片 SHA256，读取时核对元数据、有限值、掩码和实体引用。八项模型输入为 `entities`、`kinds`、`entity_mask`、`globals`、`candidates`、`sources`、`targets`、`candidate_mask`；标签另含 `policy`、`value`、`value_mask`。完整维度和信息边界见 [ENCODING.md](ENCODING.md)。

控制台恢复点、原生记录、旧教师记录和游戏存档各自遵守原格式。恢复点含优化器及进度；导出的权重用于新实验初始化。既有权重、未跟踪轨迹和本地缓存由其原目录管理。

## 保留的开发工具

- `python -m haojie_training.benchmark`：按设备、精度、批量和线程测试实际网络，输出环境及输入指纹。
- `python -m haojie_training.export`：安装 `training[export]` 后导出 ONNX；`npm run bench:training:browser -- --data 导出目录` 启动本机推理检查页。
- `npm run train:match -- --checkpoint 模型 --output 新目录 --python Python路径`：Node 权威环境与常驻 Python 模型对战。保留 `greedy`、`beam`、`mcts`、`gumbel` 及带校准证书的 `value-*` 解码入口。
- 显式研究模型加载器和旧搜索轨迹编码器继续读取既有产物；已结束的夜间实验、一次性测量及采样驱动从[固定历史版本](https://github.com/Loremmlel/haojie2/tree/d920e181dccb425c0a77d57b473fb98f45163b69/scripts/training)重现。

## 维护验证

先构建正式引擎，再显式设置 `HAOJIE_NATIVE`。Windows PowerShell 使用 `$env:HAOJIE_NATIVE = (Resolve-Path artifacts/native-target/release/haojie-engine.exe).Path`；Linux 使用下列命令：

```sh
export HAOJIE_NATIVE="$PWD/artifacts/native-target/release/haojie-engine"
python -m unittest discover -s training/tests -p 'test_*.py'
python -m ruff check training/haojie_training training/tests
python -m ruff format --check training/haojie_training training/tests
```

引擎及教师资源检查、TS/Rust 规则和编码差分、安装包验证及浏览器流程随仓库 CI 维护，入口见 [CONSOLE.md](CONSOLE.md) 与[工程说明](../docs/ARCHITECTURE.md)。
