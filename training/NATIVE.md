# 共享原生训练支持与记录边界

日常自对弈、参数更新和恢复统一使用[本地控制台](CONSOLE.md)或 [Colab / Linux 服务](COLAB.md)。旧 `python -m haojie_training.native` 独立记录 CLI 已移除；原有记录、检查点、SQLite 账本和未完成前缀继续保留为历史产物。

## 安装与共享执行层

Python 3.12、PyTorch 和设备安装见[训练包说明](README.md)。正式引擎需要 Rust 1.98.1 和链接器，构建命令：

```sh
cargo build --release --locked --manifest-path native/engine/Cargo.toml --target-dir artifacts/native-target
```

Windows 程序为 `artifacts/native-target/release/haojie-engine.exe`，Linux 无 `.exe`。GUI 和云端服务均可通过 `--engine PATH` 指定程序；测试通过 `HAOJIE_NATIVE` 指定。`ENGINE --version` 输出源码身份、规则、编码、记录和协议版本。

`haojie_training.native` 保留共享内部组件：`client.py` 负责进程和二进制帧，`model.py` 负责规则元数据、模型加载和样本构造，`pool.py` 负责内存采样调度，`execution.py` / `packed.py` 负责模型推理。GUI、云端、历史对手和评测共用这些组件。安装后的日常训练使用 Python、Rust 和随包嵌入式教师，不需要 Node 或仓库 TS 源码。

现行恢复由控制台的恢复点和云端备份完成，保存 BF16 模型、补偿 AdamW、累计更新及随机源；RAM 样本和半局采样栈不进入恢复点。完整流程见[控制台生命周期](CONSOLE.md)和[云端备份](COLAB.md)。旧 resident 的 `tasks.sqlite` 属于旧任务账本，不作为现行恢复点导入。

## CUDA 的 C++ 输入拼装与算子调度

正式模型的紧凑推理将输入拼装、主干及候选评分的 ATen 调用放到 C++，复用同一组 BF16 权重和 CUDA 算子。读取端只交付公开帧和所有权，C++ 在批量拼装时统一检查数值、词表和指针边界，再写入复用的 pinned 缓冲。数组视图等到 Python 实际需要时才创建。有效候选的特征和最终读出索引在 CPU 压紧后上传，评分再写回原来的候选位置；补齐及已遮罩项保持 `-1e9`。末层查询只保留全局及有效候选引用，前层交互和完整 K/V 保留。双缓冲忙时继续收集就绪请求，再统一组批。模型结构、特征和 MC-Q 目标保持，更新通过共用的 BF16 补偿 AdamW 完成。

BF16 紧凑路径使用变长 Flash Attention，以及 CUDA 矩阵乘法与 tanh GELU 融合；参数直接引用原网络。融合对应 Torch 2.14 的 [`_addmm_activation` CUDA 定义](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/_decomp/decompositions.py)和 [cuBLAS GELU 公式](https://docs.nvidia.com/cuda/cublas/index.html#cublasltepilogue-t)。数值回归覆盖原网络、变长序列、两种注意力内核、融合开关及更新后的权重。

注意力输出的残差加法与前馈输入 LayerNorm 合用一个 CUDA 内核，输入、参数、残差和归一化结果均为 BF16，使用原 epsilon。该内核通过 Torch 随带的 NVRTC 编译，无须额外安装 CUDA toolkit；需要梯度、布局不适用或编译不可用时使用 BF16 ATen 算子。`normalization_backend` 和 `normalization_compile_error` 记录实际后端及可选编译诊断。

Colab 安装脚本在 CUDA 检查后自动编译。已有 Linux/WSL CUDA 环境在仓库根执行一次（需要系统 C++ 编译器，无须 nvcc）：

```sh
python -m pip install -e 'training[native]'
python -m haojie_training.native.aten
```

随后照常启动控制台或云端服务。默认只加载与 C++ 源码、Torch 分发版本匹配的缓存库，未构建时继续使用 Python 路径；正常启动不调用编译器。`TORCH_EXTENSIONS_DIR` 可指定本地缓存位置，构建与服务须使用同一个值。`HAOJIE_ATEN=0` 可作 Python 对照，`HAOJIE_ATEN=1` 则要求使用扩展，缺少匹配库时现场构建并明确报告失败。状态中的 `runtime.dispatch_backend` 为 `aten-cpp` 或 `python`。CPU、XPU 和非正式网络配置沿用原执行器。

整批 BF16 二进制评分在一次 C++ 调用中写回 Rust，避免 Python 逐条写管道的 GIL 切换；二进制 v2 协议的浮点字段使用小端 BF16，实体、候选和全局帧分别占 137、145 和 64 字节。32 个及以上环境的 C++ 采样窗口将 Python 线程切换间隔收紧到至多 0.5 毫秒，排空推理后恢复宿主原值，更新和保存不沿用该调整。

默认输入仍在主进程内处理；Linux 管道共用一个 asyncio 读取线程，以 `readv` 直接读入独占帧或共享槽，Windows 保留逐槽阻塞读取。独立输入进程接收、解析并共享原帧，在同一个任务收集器中完成身份核对与蓄水；主进程只接收推理、计数及完成的轨迹。蓄水先保留独立原帧，真实终局时才为最终入选行构造训练张量，随机选择、覆盖统计和内存计账保持原流程。每槽两块缓冲在扩容且旧租约归还后立即释放旧映射，并通知父进程关闭旧句柄；映射数量不随扩容次数累积。状态另列 `read_backend`、`input_layout`、`policy_layout`、`mean_entities`、`mean_scored_candidates`、`python_switch_ms` 和共享映射当前/峰值数量。进程数量按目标机器实际终局吞吐选择，见 [Colab 配置](COLAB.md#服务器配置与测量)。

`HAOJIE_LAUNCH_WORKERS=0|1|2` 可选择在常驻线程中发射 CUDA 工作，默认 0；诊断入口对应 `--launch-workers`。额外发射线程各自持有流，直接读取同一份 BF16 参数，排空后才允许更新。该配置与 CPU 输入进程数独立，按目标设备的实际终局吞吐单独选择。

## 原生协议与旧记录

Rust 的 `--training` 协议继续保留 `sample` 和 `audit`，供协议验证、差分与已有记录审核。移除的是 Python 独立采样／账本／分片 CLI；引擎记录格式和历史哈希保持原样。

| 契约           | 版本                                             |
| -------------- | ------------------------------------------------ |
| 规则包         | `haojie-rules-package-v1`                        |
| 二进制训练协议 | `haojie-training-binary-v2`                      |
| 原生记录       | `haojie-native-record-v1`                        |
| 编码           | `haojie-entities-factorized-v1`                  |
| 模型 / 检查点  | `entity-transformer-v3` / `haojie-checkpoint-v1` |

JSON 行控制与 LE 张量共同组成训练协议。模型只接收公开编码，每个接收帧独立持有字节区；采样仍校验请求身份、模型版本和有限输出。规则和采样算法的 TS/Rust 对应见[运行结构](../docs/ai/RUNTIME.md)。

错误结案的 `debug` 保存起点、出错时的权威局面、操作者、命令数与采样随机状态，供本地可信开发协议复现；该字段只出现在错误控制消息，不进入模型输入。正常采样不保留全局面轨迹。取消和预算截断继续保持未知结果。

原生记录保存逐行 SHA256 链和执行前后状态哈希。`audit` 校验规则、权限、命令和哈希，并可在同次重放中编码；审核回执绑定实际读取的原始字节、规则包、完整编码 schema 和审核器源码身份。完整行损坏明确失败，未完成尾部保留已确认前缀和未知结果；截断或取消不补终局收益。旧记录按原版本处理，不改写历史指纹。

宿主开发环境保留独立 TS 审核：

```sh
node --import tsx scripts/training/native/pipeline/audit.ts <记录.jsonl> <新报告.json> --summary-only
```

该工具需要仓库 Node 依赖；GUI 和云端训练不调用它。TS 手工教师、旧搜索记录转换和模型对战见 [TS 训练工具](../docs/ai/TRAINING.md)，其访问分布与已执行动作监督分别保持原有语义。

## 源码交付与验收

```sh
python scripts/training/native/pipeline/package.py --output <新源码包.tar.gz>
```

源码包包含 Python/Rust、提交规则包、教师资源、现行功能测试及固定晚盘前缀，不收集模型和训练产物。旁边的 manifest 记录成员哈希。[原生 CI](../.github/workflows/native.yml) 的独立安装验收运行现行训练、控制台和云端功能回归，覆盖实际更新、服务重启和备份恢复；宿主 job 保留 TS/Rust 规则、候选、编码与记录差分。性能复跑见 [PERFORMANCE.md](PERFORMANCE.md)。
