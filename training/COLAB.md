# Colab 与服务器持续训练

入口是 [notebooks/haojie_colab.ipynb](../notebooks/haojie_colab.ipynb)。在 Colab 的“文件 → 打开笔记本 → GitHub”中选择本仓库的 `main` 分支并打开它；合并后也可选择 main。Notebook 的 `REPO_REF` 决定下载的代码版本，首次安装会打印实际提交 SHA。

## 第一次使用

选择运行时，依次执行 Notebook 单元格；集中训练建议选择 GPU。只需确认 Drive 备份目录和需要导入的旧训练位置。Notebook、安装脚本及训练服务的设备默认均为 `auto`。安装脚本使用独立 Python 3.12，读取 `training/pyproject.toml` 中固定的依赖，通过 uv 的 `--torch-backend auto` 选择驱动适配的 PyTorch 分发；识别到 CUDA 后执行矩阵计算检查，失败时停止。Rust 使用与 CI 一致的 1.98.1 和 Cargo.lock 构建 release。Notebook 内核原有 Torch 不被替换。

源码、解释器、引擎、归档临时文件和活跃训练根分别位于 `/content` 下。所有普通采样、更新和本地恢复点写入都走本地磁盘，Drive 只作为版本化归档存储。安装完成后先以暂停状态启动，面板显示真实硬件、恢复更新数和配置，再点“开始”。无需公网隧道、SSH 或 iframe；Notebook 内的 Python 客户端访问回环 API，原有 Host/Origin/会话校验保持启用。

没有可用加速器时 `auto` 选择 CPU；也可显式设置 `DEVICE=cpu` 或 `DEVICE=cuda`。显式 CUDA 不可用时报告错误。默认模型保持原结构，Notebook 不默认使用验收用 tiny 模型。

普通 Linux 或 WSL 可将全部工具安装到独立目录。安装脚本尊重 `CARGO_HOME`、`RUSTUP_HOME` 和 `UV_CACHE_DIR`；未指定时 Cargo 仍使用 `~/.cargo`。新装 Rust 使用 `--no-modify-path`，脚本仅为构建子进程设置 PATH，不修改 shell 启动文件。主机 Python 需要可用的 pip；示例从仓库根执行：

```sh
env CARGO_HOME=/var/tmp/haojie-tools/cargo \
  RUSTUP_HOME=/var/tmp/haojie-tools/rustup \
  UV_CACHE_DIR=/var/tmp/haojie-tools/uv-cache \
  python3 scripts/colab/bootstrap.py \
  --venv /var/tmp/haojie-tools/venv \
  --target /var/tmp/haojie-tools/native-target
```

选择本地磁盘上不会随短任务退出而清空的目录；WSL 的工具和构建输出优先放在 Linux 文件系统。需要先安装 CUDA 依赖、稍后再验收设备时，可用 `--backend cu130 --device cpu`：该组合仍安装指定 CUDA 分发，仅跳过安装末尾的 GPU 矩阵检查。

尚未推送的工作区也可复跑：在本地执行 `python scripts/training/native/pipeline/package.py --output artifacts/cloud-source.tar.gz`，将归档上传到服务器的新目录后解压，并在解压目录运行上述安装命令。归档包含当前 Python/Rust 源码、规则数据、已构建教师与完整周期测量入口；同名 manifest 记录每个文件及归档的 SHA256，不包含本地模型、轨迹或缓存。

安装完成后可按第二轮固定配置验收正式模型的 64 进程采样、更新与保存：

```sh
/var/tmp/haojie-tools/venv/bin/python scripts/training/benchmarks/cycle.py \
  --engine /var/tmp/haojie-tools/native-target/release/haojie-engine \
  --starts fixtures/late-starts.json \
  --environments 64 --games 64 --updates 16 --pool-mib 4096 \
  --threads 1 --batch-size 8 --inference-batch-size 32 --batch-wait-ms 0 \
  --start-indices 0 2 --seed 20261004 --history-percent 0 \
  --device cuda --sampling-precision fp32 --ordered-updates \
  --output artifacts/cloud-cycle.json
```

`fixtures/late-starts.json` 是源码归档中的真实晚盘起点；直接使用 Git 检出时改为 `tests/fixtures/native/late-starts.json.gz`。上例使用经典和神龛的索引 0/2；4 GiB 是主机 RAM 样本池预算。复跑首轮全部四份起点的功能验收时，移除 `--start-indices`，并将环境数与局数设为 32。去掉 `--starts` 即从新局开始，CPU 服务器将设备改为 `cpu` 并显式指定测量线程数。结果记录实际终局、截断、样本哈希、模型/引擎身份及各阶段时间；配对方法见 [PERFORMANCE.md](PERFORMANCE.md)。

CUDA FP32 / BF16 正式模型在固定 Torch 2.14.0 下共用紧凑实体布局和双缓冲推理，计算沿用所选精度。GUI、Colab、历史对手和评测共用该执行器。采样仍在同一版权重下完成整轮，之后才做参数更新；暂停、保存和换模型前排空在途请求。统计中的 `packed_forwards`、`async_forwards`、`async_overlaps` 可核对实际采用的路径。CPU、XPU 和研究模型沿用普通推理，同时共享引擎、编码、训练及保存优化。

64 环境的动态批量会产生不同尺寸的临时张量。训练包在导入 Torch 前默认设置 `PYTORCH_ALLOC_CONF=expandable_segments:True`，降低 CUDA 缓存预留；已有任一分配器配置都会原样保留。如果嵌入宿主已先导入 Torch，应在启动宿主进程前设置环境，例如：

```sh
if [ -z "${PYTORCH_ALLOC_CONF+x}" ] && [ -z "${PYTORCH_CUDA_ALLOC_CONF+x}" ]; then
  export PYTORCH_ALLOC_CONF=expandable_segments:True
fi
```

本机 Windows 和 Linux 的 Torch 2.14 / CUDA 13 验收均无不支持警告；64 环境的缓存预留峰值约为 0.62 GiB，Windows 普通分配器的同负载测量约为 2.28 GiB。实际张量峰值约为 0.58 GiB，驱动与其他程序占用另计。默认值不指定 CUDA 专属的 `backend`，不导入或探测 CUDA；该选项不写入训练恢复点。完整结果见 [PERFORMANCE.md](PERFORMANCE.md)，比较代码速度时双方使用相同的分配器配置。

## 从已有成果继续

| 已有成果                  | 参数与行为                                                                                                                                                                                                     |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 本 Notebook 的完整备份    | `RESTORE_SOURCE` 留空；空本地训练根会从 `DRIVE_BACKUP_DIR` 自动选最近的完成清单，损坏时尝试前版。                                                                                                              |
| 电脑 GUI 的完整训练根     | 先停止电脑上的训练服务，把整个受管根复制到 Drive，将目录填入 `RESTORE_SOURCE`。复制活动清单、各实验、恢复点与历史权重；原目录不修改；不支持系统锁的远端挂载采用只读复制和前后清单/哈希核对，仍须先停止原服务。 |
| 某个完整恢复点 `.pt`      | 将文件填入 `RESTORE_SOURCE`。继承其中的优化器、随机状态和累计进度；缺少该根的历史/基准文件时按控制台原有逻辑重建。                                                                                             |
| 只有模型权重              | `IMPORT_WEIGHTS` 指定兼容权重；建立新优化器与新学习进度。它与完整恢复二选一。                                                                                                                                  |
| 原始对局 JSONL / 训练分片 | 原生旧记录保留 [NATIVE.md](NATIVE.md) 所列的独立审核方式，已有分片继续使用[离线训练](README.md)。恢复持续训练需要完整恢复点。                                                                                  |

完整恢复保留模型、优化器、精度、已保存的随机状态、任务/更新计数、设置、活动实验，以及归档中的历史、基准、最佳、导出和小型指标。RAM 样本池和在途对局重新生成；恢复后显示实际保存的更新数。融合 CUDA 优化器须在兼容 CUDA 模式恢复，跨不兼容设备迁移用新实验继承权重。

非空本地根会继续使用本地状态，不自动被 Drive 覆盖。要显式恢复另一份归档，应选择新的 `LOCAL_ROOT`，保留原根。手工导入完成后，下一次启动可清空 `RESTORE_SOURCE`，继续该本地根。旧恢复点只补齐新增的三个执行选项，原有学习参数保留；Notebook 的采样/更新设置中，数字 0、“沿用”和等待时间 -1 表示不覆盖已有参数。`TORCH_THREADS=0` 是单独的服务启动选项，表示按本次设备和有效核配额自动选择线程数。

## 周期备份与退出

默认每 10 分钟触发，保留最近 3 个完成版本。请求先交给训练控制器：在采样反馈、更新步、评测片段或空闲边界保存完整恢复点；固定这些已发布文件后，再由备份线程压缩、计算 SHA256、复制到 Drive 并回读校验。训练张量不会被上传线程并发读取。正常同盘冻结用硬链接，跨盘才复制。

每个备份包含一个 `.tar.gz` 和同名 `.json` 完成清单。归档先传到 `.partial`，校验通过后发布，最后发布清单；恢复只认匹配的完整清单和逐文件哈希。上传失败保留上一版本，后续周期重试。新版本完整发布后才轮换此备份目录中本工具拥有的旧版本，陌生文件不清理。建议一个备份目录只由一个正在运行的服务写入。

控制面板提供“备份到 Drive”，显示排队、归档、上传、完成或错误。周期计时属于训练服务，与面板是否刷新无关。“暂停刷新”只停止读数，训练继续；离开前运行“正常退出”单元格，服务会暂停、保存并等待最后一份备份。强制回收 VM 时使用上一次完成的远端备份。

第 3 步以独立会话启动服务，标准输入与 Notebook 内核断开；第 4 步面板中断产生的进程组信号不会送到训练服务。连接失败会保留上次图表并显示“状态未知”和日志位置，重新运行第 3 步检查服务；已退出的子进程先回收再判断存活，随后可从本地训练根重新启动。旧版已启动的服务需要正常退出后重新执行第 3 步，才能采用新的启动方式。

第 4 步合并控制、配置、监控和评测；第 5 步正常退出。面板展示策略/价值损失、有效终局样本吞吐和 CPU/GPU 利用率图表，以及实际并发、累计更新、内存、显存和备份状态。损失读取服务保存的更新记录；吞吐与利用率保留本次监控最近 180 次刷新，缺失读数留白。详细 JSON 折叠在底部，供排错使用。

第 4 步显示面板后立即结束执行，浏览器经 Colab 官方回调轮询与提交操作，避免长期执行单元格阻塞按钮消息。更新保留原有页面节点、折叠状态、焦点、输入草稿和滚动位置；详细 JSON 展开后固定该快照，收起后继续更新，并使用独立滚动区阅读长记录。并发输入直接读取提交时的数字，收到服务回执后立即回读状态；只有目标值匹配才显示“服务已确认并发目标”，实际进程仍在下一采样窗口切换。

重复运行第 4 步会使旧面板失效，旧输出无法再提交控制请求；“暂停刷新 / 继续刷新”控制面板读数。正常退出前也会关闭面板回调。回归同时覆盖真实回环鉴权、并发提交与进程切换，以及多次刷新中的展开、选区、焦点和滚动，见 `tests/browser/console/notebook.mjs`；测试替代 Colab 的消息传输，训练服务、Python 面板回调和 Rust 引擎均使用正式实现。

自动评测在新增有效训练终局达到 `eval_games`、且模型有新更新后触发；新实验默认阈值为 `max(64, 环境数 × 4)`。面板显示实际阈值和距下次触发的终局数；“其他设置”中可在暂停后调整自动间隔、每规则换座对数和每局时限。每个换座对数对应三档手工 AI 在经典/神龛中分别执先后手各一局，加初始模型对照，共 16 局；默认一对。已有实验沿用保存值。

“立即评测”可提前排队，“取消评测”取消该批任务。训练运行时，评测与训练按时间片交替，每次最多评测四局再训练一轮；整批评测绑定冻结权重。暂停训练后也可单独执行评测。采样、更新和评测继续串行使用设备。

面板展示本轮评测进度、最近完成批次，以及各难度的总胜率、模型先手/后手胜率和有效局数；展开可查看经典/神龛与先后手组合。胜率分母仅包含真实终局，平局计入分母，超时/截断/待测及错误单列。实时批次与已完成批次分别显示，保留各自的模型身份和更新数；旧摘要缺少逐局记录时仅展示已有总计。

训练根仍沿用总计 20,000,000,000 字节配额。归档暂存和 Drive 的多个版本另占空间；空间不足时明确报错并保留已有恢复点。恢复拒绝覆盖非空目录、链接/越界条目、重复归档条目和哈希不匹配的内容。

## 服务器配置与测量

| 设置                   | 范围 / 含义                                                           |
| ---------------------- | --------------------------------------------------------------------- |
| `environments`         | 1–128，MC-Q 常驻 Rust 对局进程数，与每轮完成目标独立。                |
| `games_per_round`      | 1–4096，无可消费样本窗口的完成任务上限；有终局样本即学习。            |
| `inference_batch_size` | 1–256，单次推理请求上限；按模型分组，普通推理另按实体长度分桶。       |
| `batch_wait_ms`        | 0–10，额外组批窗口；默认 0，先测吞吐再增加。                          |
| `batch_size`           | 1–256，梯度更新批量；与推理批量独立。                                 |
| `sampling_precision`   | FP32 / BF16；默认 FP32，BF16 为显式选择并检查设备支持。               |
| `pool_mib`             | 16–131072，RAM 样本池预算；模型、进程和缓冲另占内存。                 |
| `--torch-threads`      | Torch CPU 算子线程数，默认 0 自动选择；不限制 Rust 子进程使用其他核。 |

GUI、cloud 和 Notebook 的新实验共用 `defaults(device)`，根据 CPU affinity、可见 cgroup 各层 CPU 时间配额、RAM 限额与可回收缓存生成初值。CPU 取有效核数，CUDA/XPU 取其两倍，向上取整且最多 128 环境。可用 RAM 先分配四分之一给样本池（64 MiB 至 32 GiB），再扣除 512 MiB 的模型等起步空间，剩余预算按每环境 256 MiB 限制并发。每轮局数为环境数的两倍，推理批量为 `min(32, 环境数)`，训练 batch 保持 8。设置中的显式值最后覆盖自动结果。

初值在创建实验时确定；恢复实验保留保存的学习设置、模型和优化器。默认 MC-Q 持续补局，有真实终局产生可消费样本后即在下一决策根停驻并更新，其他长局原地保留；继续时先消费 RAM 中已有样本。`games_per_round` 负责让无样本的截断窗口归还控制。GPU 按采样、更新分阶段运行。完成任务包含真实终局和预算截断，只有真实终局产生训练样本。单局命令与回合预算保持原值。暂停或退出会取消在途局，磁盘恢复点仍只保存学习状态；RAM 样本与在途对局不落盘。

第 4 步的“并发环境 → 应用并发量”支持运行时提交 1–128，使用独立的 `concurrency` 控制命令，仅接受 `{"environments": 整数}`。实际切换在当前采样窗口和更新完成后的下一采样窗口进行；监控区分别显示实际进程数、请求目标及“待生效”。切换重建采样进程，未终局沿用暂停的取消记账，不产生训练标签；模型、优化器、累计更新和已完成样本保留，每轮局数与推理/训练批量保持所设值。目标写入当前配置，随后保存的恢复点会携带它；其他设置仍先暂停再修改。

当前方在跨轮续局时使用最新模型，同一次命令分解的推理固定权重。历史对手整局冻结，内存中最多驻留 `history_size` 份；达到上限时，新局复用已有历史对手。控制台“保留待续”表示对局将在更新或评测后继续，旧模仿方法仍按有限批次运行。

Torch 线程数独立于自动环境配置：控制台、cloud CLI 和 Notebook 默认均为 0（自动），CPU 使用 `min(4, max(1, floor(有效核配额)))` 个算子线程，CUDA/XPU 使用 1 个；算子间线程数固定为 1。Notebook 的 `TORCH_THREADS` 和 CLI 的 `--torch-threads` 可指定正整数覆盖。GUI 新实验选择其他设备时自动重新选择线程数，显式正整数仍保持。线程策略不写入设置 JSON 或恢复点；嵌入调用与性能基准继续使用调用者指定的线程数。多个引擎进程和模型算子共同使用 CPU。

Python 输入链路可独立分进程：Notebook 第 1 步的 `HOST_WORKERS=2` 或 `4`，以及服务器启动环境 `HAOJIE_HOST_WORKERS=2` 或 `4`，将 Rust 收帧、校验和紧凑索引准备分摊到对应数量的 CPU 进程；模型、历史权重和优化器仍由主进程拥有。每槽两个共享内存缓冲按实际帧增长，输入在归还前保持有效，入选训练样本独立复制。评分按输入进程合并回传，队列只传元数据与评分。默认 `0` 沿用线程路径：本机 CUDA 配对尚未得到稳定的完整周期增益，L4 应先分别比较 0/2/4。该执行选项不写入训练设置或恢复点，需重启服务生效；Notebook 须先运行第 5 步退出，再重新启动。

原始状态中的 `runtime.host_workers`、`host_worker_pids` 和 `host_prepare_seconds` 可确认进程路径，`resources.processes` 查看每个 Python 进程的 CPU。`send_seconds` 在进程路径计主进程提交回包的时间；读帧、校验和输入准备是跨线程/进程累计时间，可以重叠。结合阶段墙钟、实际推理批量和终局样本/秒判断吞吐，不能把这些累计时间直接相加。

固定同一个检查点、每轮完成目标、模型、训练批次和种子，分别测 4/8/16/32 环境。面板报告累计/最近一轮采样墙钟、请求数、真实终局、有效保留样本、平均实际推理批量、平均活跃环境数和跨轮保留局数，另报告互斥的采样/更新/评测/保存阶段时间。比较持续池时同时观察首次更新等待时间、后续更新间隔和真实终局样本/秒；不同完成任务集合分别列出。CPU 统计来自本服务进程树并按有效配额归一化，包含已回收子进程耗时；无权限读取时显示未知。GPU 报告利用率、分配与保留显存。

改变训练 batch 会改变自动更新步数，改变窗口上限会改变无样本时的控制归还频率；比较训练效果时同时记录这些参数。显存容量和逻辑 CPU 数不是自动并发目标，优先比较有效样本/秒和完整周期耗时。

普通 Linux 服务器也可直接运行：

```sh
python -m haojie_training.cloud \
  --engine artifacts/native-target/release/haojie-engine \
  --root /path/to/local-training \
  --backup-dir /path/to/persistent-backups --backup-seconds 600 --backup-keep 3 \
  --connection-file connection.json
```

以上命令自动选择计算配置并开始训练；添加 `--paused` 可先查看配置。需要覆盖时再添加 `--settings server-settings.json`，文件只填写要调整的字段，例如：

```json
{
  "environments": 16,
  "games_per_round": 32,
  "inference_batch_size": 32,
  "batch_size": 8
}
```

## 验证与平台资料

纯 Python 回归覆盖组批所有权、协议引用身份、CPU/RAM 配额、旧设置升级、备份上传中断、损坏回退、源目录保护和快照生命周期。原生 CI 运行真实模型更新、归档、重启、优化器逐张量比较、续训，以及 32 个 Rust 进程的集成测试。Notebook 单元格接受语法检查；安装和依赖解析失败会保留日志而不是偷偷更换仓库版本。

2026-10-20 已在 WSL Ubuntu 24.04 的独立目录完成 Python 3.12.3、Torch 2.14.0+cu130、Rust 1.98.1 安装、依赖一致性检查及 release 引擎构建，并通过 Notebook 语法与异步监控单测。安装记录位于本地 `artifacts/performance-20261020/linux/bootstrap.log`，依赖与引擎身份见同目录 `environment.json`。

最终源码在同一 Linux 环境通过 CPU 小模型 32 进程的备份恢复验收（实际峰值 32），以及 CUDA 正式模型的双进程恢复验收。CPU 采样和更新均为 FP32；CUDA 采样为 FP32、更新为 BF16，这是基线和当前实现共有的默认精度。两者均逐项核对 Adam 状态、Python/Torch CPU/CUDA 随机状态、累计样本账目与空 RAM 样本池，再重新采样并继续更新。日志分别为上述目录中的 `final/cpu32-tiny-recovery.log`、`final/cuda-full-recovery.log`。现有 `CloudTests.test_real_training_backup_restart_and_optimizer_continuation` 可用 `HAOJIE_TEST_ENVIRONMENTS`、`HAOJIE_TEST_DEVICE=cpu|cuda` 和 `HAOJIE_TEST_FULL_MODEL=1` 选择验收配置；默认仍为 CPU 小模型。该生命周期测试保留 32 命令预算。

正式模型还完成了全部四份晚盘起点的 32 进程完整周期：32 局均真实终局，保留 3800 条样本、完成 16 次更新与保存，容量淘汰为零。报告为 `final/cuda32-late.json`；性能配对与自然开局记录见 [PERFORMANCE.md](PERFORMANCE.md)。

真实 Colab GPU 性能和实际 Drive 挂载仍须在用户分配的运行时验收；本地目录模拟测试验证的是备份协议与续训状态。

官方资料：[Colab FAQ](https://research.google.com/colaboratory/faq.html)、[uv 的 PyTorch 后端选择](https://docs.astral.sh/uv/guides/integration/pytorch/)、[Rust 安装](https://www.rust-lang.org/tools/install)。Colab 资源随分配变化且 VM 有生命周期；其免费运行时另列棋类训练、分布式 worker 和绕过 Notebook UI 等限制。这里不做保活、反回收或配额规避；长期集中训练应使用符合平台条件的运行时。
