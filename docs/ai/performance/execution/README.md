# Python / PyTorch 执行链路，2026-10-04

本轮在已合并转正成果的 `a1679bb` 上实施，分支 `codex/pytorch-execution`。Rust、规则、公开信息、编码、模型结构、损失、样本权重和 AdamW 数学定义均未改动。默认模型 **11,695,874 参数**；小模型只承担回归。代码、固定请求与连续更新、实际 Rust 闭环均已运行，不是后续研究计划。

推荐：**采样 CUDA FP32、8 环境、零额外组批等待；更新 CUDA BF16 + fused AdamW + 单批预取**。四环境同样受益。要求逐位恢复时显式开启 `--deterministic`；恢复自动继承该设置。CPU 回退保留；本轮只在起始/收尾做 CPU 模型测量，实施期间以 CUDA 为主。

## 实施与瓶颈

- 原路径已有 inference_mode、SDPA、二进制传输、mmap 和就绪调度，未将这些原有能力计为新增成果。
- 采样以前为每次请求构造 one-hot policy、value/value_mask，重复运行训练边界校验，搬运标签，计算并回传未使用的价值。现在训练和推理共享填充约定，推理只构建八个输入张量；`encode/score_policy` 共用原网络，`policy_logits` 跳过价值头。全部原参数名与权重兼容，评分头继续 FP32。
- 二进制帧直接 `readinto` 独占 bytearray，NumPy / Tensor 共享该帧内存，省掉分段 bytes→bytearray 扩容和八份数组复制。外部调用 decode 时仍先取得独占副本。帧所有者由张量引用持有，后续消息不会覆盖旧输入；刻意没有引入接收帧池，避免审核保留历史张量时出错。
- 协议边界直接在 NumPy 视图校验有限数、种类、有效候选、指针范围和填充实体；形状/dtype/字段由固定帧布局限定。训练文件加载仍核验哈希及全部输入/标签，内部可信抽样不重复扫描。没有全局关闭验证。
- 主机推理缓冲按连续一维容量区复用，CUDA 使用页锁定内存与非阻塞上传；一次 logits 回传完成之后才复用主机区，不逐算子同步。收尾发现多维容量切片会在上传时生成 pageable 副本，已改成连续前缀视图；针对性 profiler 确认八次上传全部为 Pinned→Device（该小批合计 18.0μs），一次结果下载 1.8μs。没有缓存不同动作前缀的主干结果。
- 读取线程直接提交事件队列，去掉每个 receive 的 Future/线程池二次转交。就绪节点按实体长度分组，最大实体长度倍率不超过 2；不等待尚未就绪的其他环境。0.5ms 等待已做反例实验，保留零等待默认。
- mmap 分片加载时一次准备末尾有效位置和稳定长度排序。抽样仍使用原锚点/桶内随机算法、原边际概率、原累计 step 种子；有孔掩码不误按有效数量裁剪。单工作线程预取下一批，最多当前/下一批两份主机数据，未引入多进程 DataLoader。
- CUDA FP32/BF16 使用 fused AdamW；CPU、XPU 和 FP16 保持 foreach。FP16 保留原 GradScaler 跳步计数，避免 fused step 钩子将溢出误报成有效更新。检查点保留实际优化器组、AMP、CPU/CUDA 随机源及累计步骤，并补充确定性/矩阵精度设置；显式不相容恢复会报错。

初始 CUDA FP32 算子剖析中，矩阵乘法约占 GPU 自耗时 66%，注意力约 23%；主成本是实际计算。Windows 安装的 PyTorch 使用 memory-efficient SDPA，`is_flash_attention_available()` 为 false；没有把已有 SDPA 重新开启当优化。更新热点仍是矩阵乘法和注意力反向。

## 输入、设备与测量边界

- Windows、Ryzen 7 5800H（8 核 / 16 线程）、RTX 3060 Laptop 6GB，驱动 610.88，PyTorch `2.14.0+cu130`。CUDA 宿主线程 1，interop 1；不改全局 FP32 matmul precision 来冒充同精度收益。
- 冻结默认模型：`artifacts/engine-20261003/model-default-initial.pt`，SHA256 `3b4b39edac99743170303855c7314244c2b1937ac22a7b0423a9f7913d5915ea`。这是已有未训练初始化检查点，不据此宣称已训练模型的排序稳定性或棋力。
- 原生引擎沿用 `artifacts/engine-20261003/final/engine.exe`，SHA256 `2c71554285a1790171f06e57a32dc8158393a9149281e141d11bdc0f438594eb`，未重新编写或更换内核。
- 从已审核的 `default-model-profile/profile/data/train.pt`（1,182 样本）按每个真实轨迹等距抽四个节点，冻结 84 请求，覆盖经典/神龛、开中晚盘。共有 20,238 实体、2,773 候选。按实体/候选长度排序成 21 个四请求批，批最大形状从 6 到 420 实体、2 到 185 候选。保留来源 records/indices；不重复生成对局。
- `artifacts/pytorch-20261004/requests.pt` 为 6.4MB，SHA256 `52d8ec8766240f8d995ace9ccb7ac980c4723fe19523ce6a5cb72b4be8493cdb`。有效实体填充率 98.17%，候选 40.68%；没有通过删候选或裁剪有效实体提速。
- A/C 同进程加载并复用相同检查点，原实现从开工前完整冻结包导入。预热、参考输出、梯度检查、正式计时分开。三组顺序为旧/新/BF16、BF16/新/旧、旧/新/BF16；计时没有并行编译或回归。
- 桌面、浏览器等后台程序保留。首次正式短测开始的 GPU 快照为利用率 33%、51°C、81.6W、1972MHz、1275MiB；缓冲修复后为 26%、50°C、89.9W、1972MHz、1486MiB。这些是瞬时快照，不是后台负载全程追踪；后组绝对吞吐下降，不能单凭快照归因。早期一轮优化 FP32 有 171ms 抖动，保留在 `group-cuda/`，未选作最终倍率。初始 CPU 两配置意外重叠，原结果保留但不用于跨设备结论；收尾重新串行测量。

## A：固定真实请求

最终连续缓冲修复后的 `final-contiguous/report.json`，每配置每轮 63 个四请求批、252 请求；表中吞吐为三轮中位，倍率为三组成对倍率中位，故与中位吞吐之商可能稍异。

| 执行方式                | 请求/秒 | 对旧 CUDA FP32 | 四请求批 p95 范围 |
| ----------------------- | ------: | -------------: | ----------------: |
| 旧 CUDA FP32            |  378.65 |         1.000× |     14.46–15.26ms |
| 新 CUDA FP32            |  463.49 |     **1.238×** |     12.23–13.49ms |
| 新 CUDA BF16，评分 FP32 |  567.44 |     **1.474×** |       8.51–9.86ms |

按同一有效输入折算，三轮中位约为旧 FP32 91,227 实体/秒、12,500 候选/秒；新 FP32 111,667 / 15,301；BF16 136,711 / 18,732。它们是模型处理量，不能当作游戏决策或新增命令吞吐。

最终 FP32 三组成对倍率 1.103/1.238/1.258，BF16 1.474/1.338/1.634，保留波动。修复前 `final/` 的旧/新/BF16 为 469.00/526.19/774.97 请求/秒、成对中位 1.120/1.666×；两个阶段不能连乘，也不能从后组绝对速度较低反推修复有害。发现具体副本问题后只追加本组 A 与八环境 B，不重跑更新全链。

FP16 筛选值 741.74 请求/秒，BF16 765.68；页锁定复用约 510.49，对 pageable 485.08。仅作组合选择，未对所有线程/精度/批大小做笛卡尔积。CPU 收尾同输入：1 线程旧/新 12.062 / 12.049 请求/秒，4 线程 31.682 / 31.705。CPU 同设备收益约为零；最终 CUDA FP32 参考约为 CPU 四线程参考的 12.0×，包含设备差异；新 CUDA BF16 约为 CPU 四线程参考的 17.9×，同时包含设备、代码和精度差异。CPU/GPU 未交错测量，不能忽略时段背景差异。

## B：Rust + Python 实际闭环

主体验收 `closure-final/report.json`。八个固定起点（四个自然开局、四个中晚盘），每局最多新增 24 命令；每个配置实际 **178 新命令、554 模型请求**。两个晚盘已终局，余下截断继续为 unknown。四环境分两批启动，八环境一批；因启动/加载次数不同，跨并发倍率不能全部归于组批。

| 配置            | 采样端到端秒 | 新命令/秒 | 对同并发旧 FP32 |
| --------------- | -----------: | --------: | --------------: |
| 旧 FP32，4 环境 |        4.126 |     43.14 |          1.000× |
| 新 FP32，4 环境 |        3.247 |     54.82 |      **1.271×** |
| 新 BF16，4 环境 |        3.420 |     52.05 |          1.207× |
| 旧 FP32，8 环境 |        3.348 |     53.16 |          1.000× |
| 新 FP32，8 环境 |        2.584 |     68.88 |      **1.296×** |
| 新 BF16，8 环境 |        2.752 |     64.68 |          1.217× |

采样墙钟包括进程启动、模型加载、前缀输入/重放、实际采样和记录发送；原生审核另为各配置 1.07–1.11 秒，未从采样数字中暗扣。六配置逐局 finalHash 全部相同。没有以完整长对局跑三轮倍率；这是小型闭环对照，不是稳态海量环境上限。

连续缓冲修复后的针对性八环境复核 `closure-contiguous/`：旧/新 FP32 为 **4.413→3.253s、40.33→54.72 新命令/秒，1.357×**，各侧仍为178命令/554请求、同 finalHash，审核各1.08s。后时段旧基线也明显变慢；不把早组旧基线和后组新结果拼接。最终平均批1.242、实体/候选填充率95.91%/69.43%，上传+模型+回传2.421s、组批0.226s、读取0.038s、解码校验0.117s、等队列0.104s、发送0.034s。保留此前完整并发/精度矩阵作为配置依据。

新 FP32 八环境的分账：上传+模型+同步回传 1.858s，主机组批 0.130s，管道张量读取各线程累计 0.031s，解码校验累计 0.094s，主循环等队列 0.128s，发送结果 0.027s。平均批仅 1.237，实体/候选填充率 97.16% / 69.62%。线程、设备和等待有重叠，这些量不能全部相加。CUDA 正常运行只在整批结果回传处等待；细粒度事件与 profiler 仅在诊断中使用。

四个历史前缀共 **19,864 条旧命令**，不计入新增吞吐。单独首请求取消探针 `prefix/report.json`（不调用模型）测得，2295/4571/4339/8659 条前缀的首请求就绪分别为 0.086/0.203/0.197/0.438s；这是含记录头、首次查询/编码的前缀成本上界，非纯重放时间。八个进程握手各约 20–25ms。这些串行探针不能直接从并发墙钟扣除。

首轮四环境 FP32 加 0.5ms 等待，3.316→3.518s，响应 p95 上升；因此默认零等待。离线 A 的大批 BF16 收益没有转化到当前小批闭环：AMP 转换/启动开销仍存在，推荐采样 FP32。未继续调几十种等待参数。

## C：连续参数更新

每轮同一起始模型、同一分片和固定 96 次索引序列，每次四样本，384 有效样本；三轮均实际完成全部更新。沿用原长度桶 64 抽样算法；新旧都使用相同桶和每步随机源。每组预热四步后重置权重与优化器，再正式计时。包含选样/填充、上传、完整前反向、裁剪、AdamW、AMP 和逐步有限 loss 检查；不含加载、验证、保存。

| 配置                   | 样本/秒中位 | 更新/秒中位 | 对旧 CUDA FP32 |
| ---------------------- | ----------: | ----------: | -------------: |
| 旧 FP32 / foreach      |      124.63 |       31.16 |         1.000× |
| 新 FP32 / fused + 预取 |      145.03 |       36.26 |     **1.172×** |
| 新 BF16 / fused + 预取 |      208.37 |       52.09 |     **1.669×** |

新旧数据加载 0.421/0.399s，缓存长度/排序没有让加载更快。稳态 GPU 诊断中，优化器自身约 3.18→1.53ms；完整确定性诊断大批的事件分段另为：旧/新 FP32/BF16 上传 1.66/1.83/1.34ms，前向含损失 8.82/9.35/8.48ms，反向 22.60/23.82/13.30ms，裁剪+优化器+scaler 4.37/2.07/2.24ms。事件分段包括流内派发间隙、形状不同于全组平均，不代替主表。

确定性模式单组短测为旧/新 FP32/BF16 117.0/130.2/145.7 样本/秒，有明确成本，不将其恢复验收和非确定性最快吞吐混为一个配置。默认保持 PyTorch 非确定性执行；严格逐位恢复使用 `--deterministic`。

实际训练 CLI 的恢复验收使用既有 147 训练/31 验证样本（74/16 个真实价值标签），每次均保留训练和验证集前后全量指标、检查点保存。加载约 1.86–2.00s、前验证 0.77–0.78s、后验证 0.37–0.39s、保存 0.15–0.16s，分别报告。两步/四步仅作为恢复验收，不用作训练吞吐。

## 数值、恢复与兼容

- A 预先设 FP32 `atol=2e-6, rtol=2e-5`；FP16 `0.002/0.02`，BF16 `0.01/0.05`。新 FP32 的冻结 logits/value 最大误差均 0；掩码/填充/输入所有权回归通过。默认模型梯度相对 L2 为约 `1.03e-7`，阈值 `2e-5`。
- BF16 logits 最大绝对误差 `0.0006732`，value `0.0031400`，梯度相对 L2 `0.01039`（预定上限 0.05）；84 个 top-1 无变化，**32 个请求的完整候选排序发生变化**。FP16 筛选 logits/value 最大误差 `0.00006618/0.0003631`，top-1 无变化。没有据此承诺任意训练权重或未来轨迹都一致。
- 首次非确定性 BF16 另进程 2+2 与连续 4 步不逐位相等，最大参数差 `0.0007823`；模型步数、元数据、优化器组和 CPU/CUDA 随机源相同。失败现场 `recovery/` 保留。启用确定性后，`recovery-deterministic/` 的模型、优化器、scaler、累计步骤与更新、CPU/CUDA RNG、metadata **全部逐张量相等**；新权重实际采样和原生审核通过。
- 32 项 Python 测试通过、无跳过；包含新 CUDA 输出/梯度、缓冲所有权、非法帧、预取索引、fused/foreach 一步容差与恢复。连续缓冲修复后另外3项CUDA/协议回归通过。收尾 CPU 原有训练/恢复/原生审核回归通过。纯 Python 变更没有重跑 TS/Rust/浏览器全套；检查发行一致性，不部署页面。
- `deploy:check` 原命令因工作树根 HTML 的131处CRLF失败；进一步核对，规范化后的工作树与构建相等，**HEAD中Git原始LF文件与843,066字节构建逐字节相等**，SHA256均为 `24e573878236fc2f4408ad0dd0ca5d6acb12f784a9ce73b373c253474d2ea3e5`。这是既有检出换行差异，未改根HTML、构建脚本或部署；不把原命令写成通过。
- 仍支持 Rust+Python 独立运行，新增依赖均为已有 torch/NumPy 或标准库；源码包递归包含 batching 子包。保留 CPU foreach 回退及旧检查点读取。新 fused 优化器检查点要求 CUDA 原执行模式恢复；需要跨设备时可以单独继承模型权重开启新阶段，不能冒充原优化器连续恢复。
- Linux 隔离无 Node 安装验收仍使用原 CI，未把本机原生采样冒称新一轮隔离容器证明。XPU、其他 CUDA 架构/平台本轮未测。

## 未采用方案与剩余成本

`torch.compile(fullgraph=True)` 在本机实际尝试 7.218s 后因缺少可用 Triton 失败；没有产生成功图，更没有稳态编译收益数字。保留原错误，不自动安装非官方 Windows 编译栈或静默退回后宣称成功。官方参考：[compile](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html)、[SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)、[AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)、[确定性](https://docs.pytorch.org/docs/2.14/notes/randomness.html)。

没有强行加入 CUDA Graph、共享内存、近似主干缓存或 DataLoader 多 worker。剩余明显成本为默认模型的矩阵乘法/注意力反向、闭环大多单请求的启动开销、数据全量验证与检查点装载；候选填充仍可改善。它们是后续可能的工程点，不宣称已证伪所有高级优化。

固定请求脚本记录 3,300 次推理调用，另有4次连续缓冲传输探针（含预热/诊断，不含独立参考值/梯度和训练前向）；两组短闭环配置验收及具体副本问题的针对性复核合计 7,202 请求，恢复新权重另 13 请求。闭环配置合计 13 组八起点短切片，未启动长时间自对弈；首次路径错误在创建引擎前失败，不算成功链路。完整默认模型更新计时为筛选 4×32、正式 9×96、确定性诊断 3×32 步，均另计预热。

截至收口的已记录计时窗口约 **218 秒**，其中起止 CPU 约 110 秒、闭环含审核约 58 秒；不含进程加载、部分预热、功能回归及编辑分析时间。算子探针和失败 compile 分列；不是整轮工作耗时。全部工作约四十分钟量级。没有反复长对局或重新生成训练库。

执行性能已足够进入下一轮**独立、可证伪的学习有效性研究**；当前结果只证明计算更高效，不证明监督设计、价值信号或棋力问题已解决。继续保留 unknown 和未晋级模型边界。

## 启动与复现

PowerShell 7，在仓库根目录；以下默认模型与分片均为本地保留产物，输出目录须选新路径。Windows 沿用 `training/.venv/Scripts/python.exe`，Linux 换自己的已安装 CUDA Python 与原生引擎路径。

```powershell
$env:PYTHONPATH = 'training'
$env:PYTHONUTF8 = '1'
$py = 'training/.venv/Scripts/python.exe'
$engine = 'artifacts/engine-20261003/final/engine.exe'
$checkpoint = 'artifacts/engine-20261003/model-default-initial.pt'

# starts.json 支持1至8环境；生产采样推荐FP32，不加等待。
& $py -m haojie_training.native --engine $engine --threads 1 sample --checkpoint $checkpoint --starts starts.json --output sample-new --device cuda --precision fp32

# 已审核同源分片；保持原抽样方式。需要桶时新旧均用同一length-bucket-size。
& $py -m haojie_training.train --data data/train.pt --validation data/validation.pt --initialize-from $checkpoint --checkpoint updated-new.pt --device cuda --precision bf16 --threads 1 --steps 96 --batch-size 4 --length-bucket-size 64 --report update-new.json
# 精确重复研究：上条命令加 --deterministic；续训省略时自动继承。
& $py -m haojie_training.train --data data/train.pt --validation data/validation.pt --resume updated-new.pt --checkpoint resumed-new.pt --device cuda --precision bf16 --threads 1 --steps 96 --batch-size 4 --length-bucket-size 64
```

固定输入生成/起始参考用 `scripts/training/performance/execution/replay.py --freeze --data <审核分片train.pt> --checkpoint <默认模型> --fixture <新requests.pt> --output <新目录> --device cuda`。原实现冻结在 `artifacts/pytorch-20261004/baseline/haojie_training`，也可从 `a1679bb` 导出整个包；不得只替换旧 pipeline 而使用新 data/model/runtime 充当旧基线。

```powershell
& $py scripts/training/performance/execution/measure.py --checkpoint $checkpoint --fixture artifacts/pytorch-20261004/requests.pt --reference artifacts/pytorch-20261004/baseline/haojie_training --data artifacts/engine-20261003/default-model-profile/profile/data/train.pt --output measure-new --training --steps 96 --rounds 3 --cases reference optimized bf16
& $py scripts/training/performance/execution/closure.py --engine $engine --checkpoint $checkpoint --reference artifacts/pytorch-20261004/baseline/haojie_training --starts artifacts/engine-20261003/mixed-starts-24.json --output closure-new --final
& $py scripts/training/performance/execution/recovery.py --engine $engine --checkpoint $checkpoint --data artifacts/engine-20261003/final-ci-installed/native-evidence/closure/data --output recovery-new
```

原始报告、失败现场和 profiler 在 `artifacts/pytorch-20261004/`；`final-contiguous/` 为最终 A，`final/` 保留主体 A/C 三组结果，`closure-final/` 为完整 B 配置矩阵、`closure-contiguous/` 为缓冲缺陷修复后的 B 针对性复核，`recovery-deterministic/` 为恢复验收，`tests.log` 为本机回归。脚本均直接运行 Python 与指定 Rust 程序，不依赖 Node。
