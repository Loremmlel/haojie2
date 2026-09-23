# PyTorch训练子项目

Python 3.12独立环境，与TypeScript规则代码同仓库维护。当前实现约11.70M参数的实体Transformer、候选评分/价值头、共享TypeScript公开状态编码和动作分解、教师数据准备、整局验证划分、张量训练与续训，以及CPU/XPU精度基准。网络CLI已能完成完整对局，浏览器已验收本地HTTP前向；MCTS和单HTML模型部署尚未接入。训练loss与教师拟合率没有对作者胜率含义。

当前阶段与实际实验记录维护在[训练进度](../docs/ai/TRAINING-PROGRESS.md)。

## 安装与检查

在仓库根目录执行。Windows/XPU精确依赖记录在requirements-win-xpu.txt；PyTorch使用官方XPU源，仍支持同一进程选择CPU。无需安装torchvision或torchaudio。

```powershell
python -m venv training/.venv
training/.venv/Scripts/python.exe -m pip install -r training/requirements-win-xpu.txt
training/.venv/Scripts/python.exe -m pip install --no-deps -e training
training/.venv/Scripts/python.exe -c "import torch; print(torch.__version__); print(torch.xpu.is_available())"
```

其他平台先安装适合设备的PyTorch 2.14.0，再`python -m pip install -e 'training[dev]'`。`.venv`、缓存和egg-info不提交，也不进入Prettier/目录结构检查；数据与模型放在已忽略的artifacts/training。代码、配置与依赖版本提交Git，不将大模型权重放进源码仓库。

Windows/NVIDIA CUDA环境使用单独的依赖清单，torch来自官方CUDA源；选择与本机显卡匹配的一份清单安装，不在同一环境叠加XPU与CUDA版torch：

```powershell
python -m venv training/.venv
training/.venv/Scripts/python.exe -m pip install -r training/requirements-win-cuda.txt
training/.venv/Scripts/python.exe -m pip install --no-deps -e training
training/.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

无需激活环境，直接使用上述Python路径即可。训练指定`--device cuda --precision bf16`；基准指定`--devices cuda`，因为基准默认设备仍为CPU/XPU。RTX 3060 Laptop与历史CPU/XPU的对照见[CUDA实测](../docs/ai/TRAINING-PYTORCH-2026-09-22.md#cuda补测rtx-3060-laptop)。

## 网络骨架

- 6层Pre-LN Encoder，width=384，heads=6，FFN=1536，GELU，dropout=0。
- 64维实体特征投影＋256项类别嵌入；32维全局特征形成始终有效的全局token。
- 原生scaled_dot_product_attention，填充掩码为True的实体可以被关注；没有基于实体列表位置的序号嵌入，空间信息需由特征编码提供。
- 64维候选先经Linear/GELU/LayerNorm独立投影，再拼接全局、来源实体、目标实体，经MLP输出每候选logit。候选投影与评分头保持FP32，实体主干/价值头使用所选AMP精度；避免相邻落点的细小分差被BF16量化吞掉。候选数和实体数均可变化，模型不截断。
- 价值头输出[-1,1]终局收益，采用该样本观察所属玩家的视角。
- 当前网络`entity-transformer-v2`共**11,695,874**参数；全部按FP16存储约23.39MB，仅是权重体积估计。实际混合FP16 ONNX为25.4MB，内置浏览器本地HTTP前向已验收，单HTML离线打包待做。训练仍保留FP32参数、梯度和Adam状态，不能用发布权重大小估计训练内存。

原11,547,266参数骨架在64个真实分步样本上的落点拟合较慢，且高logit下BF16分差不稳定，因此增加独立候选投影并使用FP32评分。此变化约增加1.3%参数；旧骨架检查点须用原代码读取，当前程序会明确拒绝混用，不做静默部分加载。旧CPU/XPU报告仍保留原网络测量，不能当作v2的新速度结论。

候选由共享动作树提供，路径逐格选择、材料逐个选择，完整命令经过公开预检。中间参数可能没有合法补全，后续网络解码器须回溯；candidate_mask不能解释为全命令合法性保证。编码语义、信息边界与覆盖限制见[ENCODING.md](ENCODING.md)。

## 真实教师数据与小样本拟合

以下命令从仓库根目录执行，文件和输出目录均拒绝意外覆盖：

```powershell
npm run train:selfplay -- --games 4 --seed 2026092201 --rules classic --difficulty easy --nodes 40 --plies 80 --commands 1200 --output artifacts/training/pilot-20260923/teacher.jsonl.gz
training/.venv/Scripts/python.exe -X utf8 -m haojie_training.prepare artifacts/training/pilot-20260923/teacher.jsonl.gz --output artifacts/training/pilot-20260923/encoded
training/.venv/Scripts/python.exe -X utf8 -m haojie_training.train --data artifacts/training/pilot-20260923/encoded/train.pt --validation artifacts/training/pilot-20260923/encoded/validation.pt --overfit-examples 64 --device xpu --precision bf16 --batch-size 32 --steps 300 --checkpoint artifacts/training/pilot-20260923/overfit-v2.pt --report artifacts/training/pilot-20260923/overfit-v2-300.json
```

教师轨迹使用`haojie-training-record-v1`：每局只保存种子/模式（导入存档则另带初始局面）、实际截断上限，逐步保存命令、操作者、前后指纹和教师统计，结尾保存真实结果。没有逐步Observation或异常局面快照。`.jsonl.gz`使用Node内置gzip流式压缩；`.jsonl`仍可用于人工检查。旧快照训练JSONL不兼容，请重新生成；游戏存档兼容逻辑不变。

prepare调用本仓库Node/tsx编码器，统一重放入口验证版本、顺序、权限、指纹和结果，再生成操作者的公开观察。Python通过管道接收编码输入，只做张量化、填充和标签连接，直接生成二进制`train.pt`和`validation.pt`，无需落盘大数组JSON。重放只发生在预处理阶段，每轮训练读取`.pt`。可传多个压缩/普通轨迹，拒绝重复game_id；同一规则/种子的换边对局共用group种子族并放在同一个集合。默认分出25%验证组，训练/验证至少各一个种子族；manifest记录源码/输入指纹、来源、结果和动作覆盖。缺少outcome的完整命令前缀或明确取消的对局仍保留策略数据，价值遮罩为False；损坏JSON/gzip明确报错，不静默截掉坏尾部。

`--overfit-examples 64`从训练集固定抽取64个分步样本，只用于确认网络能记住真实输入。移除此参数才使用整个训练集；验证集始终不参与优化。报告分别列出训练前后策略loss、全部/多候选/根决策拟合率和已知终局的价值MSE；单候选的强制步骤不能抬高多候选指标。价值仅训练每个实际命令的根决策，按actor视角，不按active猜测。

首批实际数据的实体序列可超过64，含死亡记录和附属状态，不能直接套用合成性能基准的batch 128。当前张量文件整体驻留CPU，每个训练批次再移除全无效的尾部填充；扩到5万–20万决策前先实现分片和长度分桶。

## 教师对照与数据审计

指定`--opponent`后，相邻两局使用相同种子并交换主教师席位。下例是20个种子、共40局；两个预算均为固定work，墙钟只记录成本。省略`--opponent`保留原有同难度、逐局递增种子的采样方式；完全相同的两个教师配置会被拒绝，避免产生同种子的重复轨迹。

```powershell
npm run train:selfplay -- --games 40 --seed 2026092230 --difficulty hard --nodes 800 --opponent medium --opponent-nodes 320 --plies 100 --commands 3000 --output artifacts/training/teachers/paired.jsonl.gz --report artifacts/training/teachers/paired.json
npm run train:inspect -- artifacts/training/teachers/paired.jsonl.gz --encode --output artifacts/training/teachers/inspection.json
npm run train:audit-teacher -- --input artifacts/training/pilot-20260923/teacher.jsonl.gz --positions 32 --output artifacts/training/teachers/budget-audit.json
```

`--report`保存两方配置、运行源码指纹、基础提交、硬件和每局结果。采样逐命令记录来源与耗时；取消或故障写出明确中断原因，已执行的命令仍可重放。同步教师搜索只能在调用之间响应取消，返回后会再次检查取消状态，不提交迟到命令。

`train:inspect`接收多个压缩/普通轨迹，通过统一入口重建公开Observation、核对真实引擎结算和前后指纹；实际截断上限来自轨迹头，同名JSON报告存在时也核对其上限。按教师统计命令、来源棋种、技能、反应、路径、缓存、回应和终局收益；全部命令与非缓存搜索的耗时分别报告。只对双方席位均已真实终局的种子给成对分数，覆盖率不代表学会能力。加`--encode`检查共享动作树与编码器，统计分步/多候选数量及最大输入长度。缺少结束行时，检查器与编码器统一标记`missing-outcome`中断，不给胜负标签。

预算审计从每局/实际回合/操作者的首个非缓存、多候选play位置中，按公开指纹哈希选样；要求至少4个场上棋子。对每个选中位置重新比较medium/320、hard/800、hard/1600，并重复决策核对确定性；不继承原局的缓存或预算历史。`changedHardBudgetCommands`只是换招数量，不能当作1600更强的证据。

混合教师文件默认编码双方实际动作。向prepare传`--teacher-difficulty hard`可只保留hard方完整命令的全部分步标签；manifest逐局记录保留/排除的命令数，种子族划分仍保持完整。教师来源缺失或筛选后出现空局会报错。筛选不会把medium的动作改成hard答案；在弱方实际到达局面上重新查询强教师仍待实现。终局价值反映原混合对手续局，不能解释为对统一强对手的胜率。大批采样及训练仍按[训练进度](../docs/ai/TRAINING-PROGRESS.md)的分阶段门槛推进。

## 训练与续训

```powershell
# 只验证闭环；合成标记会保存在检查点内。
training/.venv/Scripts/python.exe -X utf8 -m haojie_training.train --synthetic --device xpu --precision bf16 --steps 20 --checkpoint artifacts/training/smoke.pt

# 从同一数据来源、精度及模型结构继续20步，显式允许更新该检查点。
training/.venv/Scripts/python.exe -X utf8 -m haojie_training.train --synthetic --device xpu --precision bf16 --steps 20 --resume artifacts/training/smoke.pt --checkpoint artifacts/training/smoke.pt

# CPU也可运行；tiny仅供快速验证，不代表正式模型规模。
training/.venv/Scripts/python.exe -X utf8 -m haojie_training.train --synthetic --tiny --device cpu --steps 10 --checkpoint artifacts/training/tiny.pt
```

`--steps`表示本次新增步数。训练共用AdamW（lr=3e-4、weight_decay=0.01、foreach=True）、梯度范数裁剪1.0与策略交叉熵＋带遮罩的价值MSE。FP32为基线，BF16/FP16使用autocast；FP16额外使用动态GradScaler，初始scale=1024，记录真正执行的optimizer更新数，不将溢出跳步算作有效更新。设备不可用时，显式`--device xpu`会报错；只有`auto`允许选择其他可用设备。

检查点保存模型配置、权重、优化器、缩放器、步数、训练随机状态和数据来源；拒绝无意覆盖、数据/精度/结构混用。完成训练且数值检查通过后，用临时文件替换检查点。Ctrl+C保留已有检查点，不保存可能中断在优化器内部的状态。本轮未实现自动定时检查点。

## 输入张量约定

`--data encoded.pt`接受weights_only可加载的字典：`{format: 'haojie-training-tensors-v1', metadata, tensors}`。metadata必须有ruleset、encoding及布尔synthetic字段。训练会额外保存源文件SHA256。当前一次将文件加载到CPU内存，需要扩大数据量时再分片。

| 张量              | 形状／类型       | 含义                                               |
| ----------------- | ---------------- | -------------------------------------------------- |
| entities          | [B,N,64] float32 | 公开实体数值特征                                   |
| kinds             | [B,N] int64      | 实体类别编码，范围[0,256)                          |
| entity_mask       | [B,N] bool       | True为有效实体；空棋盘保留一个无效填充槽           |
| globals           | [B,32] float32   | 观察方、阶段和公开全局特征                         |
| candidates        | [B,A,64] float32 | 候选动作特征                                       |
| sources / targets | [B,A] int64      | 来源/目标实体索引，-1指全局token                   |
| candidate_mask    | [B,A] bool       | 可训练候选，每样本至少一个                         |
| policy            | [B,A] float32    | 归一化教师one-hot或未来MCTS软目标；无效候选概率为0 |
| value             | [B] float32      | 观察所属方的真实终局收益；未知时填有限占位值       |
| value_mask        | [B] bool         | 仅真实终局标签为True；截断/未决为False             |

只接受上述张量键，不接收seed/rng/Session/原始观察对象。此检查不能替代编码器对来源的审计；共享游戏编码器只读取公开Observation，宿主元数据不进入特征。来源/目标索引必须指向有效实体，所有输入和标签须有限。原始教师JSONL先运行prepare，不能直接交给train的--data。

## 设备与精度基准

```powershell
training/.venv/Scripts/python.exe -X utf8 -m haojie_training.benchmark --batches 8 32 128 --cpu-threads 8 --output artifacts/training/benchmark.json
training/.venv/Scripts/python.exe -X utf8 -m haojie_training.benchmark --devices cpu --precisions fp32 --batches 32 --cpu-threads 4 8 14 --output artifacts/training/cpu-threads.json
```

每组配置启动独立进程，使用相同CPU初始化权重及合成输入；报告保存两者SHA256、源码SHA256、硬件/驱动、线程数、形状、精度与PyTorch版本。默认64个实体填充槽（另有1个全局token）、64个候选，约一半至全部槽有效。这里只模拟计算形状，不对应实际对局分布。

默认预热3步，再测3轮、每轮5步，取每步耗时中位数。前后同步设备，计时包含清梯度、前向、损失、反向、裁剪、AdamW和AMP缩放更新。输入已驻留设备，数据生成/编码、Node对弈、文件IO、初次加载和CPU参考比较不计时；CPU→设备拷贝另测。每组最多180秒，失败或超时明确记录，绝不悄悄切换设备或精度。

报告检查实际矩阵输出dtype、权重变化、参数/梯度/loss有限性、跳过更新数，并在训练前比较相同FP32权重的输出差异。随机模型短测的误差和top1一致率不代表训练收敛或游戏棋力。使用eager模式，没有torch.compile；后续编译优化需分别报告编译成本与稳态速度。

本机Core Ultra 5 225H / Arc 130T的原11.55M骨架测量见[训练性能报告](../docs/ai/TRAINING-PYTORCH-2026-09-22.md)；v2在batch 32、256实体、117候选下的新版对照见[训练进度](../docs/ai/TRAINING-PROGRESS.md)：CPU FP32约23.93样本/秒，XPU BF16约84.88样本/秒。两次形状/网络不同，不能直接比较绝对吞吐。命令中的8线程来自本机CPU预试，换机器应重新选择。

## 浏览器推理验收

安装可选导出依赖后，用真实数据导出FP32与混合FP16两个动态ONNX；输出目录必须是新目录。导出器加载同规则/编码的检查点，保留FP32评分头，对四个真实输入和batch 4做原生CPU数值对照，并生成无教师标签的八输入测试资产。合成张量仅用于图追踪，不进入浏览器速度成绩。

```powershell
training/.venv/Scripts/python.exe -m pip install -e "training[export]"
training/.venv/Scripts/python.exe -X utf8 -m haojie_training.export --checkpoint artifacts/training/pilot-20260922/overfit-v2.pt --data artifacts/training/pilot-20260922/encoded/train.pt --output artifacts/training/pilot-20260922/browser
npm run bench:training:browser -- --data artifacts/training/pilot-20260922/browser
```

在浏览器打开终端给出的127.0.0.1地址，按顺序运行CPU FP32、WebGPU FP32、WebGPU混合FP16。页面显示设备、首次/中位/P95耗时、输出误差和可复制的完整JSON；“检查GPU算子执行”另开非计时会话，通过详细节点分配日志确认GPU/CPU分工。没有GPU或shader-f16时报错，不自动改测其他后端。

所有资产由本机服务提供，服务器只允许GET预列路径；不发布目录、不访问CDN。ONNX Runtime Web 1.30.0的`ort.webgpu.bundle.min.mjs`必须配套`ort-wasm-simd-threaded.asyncify.wasm`，不可混用旧JSEP文件。CPU当前用同一运行库的WASM单线程；混合FP16是主干/价值头FP16＋候选评分FP32，外部浮点输入/输出仍为FP32。

实测见[训练进度](../docs/ai/TRAINING-PROGRESS.md#浏览器实测117m模型)。这只是独立开发工具，不进入发行HTML；本次Codex内置浏览器阻止file://，所以记录为本地HTTP。浏览器完整命令解码、MCTS、单HTML内嵌运行库/权重、离线启动和峰值内存另行验收。

## 网络CLI对战

```powershell
npm run train:match -- --checkpoint artifacts/training/pilot-20260922/overfit-v2.pt --output artifacts/training/neural-cpu256-20260922 --games 20 --seed 2026092205 --plies 100 --commands 1200 --device cpu --precision fp32 --threads 4
```

命令启动一个常驻Python推理进程，Node持有权威环境、分步动作树和原手工教师。模型只接收八项公开输入张量，握手核对完整编码schema和规则版本；不向Python发送原始Observation、正式seed/rng、教师选择或标签。`--device xpu --precision fp32`可显式选择核显，另支持bf16/fp16；精度变化可能改变选招，不能直接用不同轨迹的整局时间证明加速。

相邻两局用同一种子交换模型席位，`--games`为总局数，偶数可组成完整配对。默认对手easy、每决策40节点，可用`--difficulty`、`--nodes`调整。当前检查点只用于流程验收，不能当作正式棋力模型。`--output`必须是不存在的新目录，生成：

- `games.jsonl.gz`：压缩增量轨迹，包含每局种子/席位/实际上限、逐命令前后指纹、动作/分解路径、耗时及终局/截断/中断；异常只记索引、指纹与原因，局面沿此前命令重放恢复。
- `report.json`：模型/规则/编码及实验源码指纹、设备参数、完整决策P50/P95、需要推理的命令单独统计、推理次数、首个真实推理决策、终局/截断/异常和胜负计数。

解码按网络logit稳定排序，逐参数深度优先选取，第一个可合法完成的分支即返回；空分支回溯，不做全路径联合概率搜索。单候选节点直接推进，无需推理；价值输出只记录根决策，不把带动作前缀的价值当根局面价值。这是纯网络策略对战，尚未接MCTS。

`--decode-nodes`默认256、`--evaluations`默认32，是每条完整命令的固定工作量上限。落点后选目标的分支可能枚举117格；首轮实测64节点不足，相关记录保留在进度文档。耗尽即暂停并记录最后前缀，不强行end、不改用教师；`--turn-commands`默认200，同一实际全局回合的同操作者/公开指纹出现第四次也会暂停检查。这是防失控保护，不是规则判和依据。`--timeout-ms`默认60000，只处理子进程故障，不据墙钟改选动作。Ctrl+C会取消在途推理并在落子前再次校验，不提交迟到结果。

统计中的完整命令耗时包含观察、动作树/合法性查询、编码、JSON管道、网络计算和引擎提交，排除模型启动、提交后trace构造/落盘及对方思考。Python `model_ms`含输入传输和输出回读并同步设备；端到端数值仍以Node计时为准。此处是原生PyTorch CPU/XPU性能，不能当作浏览器WASM/WebGPU延迟。动作前缀进入主干，本轮没有直接缓存主干结果；回合外可选巨大化抢占调度和浏览器Worker接入另做。

异常或取消返回非零退出码；正常触及局数/命令/回合上限的截断不判胜负。保持`report.json`及逐命令trace配套保存，不把本地开发对局胜率当对作者胜率。

## 验证

```powershell
training/.venv/Scripts/python.exe -m unittest discover -s training/tests -v
training/.venv/Scripts/python.exe -m ruff check training/haojie_training training/tests
training/.venv/Scripts/python.exe -m ruff format --check training/haojie_training training/tests
```

参考：[PyTorch XPU](https://docs.pytorch.org/docs/2.14/notes/get_start_xpu.html)、[AMP](https://docs.pytorch.org/docs/2.14/amp.html)、[SDPA遮罩语义](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)。设备速度结论必须以本仓库实际模型基准为准。
