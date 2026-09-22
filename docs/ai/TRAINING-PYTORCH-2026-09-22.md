# PyTorch训练性能：CPU、XPU与CUDA

本页前半保留Core Ultra 5 225H／Arc 130T上的旧11.55M模型记录；[CUDA补测](#cuda补测rtx-3060-laptop)使用另一台Ryzen 7 5800H／RTX 3060 Laptop，对齐训练进度中的新版11.70M模型、256实体、117候选。两种模型与输入规模的样本吞吐不能直接混比。

2026-09-22实测。首版模型为11,547,266参数的实体Transformer＋候选策略/价值头。本机Arc 130T核显可以有效训练：batch 128下，XPU BF16约466个样本/秒，是CPU FP32约96个样本/秒的4.84倍。训练先用XPU BF16，CPU使用FP32；batch 128可作为后续实验起点，仍需用真实数据验证收敛。

这是完整优化步的合成张量基准，不含游戏采样、编码、MCTS或浏览器推理，也没有验证对作者胜率。当前实现与安装命令见[Python训练子项目](../../training/README.md)，后续路线见[训练文档](TRAINING.md)。

## 环境与测量方法

- Intel Core Ultra 5 225H，14个逻辑CPU；安装32GiB内存，系统报告约31.5GiB。
- Intel Arc 130T集成GPU，共享系统内存；Windows驱动32.0.101.8243。
- Windows 11 build 26200，Python 3.12.9，PyTorch 2.14.0+xpu，NumPy 2.5.3。精确环境记录在[requirements-win-xpu.txt](../../training/requirements-win-xpu.txt)。CPU与XPU使用同一PyTorch安装。
- Windows电源方案为“平衡”，未修改功耗策略。未固定温度、频率或长期功耗，结果是本次短测，不承诺长时间训练始终维持相同速度。
- 先对CPU 4/8/14线程做预试，正式CPU测量使用8个intra-op线程，XPU主机使用4线程；所有配置均为1个inter-op线程。
- 每配置独立进程、固定相同初始权重，同batch使用相同输入。原始报告记录模型/输入SHA256、训练包源码SHA256、硬件、版本和全部分轮结果。
- 主干6层、width=384、heads=6、FFN=1536；64个实体填充槽＋1个全局token、64个候选。实体与候选约一半至全部有效，未以真实对局分布校准。
- eager模式，未使用torch.compile。预热3步后测3轮，每轮5个完整训练步，取各轮平均步耗时的中位数。
- 同步设备后计时，包含清梯度、前向、策略/价值损失、反向、范数裁剪、AdamW和AMP缩放更新。输入已驻留设备，不含样本生成、传输、参考输出比较、模型加载或文件IO。
- BF16/FP16是autocast混合精度，参数、梯度及Adam状态仍为FP32。FP16使用动态GradScaler；并非把整个模型与优化器转成FP16。

## 完整训练步速度

单位为ms/优化步，越小越快；右列是batch 128的样本吞吐，越大越快。

| 设备／精度   | batch 8 | batch 32 | batch 128 | batch 128样本/秒 |
| ------------ | ------: | -------: | --------: | ---------------: |
| CPU FP32     |  115.16 |   356.88 |   1329.12 |            96.30 |
| CPU AMP BF16 |  122.57 |   376.42 |   1374.66 |            93.11 |
| CPU AMP FP16 |  148.15 |   491.46 |   1787.35 |            71.61 |
| XPU FP32     |   51.89 |   104.07 |    379.12 |           337.62 |
| XPU AMP BF16 |   49.31 |   105.54 |    274.46 |           466.37 |
| XPU AMP FP16 |   46.15 |   101.24 |    275.79 |           464.12 |

XPU BF16相对CPU FP32，在batch 8/32/128分别快约2.34/3.38/4.84倍。相对XPU FP32，BF16在batch 128的吞吐提高约38%；batch 8与32的差距很小。FP16与BF16在大批量时基本持平，本轮不足以为小幅差异排出稳定优劣。

CPU上混合精度没有收益，FP16反而慢约29%–38%（步耗时）。CPU以FP32作为默认实验精度更合适；不应因为浏览器将来可能使用FP16权重，就要求CPU训练也用FP16。

CPU线程预试采用batch 32、FP32、预热2步、2轮×3步：[原始记录](benchmarks/torch-cpu-threads-20260922.json)。4/8/14线程分别为619.03/373.48/400.47 ms/步。8线程在已测配置中最好；正式多轮结果为356.88 ms/步。线程数更多不自动代表吞吐更高，此处没有声称找到了所有机器或所有batch的最优线程数。

## 内存与数值检查

batch 128时，PyTorch报告的XPU峰值allocated/reserved分别为：

| 精度     | allocated MiB | reserved MiB |
| -------- | ------------: | -----------: |
| FP32     |       1530.63 |         1788 |
| AMP BF16 |       1151.84 |         1298 |
| AMP FP16 |       1151.84 |         1298 |

混合精度allocated比FP32低约25%。这些是PyTorch分配器统计，未覆盖整个进程、驱动及其他应用，也不是独立显存容量。11.55M模型的FP16原始权重约23.09MB，训练内存显著更大。CPU到XPU输入拷贝另测约1.1–1.7 ms/批，未计入主表；真实数据读取与特征编码仍待测量。

正式18组配置全部通过：参数、梯度、loss均有限，矩阵输出实际dtype匹配所选精度，权重发生变化。每组预热3步、计时15步均完成真实优化器更新，无溢出跳步；没有设备或精度回退。

相同随机初始权重下，对比CPU FP32输出，XPU BF16的有效候选logit RMS相对误差约0.78%–0.88%，价值最大绝对误差不超过0.00388；XPU FP16对应约0.11%–0.12%和0.00055。所有配置在这批初始输入上的top1一致率为100%。这些只验证短测数值可用，不是最终策略精度、收敛性或棋力保证；BF16仍需与FP32在真实验证集上对照。

## 当前可运行范围

训练子项目提供模型、软策略交叉熵、带未知标签遮罩的价值MSE、AdamW、AMP、张量数据加载、原子检查点保存与续训。测试覆盖填充不污染有效输出、无效候选概率为零、未决样本不提供价值监督、参数学习、CPU续训下一步一致和数据来源变更拒绝。另已运行全尺寸XPU BF16训练10步，保存后续训至20步，更新与数值检查通过。

本轮通过Python 4项测试、Ruff检查与格式检查、CPU预编码数据文件训练、TypeScript检查、320项行为测试、目录结构检查、发行构建/成品一致性检查、34项离线file://浏览器场景与5项Pages验收。Windows全库Prettier检查使用`--end-of-line auto`兼容现有CRLF检出文件；本轮修改文件也通过默认换行配置的定向格式检查。

目前真实Observation到实体/候选特征的版本化编码器尚未实现，Node教师JSONL不能直接作为模型输入。`--synthetic`只验证训练闭环；预编码张量可通过`--data`加载。下一步应完成公开状态与动作表示，再接教师预训练及真实验证集，随后接MCTS。大规模对弈耗时不能由本表反推。

核显训练和Node采样尚未同时测量。两者并行运行时需重新评估共享内存、带宽与功耗；初期可先收集样本再训练。此报告不替代浏览器batch=1/小批推理、WASM/WebGPU算子兼容和完整回合延迟验收。

## 复现与原始记录

在仓库根目录、安装训练子项目后执行；报告拒绝覆盖已有文件，重测时更换输出名。

```powershell
training/.venv/Scripts/python.exe -X utf8 -m haojie_training.benchmark --devices cpu --precisions fp32 --batches 32 --cpu-threads 4 8 14 --warmup 2 --steps 3 --rounds 2 --output artifacts/training/cpu-threads.json
training/.venv/Scripts/python.exe -X utf8 -m haojie_training.benchmark --batches 8 32 128 --cpu-threads 8 --output artifacts/training/cpu-xpu-final-20260922.json
```

主表原始记录：[torch-cpu-xpu-20260922.json](benchmarks/torch-cpu-xpu-20260922.json)。报告中的Git提交是新增Python文件前的`c0d5cd3`，必须结合训练包源码SHA256 `a43b9f97ab99e91b2a8f03738a75a6dd7225e4dbd3ec90ba80f51d434e2a463d`识别本次代码，不能仅凭提交号复现。所有模型权重、临时数据和虚拟环境留在已忽略目录中。

实现参考：[PyTorch XPU官方说明](https://docs.pytorch.org/docs/2.14/notes/get_start_xpu.html)、[AMP官方说明](https://docs.pytorch.org/docs/2.14/amp.html)。性能结论来自上述本机实测。

## CUDA补测：RTX 3060 Laptop

2026-09-22在另一台Windows电脑完成安装与实测。结论：这台RTX 3060 Laptop适合当前约11.7M网络的本地训练。同口径短测中CUDA BF16为305.24样本/秒，是历史Arc 130T XPU BF16的3.60倍、历史Intel CPU FP32的12.76倍、本机Ryzen CPU FP32的24.00倍。延长测量后BF16与FP16均约337–340样本/秒，当前优先以CUDA BF16、batch 32开始真实数据实验。

### 安装与可比性

- 本机AMD Ryzen 7 5800H，8核16线程，约31.84GiB系统内存；NVIDIA GeForce RTX 3060 Laptop，6GiB独立显存，计算能力8.6，驱动610.88。
- Windows 11 build 26200，Python 3.12.10，PyTorch 2.14.0+cu130，CUDA运行时13.0，cuDNN 92400，NumPy 2.5.3。安装使用[PyTorch官方CUDA源](https://download.pytorch.org/whl/cu130/torch/)，依赖锁定在[requirements-win-cuda.txt](../../training/requirements-win-cuda.txt)，环境位于`training/.venv`。未更换驱动，也未另装系统CUDA Toolkit。
- 插电、电量100%，Windows“平衡”电源方案；没有修改功耗或锁定频率，没有关闭用户桌面应用。开始检查时桌面显存占用约2.9GiB。[GPU遥测](benchmarks/cuda-20260922/telemetry.csv)包含测量、CPU对照及进程间空闲，归档262个完整的一秒采样点，去除停止采集时未写完的最后一行：温度42–63°C、板卡功耗23.31–118.83W、整卡显存2613–4655MiB。不能将全段平均功耗当作训练能耗，也没有测小时级热稳态。
- 对齐[历史v2基准](benchmarks/torch-v2-20260922.json)：11,695,874参数，batch 32、256实体＋1全局token、117候选；相同种子、预热3步、3轮×5步、独立进程、同步完整优化步，eager模式。CPU正式8线程，CUDA宿主4线程，inter-op均为1。
- 全部参数、梯度、Adam状态、候选投影和策略头均为FP32；仅主干／价值头使用AMP。FP16启用GradScaler。FP32矩阵精度为highest，CUDA matmul TF32关闭。输入驻留设备，排除编码、游戏采样、传输、参考前向、检查点IO。
- 本轮模型SHA256 `da44662f3033f3dbc852d6ace29670bf34f53daab0c90cee7aff3019b04e693b`、输入SHA256 `89beeed619bc9de131429d1daf8f54d36ad5df24b77f818cf6e3db5f0666bd61`与历史v2记录逐项一致。当前基线提交`63ac066f943e184e7b533434c2b30e0a700b3cc4`，训练包源码SHA256 `0856d489b23189779e96da328bdb9218ec3ccd47ede9831e46feef17f1dbafa4`；包级源码hash与旧报告不同，模型、完整训练步和基准计时实现未变，数据校验等外围代码已有更新，当前检出为CRLF。

### 同输入完整优化步对照

| 机器／设备／精度               |   ms/步 | 样本/秒 | CUDA峰值allocated/reserved MiB |
| ------------------------------ | ------: | ------: | -----------------------------: |
| 历史Core Ultra 5 225H CPU FP32 | 1337.34 |   23.93 |                              — |
| 历史Arc 130T XPU FP32          |  457.86 |   69.89 |                              — |
| 历史Arc 130T XPU BF16          |  377.02 |   84.88 |                              — |
| 本机Ryzen 7 5800H CPU FP32     | 2515.59 |   12.72 |                              — |
| 本机RTX 3060 CUDA FP32         |  223.04 |  143.47 |                 1425.15 / 1640 |
| 本机RTX 3060 CUDA BF16         |  104.84 |  305.24 |                  945.29 / 1054 |
| 本机RTX 3060 CUDA FP16         |   95.19 |  336.18 |                  945.29 / 1054 |

原始结果：[本机CPU](benchmarks/cuda-20260922/v2-cpu.json)、[CUDA三精度](benchmarks/cuda-20260922/v2-cuda.json)。这是一组跨机器、跨后端实现的实测比较，不能把倍数归因于CUDA接口本身。CPU预试4/8/16线程分别4388.66/2454.44/2588.04ms/步，正式选择8线程；[预试](benchmarks/cuda-20260922/cpu-threads.json)只用2步预热、2轮×3步，不替代正式结果。

CUDA三组全部完成15/15次计时优化更新，预热也无跳步；参数、梯度、loss有限且权重确实变化。实际矩阵输出dtype匹配所选精度，未回退CPU。初始输出相对CPU FP32的有效logit RMS误差：BF16约0.294%、FP16约0.040%；价值最大绝对误差分别0.003732/0.000359，三组top1一致率均100%。输入拷贝另测约1.34–1.78ms/批，未计入主表。

### 较长测量、批量与较大输入

相同batch 32和256／117形状，改为预热10步、5轮×50步：[较长测量原始记录](benchmarks/cuda-20260922/v2-sustained.json)。BF16为94.24ms/步、339.56样本/秒，FP16为94.90ms/步、337.19样本/秒；两者都完成250/250次计时更新，无跳步。各轮BF16为92.51–94.53ms/步，FP16为73.67–97.29ms/步，短测中FP16领先没有保持，不能据此给两者稳定排序。这仍是每精度约24秒的计时窗口，不是长期训练稳定性证明。

其他形状沿用3步预热、3轮×5步：

| 批量×实体×候选 | 精度 |  ms/步 | 样本/秒 | 峰值allocated/reserved MiB |
| -------------- | ---- | -----: | ------: | -------------------------: |
| 8×256×117      | BF16 |  25.55 |  313.07 |               373.34 / 418 |
| 8×256×117      | FP16 |  28.12 |  284.47 |               373.34 / 418 |
| 64×256×117     | BF16 | 114.58 |  558.58 |             1712.42 / 1872 |
| 64×256×117     | FP16 | 115.51 |  554.06 |             1712.42 / 1872 |
| 32×348×212     | BF16 |  83.38 |  383.80 |             1248.75 / 1408 |
| 32×348×212     | FP16 | 130.72 |  244.79 |             1248.75 / 1406 |

原始结果：[批量扩展](benchmarks/cuda-20260922/v2-batches.json)、[较大输入](benchmarks/cuda-20260922/v2-dense.json)。348实体／212候选取自文档中的hard样本最大长度，张量仍为合成数据。所有配置均有效更新、无跳步。不同形状独立顺序测量、频率与桌面负载未锁定；较大输入BF16甚至快于先前较小形状，是本次短测波动的线索，不能解释为序列越长越快，也不能拿这些行推导精确扩展规律。batch 64已实测可行，正式训练先保留batch 32，之后用变长真实数据与验证曲线决定是否增大。

较大输入BF16的初始top1与CPU FP32一致率为31/32，FP16为32/32；BF16有效logit RMS误差约0.320%，价值最大绝对误差0.002774。数值有限不等于动作完全不变，更不等于真实验证集收敛相同。混合精度训练需持续保留FP32验证参照。

显存列为PyTorch分配器统计，不含完整CUDA上下文、驱动和桌面程序；WDDM下它与整卡驻留显存并非简单相加。未测batch 128的容量边界，也没有证明任意更长局面都能装入6GiB。不要根据模型约23.4MB的半精度权重大小估计训练显存。

### 安装验收与复现

已通过CUDA张量前向／反向、`pip check`、现有7项Python行为测试。全尺寸CUDA BF16合成训练10步、保存后另起进程续训至20步，优化器更新数也为20，参数／梯度有限；[首次训练](benchmarks/cuda-20260922/train.json)、[续训](benchmarks/cuda-20260922/resume.json)保留记录。本轮未训练真实棋谱模型，未测收敛、棋力或浏览器性能。

安装命令见[训练子项目](../../training/README.md#安装与检查)。基准文件拒绝覆盖，重复测量须改输出名：

```powershell
training/.venv/Scripts/python.exe -X utf8 -m haojie_training.benchmark --devices cpu --precisions fp32 --batches 32 --cpu-threads 8 --entities 256 --actions 117 --output artifacts/training/cuda-repeat/cpu.json
training/.venv/Scripts/python.exe -X utf8 -m haojie_training.benchmark --devices cuda --precisions fp32 bf16 fp16 --batches 32 --entities 256 --actions 117 --output artifacts/training/cuda-repeat/cuda.json
training/.venv/Scripts/python.exe -X utf8 -m haojie_training.benchmark --devices cuda --precisions bf16 fp16 --batches 32 --entities 256 --actions 117 --warmup 10 --steps 50 --rounds 5 --output artifacts/training/cuda-repeat/sustained.json
training/.venv/Scripts/python.exe -X utf8 -m haojie_training.train --synthetic --device cuda --precision bf16 --batch-size 32 --entities 256 --actions 117 --steps 10 --checkpoint artifacts/training/cuda-repeat/smoke.pt
```

实际采样、编码、数据加载、变长分桶和MCTS都不在以上样本吞吐内。CUDA训练明显快于历史XPU，并不表示端到端采样训练会同倍加速；Node教师仍主要使用CPU，本机CPU基准也不能直接替代教师吞吐测量。
