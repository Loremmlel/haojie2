# PyTorch训练骨架与本机CPU/XPU性能

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
