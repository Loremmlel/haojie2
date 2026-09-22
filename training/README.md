# PyTorch训练子项目

Python 3.12独立环境，与TypeScript规则代码同仓库维护。当前实现约11.55M参数的实体Transformer、候选评分/价值头、预编码张量训练、检查点续训和CPU/XPU精度基准。它验证模型结构和训练基础设施；尚未实现完整Observation编码、合法候选生成、MCTS或浏览器模型部署。合成数据训练没有游戏棋力含义。

## 安装与检查

在仓库根目录执行。Windows/XPU精确依赖记录在requirements-win-xpu.txt；PyTorch使用官方XPU源，仍支持同一进程选择CPU。无需安装torchvision或torchaudio。

```powershell
python -m venv training/.venv
training/.venv/Scripts/python.exe -m pip install -r training/requirements-win-xpu.txt
training/.venv/Scripts/python.exe -m pip install --no-deps -e training
training/.venv/Scripts/python.exe -c "import torch; print(torch.__version__); print(torch.xpu.is_available())"
```

其他平台先安装适合设备的PyTorch 2.14.0，再`python -m pip install -e 'training[dev]'`。`.venv`、缓存和egg-info不提交，也不进入Prettier/目录结构检查；数据与模型放在已忽略的artifacts/training。代码、配置与依赖版本提交Git，不将大模型权重放进源码仓库。

## 网络骨架

- 6层Pre-LN Encoder，width=384，heads=6，FFN=1536，GELU，dropout=0。
- 64维实体特征投影＋256项类别嵌入；32维全局特征形成始终有效的全局token。
- 原生scaled_dot_product_attention，填充掩码为True的实体可以被关注；没有基于实体列表位置的序号嵌入，空间信息需由特征编码提供。
- 候选评分拼接全局、来源实体、目标实体及64维候选特征，经MLP输出每候选logit。候选数和实体数均可变化，模型不截断。
- 价值头输出[-1,1]终局收益，采用该样本观察所属玩家的视角。
- 默认总参数**11,547,266**；纯FP16权重约23.09MB。训练仍保留FP32参数、梯度和Adam状态，不能用发布权重大小估计训练内存。

候选评分头需要上游提供完整候选及可用掩码，不负责生成合法命令。本轮没有将手工AI裁剪候选当成全部合法动作；将来的分层动作生成、路径与多材料表示仍需结合共享引擎设计。64维特征和类别词表容量是骨架输入约定，具体通道与catalog稳定映射尚待游戏编码器定义，不能声称已经覆盖所有状态。

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

只接受上述张量键，不接收seed/rng/Session/原始观察对象。此检查不能替代编码器对来源的审计；未来游戏编码器必须只读取公开Observation，不从元数据或历史预测正式PRNG。来源/目标索引必须指向有效实体，所有输入和标签须有限。当前Node教师JSONL尚不能直接传入；先完成版本化游戏编码器，再把真实样本接入此张量边界。

## 设备与精度基准

```powershell
training/.venv/Scripts/python.exe -X utf8 -m haojie_training.benchmark --batches 8 32 128 --cpu-threads 8 --output artifacts/training/benchmark.json
training/.venv/Scripts/python.exe -X utf8 -m haojie_training.benchmark --devices cpu --precisions fp32 --batches 32 --cpu-threads 4 8 14 --output artifacts/training/cpu-threads.json
```

每组配置启动独立进程，使用相同CPU初始化权重及合成输入；报告保存两者SHA256、源码SHA256、硬件/驱动、线程数、形状、精度与PyTorch版本。默认64个实体填充槽（另有1个全局token）、64个候选，约一半至全部槽有效。这里只模拟计算形状，不对应实际对局分布。

默认预热3步，再测3轮、每轮5步，取每步耗时中位数。前后同步设备，计时包含清梯度、前向、损失、反向、裁剪、AdamW和AMP缩放更新。输入已驻留设备，数据生成/编码、Node对弈、文件IO、初次加载和CPU参考比较不计时；CPU→设备拷贝另测。每组最多180秒，失败或超时明确记录，绝不悄悄切换设备或精度。

报告检查实际矩阵输出dtype、权重变化、参数/梯度/loss有限性、跳过更新数，并在训练前比较相同FP32权重的输出差异。随机模型短测的误差和top1一致率不代表训练收敛或游戏棋力。使用eager模式，没有torch.compile；后续编译优化需分别报告编译成本与稳态速度。

本机Core Ultra 5 225H / Arc 130T的测量结果、精度建议与完整原始记录见[训练性能报告](../docs/ai/TRAINING-PYTORCH-2026-09-22.md)。命令中的8线程来自本机CPU预试，换机器应重新选择。

## 验证

```powershell
training/.venv/Scripts/python.exe -m unittest discover -s training/tests -v
training/.venv/Scripts/python.exe -m ruff check training/haojie_training training/tests
training/.venv/Scripts/python.exe -m ruff format --check training/haojie_training training/tests
```

参考：[PyTorch XPU](https://docs.pytorch.org/docs/2.14/notes/get_start_xpu.html)、[AMP](https://docs.pytorch.org/docs/2.14/amp.html)、[SDPA遮罩语义](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)。设备速度结论必须以本仓库实际模型基准为准。
