# 原生连续轨迹重写工作区

当前训练入口已统一切换到本实现，旧模型代码、训练器、采样宿主和控制台已移除。本地正确性及吞吐见 [验收记录](VALIDATION.md)，棋力留到 Colab 长跑后评估。

## 构建与启动

在仓库根目录构建共享规则库，然后使用已有 CUDA 环境启动：

```powershell
cargo build --locked --release --lib --manifest-path native/engine/Cargo.toml --target-dir artifacts/native-target
$env:PYTHONPATH = 'training'
training/.venv/Scripts/python.exe -m haojie_training.online run --directory artifacts/root-vtrace/run --concurrency 16 --workers 2
```

Linux 使用相同源码、Cargo 命令和 Python 模块，环境变量写作 `PYTHONPATH=training`。Python 接口加载 `artifacts/native-target/release/haojie_native.dll` 或 `libhaojie_native.so`。控制面不持有 CUDA，也不启动额外 GPU 进程。

```powershell
training/.venv/Scripts/python.exe -m haojie_training.online status --directory artifacts/root-vtrace/run
training/.venv/Scripts/python.exe -m haojie_training.online resize --directory artifacts/root-vtrace/run --concurrency 128
training/.venv/Scripts/python.exe -m haojie_training.online budget --directory artifacts/root-vtrace/run --memory-mb 4096
training/.venv/Scripts/python.exe -m haojie_training.online save --directory artifacts/root-vtrace/run
training/.venv/Scripts/python.exe -m haojie_training.online stop --directory artifacts/root-vtrace/run
training/.venv/Scripts/python.exe -m haojie_training.online run --directory artifacts/root-vtrace/resumed --resume artifacts/root-vtrace/run/checkpoint.pt
```

`save` 在一致边界冻结 CPU 副本，然后后台写临时文件、同步并原子替换；`stop` 等待保存完成后退出。默认每 300 秒自动保存，`--save-seconds 0` 关闭定时保存。保存不会等待真实终局。恢复使用新的空目录。检查点含 Python 对象，只加载自己生成的可信文件。

`--seconds` 是测量窗口，不是对局步数预算。`window.json` 在收尾前冻结，`saved.json` 单列收尾后的状态。未完成局保存在检查点，不计入胜负。运行时错误写入 `failure.txt`，不会自动切换旧实现。

启动和恢复均使用新的空运行目录。当前检查点格式为 `haojie-root-vtrace-v2`；开发期 v1 实验由各自冻结源码恢复，生产入口不会静默解释不同的原生历史布局。冻结策略权重布局未变化，可通过对应实验源码导出，再使用正式策略入口评测。

内存预算按实际进程驻留加 CUDA reserved 计量。达到预算的 90% 时停止补局，已有对局和训练继续使用正常批量；实际超过预算则保存并报错。没有额外大额预留、强制单条推理或特殊公平计数器。可用 `budget --memory-mb ...` 在线调整预算。根历史在 Rust → Python 交接后只保留一份 BF16 数据；训练按根重建 GRU 历史，长历史按 256 事件分段反传。

## 契约

- 根网络为 256 维、3 层、4 头、512 维 FFN 的 dense Pre-LN Transformer，根内历史为 128 维 GRU。仅 BF16 模型计算。
- 根编码接口只接受公开实体、类别、全局数据和有效实体遮罩。前缀、候选与根内历史属于独立解码接口。
- Rust worker 独占规则对象；线程通信只传连续字节。共享库与 CLI 使用同一份规则模块。
- 每个随机选择保存完整候选支持集、实际采样概率、版本及事件历史前缀。强制选择、失败和回溯由 Rust 状态机推进。
- 每局一个学习席位，另一席整局固定对手版本。V-trace 只连接学习席位的后继；默认 64 微步一个块，真实终局奖励为 ±1/0，非终局 discount=1。
- 训练重新编码原始根张量并按根去重；用当前参数重建 GRU。共享根历史通过前缀长度防止读取未来事件。
- 推理最多两个学习策略槽；未消费轨迹不引用行为权重。对手模型独立限制为三个槽。
- CUDA 使用有界双 stream，每轮最多预提交两个推理工作批，同时只提交一个优化步。CPU 测试不创建 CUDA stream；生产学习方法只有纯 V-trace。
- 保存包括所有原生局面/RNG/搜索栈、待提交概率对应的请求、未决同席位转换、部分块、待学习块、消费确认、冻结模型与优化器。GPU 根/GRU 缓存从冻结权重重建。

算法依据：[IMPALA](https://arxiv.org/abs/1802.01561)、[AlphaStar](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)。性能与当前缺项见 [验收记录](VALIDATION.md)。

## 冻结策略与正式运行

```sh
python -m haojie_training.online.policy export artifacts/root-vtrace/run/checkpoint.pt artifacts/root-vtrace/policy.pt
python -m haojie_training.online.policy evaluate --policy artifacts/root-vtrace/policy.pt --opponent artifacts/root-vtrace/fixed-opponent.pt --directory artifacts/root-vtrace/evaluation --pairs 16 --seconds 600
python -m haojie_training.online.policy evaluate --resume artifacts/root-vtrace/evaluation/checkpoint.pt --directory artifacts/root-vtrace/evaluation-continued --seconds 600
```

冻结权重仅包含新模型配置、BF16 参数、正式规则身份与优化步编号；不接受旧模型或自动转换。`policy.load` 提供同一 `Network`，`frozen_session` 提供不进行学习的原生正式运行会话。双座使用相同正式种子，不改变对手权重；窗口结束保存未完成局，完整胜率需在全部预定对局自然结束后计算。

实验预算和选择标准见[研究计划](../research/PLAN.md)。
