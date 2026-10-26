# 原生训练验证入口

当前训练行为见[运行说明](README.md)和[架构约束](architecture/DECISION.md)。这里列维护命令，历史窗口和实验结论按[历史索引](../../docs/ai/HISTORY.md)查阅。

```powershell
pwsh -File training/native/run.ps1 check --device cpu
pwsh -File training/native/run.ps1 check --device cuda
python -B -m unittest training.colab.test_panel training.evaluation.test_evaluation
node scripts/training/native/teacher/build.mjs --check
```

原生 `check` 验证 BF16 前后向、五组补偿优化器、非有限更新原子拒绝、完整终局 V-trace、真实行为概率、长局／内存回收、异步调度及新旧完整终局 v3 保存恢复。搜索检查同时覆盖默认关闭的直接策略和显式启用 MCTS、历史对手冻结／退池引用及检查点绑定。

`check-search --device cuda` 另运行三轮固定真实公开输入的并发查询：384 宽模型、8 个调用线程、当前及历史两个版本，每轮 1024 次查询、352 个冷根编码；模型初始化和图捕获在计时外。输出查询吞吐，校验完整合法支持集和冻结版本绑定；同根并发切换版本时逐位核对策略质量及价值。

2026-10-26 在 RTX 3060 Laptop 上，保留修复前程序并用同一工作集先后运行三轮，查询吞吐中位数从 2186 提升到 8455 次／秒（3.87 倍）。修复包含 lane 并行策略头、缓存命中绕过队列、双槽编码和关闭搜索时跳过价值头；固定查询检查仍计算价值头。

正常采集另从同一个完整终局 v3 检查点交替运行旧／新程序各两次，每次 10 秒，均为 24 并发、8 worker、推理批上限 8、档内学习 batch 64、搜索关闭：

| 指标                   | 修复前两次    | 修复后两次    |
| ---------------------- | ------------- | ------------- |
| 微步／秒               | 1730 / 1706   | 4605 / 4674   |
| 产生转换／秒           | 888 / 872     | 2394 / 2439   |
| CPU 策略累计秒／墙钟秒 | 0.447 / 0.451 | 1.583 / 1.588 |

微步采集平均提升 2.70 倍，产生转换平均提升 2.75 倍；这四个短窗口均未终局、未学习、无失败。新程序保存后再次恢复继续推进原对局；完整终局更新、优化器数值续接、内存回收及 MCTS 则由 CPU／CUDA `check` 验证。

Python 控制与测评检查使用已构建的原生程序和 `mini-racer`，覆盖保存回执、冻结测评、训练模式、Colab 搜索配置及取消／超时；依赖由 `training/colab/setup.sh` 安装。只有安装／面板和现有 TS 对手执行依赖 Python，训练宿主独立运行。

原生规则、公开候选与编码另验：

```sh
npm run native:test
npm run native:check
npm run train:rules:check
node --import tsx scripts/training/native/validation/run.ts --output artifacts/native-rules
node --import tsx scripts/training/native/sampling/validate.ts --output artifacts/native-encoding
```

两条差分命令要求输出目录不存在，默认使用 `artifacts/native-target/release/haojie-engine`（Windows 自动加 `.exe`）。夹具保留完整候选与随机边界；不缩减输入或伪造终局以通过检查。

`training/native/validate.ps1` 会实际启动训练，检查缩放、保存、停止及恢复；需要时单独执行。训练吞吐读取正常运行的 `window.json` 与 `wall.json`，区分正式命令、模型查询和已学习转换。

网页代码按根 [README](../../README.md) 执行格式、TypeScript、行为测试、构建／发行一致性、目录结构和离线浏览器验收。训练控制台浏览器检查为 `node tests/browser/console/native.mjs`。本地通过与 Colab 真机验证分别记录。
