# 原生训练验证入口

当前训练行为见[运行说明](README.md)和[架构约束](architecture/DECISION.md)。这里列维护命令，历史窗口和实验结论按[历史索引](../../docs/ai/HISTORY.md)查阅。

```powershell
pwsh -File training/native/run.ps1 check --device cpu
pwsh -File training/native/run.ps1 check --device cuda
python -B -m unittest training.colab.test_panel training.evaluation.test_evaluation
node scripts/training/native/teacher/build.mjs --check
```

原生 `check` 验证 BF16 前后向、五组补偿优化器、非有限更新原子拒绝、完整终局 V-trace、真实行为概率、长局／内存回收、异步调度及新旧完整终局 v3 保存恢复。搜索检查同时覆盖默认关闭的直接策略和显式启用 MCTS、历史对手冻结／退池引用及检查点绑定。

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
