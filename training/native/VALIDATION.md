# 原生训练验证

```powershell
pwsh -File training/native/run.ps1 check --device cpu
pwsh -File training/native/run.ps1 check --device cuda
```

数值与功能检查覆盖硬件配置选择、显式覆盖与恢复保留项、BF16 前后向、五组优化器状态、非有限更新原子拒绝、保存恢复后的下一步一致、完整候选与真实行为概率、根隔离、图回退再重放、迟到快照，以及 Rust actor 停放和 RNG 连续性。

`configure` 可单独检查当前硬件和选定参数，不启动训练。Colab 安装、配置与启动单元分开；本地验证不替代 Colab 真机验收。

Colab 控制台检查：`python -B -m unittest training.colab.test_panel training.evaluation.test_evaluation`、`node tests/browser/console/native.mjs`。覆盖手工／后台定期测评、取消与超时、真实 TS AI 到 Rust 公开接口、停止／保存协议、刷新草稿、断线和窄屏。需要构建原生库并安装 `mini-racer==0.14.1`；对手静态包用 `node scripts/training/native/teacher/build.mjs --check` 核对，修改 AI 源码后运行同一命令去掉 `--check` 再生成。原生 `check` 另用临时初始化权重验证冻结检查点加载及 BF16 测评推理，不执行学习。

需要验证真实运行控制时，另行运行 `training/native/validate.ps1`。它会实际启动训练、缩放、保存和恢复；不是只读检查。

固定工作量诊断与指标定义见[诊断入口](diagnostics/README.md)。训练使用[正常入口](README.md)，不使用诊断参考行为。
