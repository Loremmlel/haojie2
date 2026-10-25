# 原生训练验证

```powershell
pwsh -File training/native/run.ps1 check --device cpu
pwsh -File training/native/run.ps1 check --device cuda
```

数值与功能检查覆盖硬件配置选择、显式覆盖与恢复保留项、BF16 前后向、五组优化器状态、非有限更新原子拒绝、保存恢复后的下一步一致、完整候选与真实行为概率、根隔离、图回退再重放、迟到快照，以及 Rust actor 停放和 RNG 连续性。

`configure` 可单独检查当前硬件和选定参数，不启动训练。Colab 安装、配置与启动单元分开；本地验证不替代 Colab 真机验收。

需要验证真实运行控制时，另行运行 `training/native/validate.ps1`。它会实际启动训练、缩放、保存和恢复；不是只读检查。

固定工作量诊断与指标定义见[诊断入口](diagnostics/README.md)。训练使用[正常入口](README.md)，不使用诊断参考行为。
