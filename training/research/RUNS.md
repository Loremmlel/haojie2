# 本机实验执行记录

2026-10-23/24，Windows RTX 3060 Laptop，Torch 2.14.0+cu130。

## 已取消的学习对照

原定三种子纯 V-trace / MC critic 辅助学习实验按作者要求取消，进程已停止。具体棋力留到 Colab 长跑。此前冻结源码与未完成窗口不作为棋力结论。

`learning-measured/pure-1` 的约第 940 秒与一次 8.6 秒 CUDA 恢复测试重叠，不用于性能结论，也不补跑。CPU 专用检查使用 `CUDA_VISIBLE_DEVICES=-1`，PowerShell 空值会删除环境变量。

本次临时检查点和日志已按作者要求删除，合计 51 个、约 972 MiB。先前带强制参数的删除被拒绝，后改用任务目录内逐文件非强制删除成功。既有 `legacy-python-build` 属于原先产物，保留。

容量与 stream 测量属于各自冻结代码阶段；最终新旧吞吐另用当前实现测量，汇总到 `../online/VALIDATION.md`。
