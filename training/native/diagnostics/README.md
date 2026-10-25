# 独立诊断入口

`diagnose` 与正式 `run` 分离。模式为 `actor`（不学习）、`learner`（只读检查点 ready 回放）、`mixed`（正常共存）、`inference`（固定根前向）、`service`（闭环根服务）、`gemm`（矩阵微基准）。诊断参考行为与固定输入不进入正常训练。

`measure.ps1` 串行运行配置并记录 `manifest.json`、原始时间记录、GPU/CPU 时钟、显存、退出码及 stderr。`--warmup` 之后的稳态计数取 `window.json - warmup.json`。learner 模式按原顺序回放同一转换前缀，热身后恢复初始参数和优化器。批量变化只在该独立诊断中允许。

`summarize.py DIRECTORY` 离线汇总，只有标准库依赖；训练宿主不调用它。每局规则、CPU 策略、根排队/在途和 lane 队列时间分别记录。根批记录根数、token 数、event 计算与回送区间；学习记录每步、实际消费、版本差和输入预取。`cuda_memory` 区分 allocated/reserved 峰值和当前值。

CUDA event 包含区间内调度等待与宿主提交空隙；逐内核并集另用 `--profile PATH`。根/s、actor 决策/s、正式命令/s、K 内实际消费转换/s 分开报告。失败或中断窗口不补造结果。

`workset` 与 `benchmark.ps1` 用同一组正式种子、实际行为概率和固定消费配额；使用 `--replay` 时逐局行为与正式停放状态须逐字节相同。保留的参考工作集位于 `artifacts/architecture-audit-20261024/scaling-workset/concurrency-1-repeat-1/workset/`。
