# 独立诊断入口

`diagnose` 与正式 `run` 分离。模式为 `actor`（不学习）、`learner`（只读检查点 ready 短块回放）、`episodes`（完整终局回放）、`mixed`（正常终局训练共存）、`inference`（固定根前向）、`service`（闭环根服务）、`gemm`（矩阵微基准）。诊断参考行为与固定输入不进入正常训练。

`episodes --resume PATH --batch 64` 要求 ready 中全部为完整自然终局，使用正式终局 learner 提交一次内存中的更新，输出真实终局价值 MSE 前后值、策略熵、梯度范数和实际消费数；不覆盖源档、不保存诊断权重、不启动新采样。`batch` 控制显存小批，不能用 `--samples` 截断轨迹。`actor --chunk 1000000000` 可在短时采集中收集完整 ready 对局，未终局数据仍留在 partial；这只是取证参数，正式训练始终保留完整轨迹。

`measure.ps1` 串行运行配置并记录 `manifest.json`、原始时间记录、GPU/CPU 时钟、显存、退出码及 stderr。`--warmup` 之后的稳态计数取 `window.json - warmup.json`。learner 模式按原顺序回放同一转换前缀，热身后恢复初始参数和优化器。批量变化只在该独立诊断中允许。

`summarize.py DIRECTORY` 离线汇总，只有标准库依赖；训练宿主不调用它。每局规则、CPU 策略、根排队/在途和 lane 队列时间分别记录。根批记录根数、token 数、event 计算与回送区间；学习记录每步、实际消费、版本差和输入预取。`cuda_memory` 区分 allocated/reserved 峰值和当前值。

CUDA event 包含区间内调度等待与宿主提交空隙；逐内核并集另用 `--profile PATH`。根/s、actor 决策/s、正式命令/s、K 内实际消费转换/s 分开报告。失败或中断窗口不补造结果。

`workset` 与本目录 `benchmark.ps1 -Games N` 用同一组正式种子、实际行为概率和固定消费配额；使用 `--replay` 时逐局行为与正式停放状态须逐字节相同。保留的参考工作集位于 `artifacts/architecture-audit-20261024/scaling-workset/concurrency-1-repeat-1/workset/`。

`summarize_trace.py` 汇总 Kineto 时间线，`verify_workset.py` 核对参考与实测工作集；两者只读取既有诊断数据。已经失效的旧 Python 训练宿主对照脚本不再维护，旧架构结果通过[历史索引](../../../docs/ai/HISTORY.md)查阅。
