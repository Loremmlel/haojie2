# 性能诊断与验收方法

[性能入口](../PERFORMANCE.md) · [执行架构](ARCHITECTURE.md) · [测量结果](MEASUREMENTS.md)

## 运行维护入口

从仓库根运行，先设置 `PYTHONPATH=training`；每次输出目录必须新建。C++ 缓存和引擎先构建，计时期间独占 GPU，不并行编译或测试。

```sh
python -m haojie_training.diagnostics \
  --engine /path/to/haojie-engine --output artifacts/diagnostic-new \
  --sampling-backend pipeline --environments 64 --host-workers 1 \
  --inference-batch-size 64 --batch-wait-ms 0 --history-percent 0 \
  --games 128 --finite --updates 16 --deadline 1200 \
  --starts tests/fixtures/native/late-starts.json.gz --start-indices 0 2
```

`--sampling-backend local` 对照本地执行；`--initialize-from` 只读加载同一初始权重。固定工作量须核对引擎、源码、起点与初始权重哈希，以及真实终局、样本集合和消费顺序。并发执行改变轨迹时分别报告工作量。

持续窗口使用 `--seconds 120 --updates 0 --history-percent 50` 并去掉 `--finite`；省略 `--starts` 即自然新局。`--resize-at 10:8 --resize-at 30:2` 验证运行中伸缩。主指标同时读取 `terminal_samples_per_second`、`consumed_samples_per_second`、`updates_per_second` 和 `training_work`，收尾的 unknown 单列。

`--profile cpu` 分账采样与学习；`--profile cuda --profile-stage learning` 捕获学习侧时间线，默认 stage 为 sampling。剖析输出与普通计时分开；不要把剖析运行列入吞吐对照。子进程分账存于输出目录 `sampling/`，主报告同时嵌入。

## 监控与计时

`runtime.sampling_seconds` 是采样进程时钟，与学习器阶段时间重叠；跨进程 CPU 累计不可直接相加当墙钟。GPU 分配/预留显存汇总两个 CUDA 所有者。队列诊断包括 `pending_terminals`、`pending_terminal_bytes`、`terminal_transfer_seconds` 和 `terminal_backpressure_seconds`。

## 2026-10-23 收口检查

当前源码的 Windows CUDA 闭环完成四个真实终局、512 条样本、两次更新与保存，零截断，关闭后 `pending_terminals=0`；报告为 `pipeline-final-smoke-1/report.json`。Windows 定向 16 项中 14 项通过、一项因 Linux 信号语义跳过、一项等待终局队列填满的 30 秒检查超时；同一背压 / 2→4→1 伸缩 / 重复关闭检查在 Linux 通过，耗时 52.403 秒。诊断记账五项在 Linux 通过，保留 Windows 超时日志供复审。

Ruff、修改文件格式、目录结构和发行一致性检查通过。按本轮要求没有重跑全量测试。一次性采样实验文档、旧脚本命令与逐轮日记已移出当前维护文档，弃用 RPC / TorchScript 源码及专用测试已删除；误复制的源码目录、未完成的约 2 GB 压缩包和旧模块缓存已清理。
