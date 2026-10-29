# Rust 原生规则引擎

crate 输出 `haojie-engine` 可信开发验证程序及 `haojie_native` 动态库/rlib。`learning/` 提供普通局合法参数解码树、实体/局部记录/几何编码、神经搜索状态机及固定工作线程池；`ai/` 等价移植网页三档手工 AI 的普通局路径。训练由一个 Python 进程通过批量 C ABI 驱动 Rust 线程池和 PyTorch。

| 位置                                      | 职责                                     |
| ----------------------------------------- | ---------------------------------------- |
| `src/main.rs`、`modules.rs`、`library.rs` | CLI、库模块与构建身份                    |
| `src/development.rs`                      | 可信开发协议与规则差分                   |
| `src/core/`、`commands/`、`setup/`        | 正式规则、原子提交、状态所有权与公开观察 |
| `src/learning/search.rs`、`random.rs`     | 完整命令 PUCT 状态机与可恢复 PCG64       |
| `src/learning/pool/`                      | 固定归属线程、对局推进与批量 C ABI       |
| `data/`                                   | 随源码交付的规则包与规范表示向量         |
| `../../scripts/native/engine/`            | 规则包生成、TS/Rust 差分与规则夹具       |

## 构建与验证

需要 Rust 1.98.1 和平台链接器。Windows/Linux 使用同一实现。

```sh
npm run native:rules:check
npm run native:test
npm run native:check
npm run native:build
npm run native:validate -- --output artifacts/native-rules
```

输出位于 `artifacts/native-target/release`：Windows CLI 为 `haojie-engine.exe`，Linux 为 `haojie-engine`；动态库为 `haojie_native.dll` / `libhaojie_native.so`。公开模块供仓库内部复用，不承诺稳定的外部 Rust API。

PowerShell 中若 npm 包装命令吞掉 `--output`，使用 `npm.cmd`，或直接运行 `node --import tsx scripts/native/engine/validation/run.ts --output artifacts/native-rules`。

`haojie-engine --version` 返回规则与构建身份；`--development` 接收受信任的规范状态和规则包，用于预检、命令执行、观察、驻留恢复和差分。它不是联网服务的输入鉴权边界。

所有正式命令经过共用规则内核，成功后提交新分支。公开视图不包含真实 RNG。状态中的 `Rc` 保持线程所有权，不跨线程共享可变局面。

## 训练线程池

默认 8 个 Rust 工作线程，每线程固定管理 8 局。规则、手工 AI、搜索与条件解码留在所属线程；等待神经回复时让出线程推进其他对局。Python 主线程持有 GPU 和模型，协调/I/O 线程处理阶段、回放和保存。训练和评测共用原生搜索，保留 64/16 模拟、25% 深搜、完整命令深度 16，以及原生 BF16 神经计算。

`hc_pool_*` 接口以 C 结构批量提供请求和连续数组视图，BF16 回复按原始 `uint16` 回填。请求视图有效到显式释放批次；正式命令记录由 Python 复制后确认，随后 lane 才推进下一条命令。低频分配、暂停和快照使用控制 JSON；原 `hc_call` 保留为可信开发与恢复接口。暂停时排空响应和记录，通过屏障后保存命令起点环境、RNG、模型版本及手工 AI 计划。

Python 桥固定 NumPy **2.5.3**：Rust 直接调用 `_generator` 导出的分布函数，并复用 `numpy_1.24_ufunc_call_info` capsule 提供的无 GIL f32 exp 内核，保持原搜索随机流和舍入。桥负责保活 NumPy 模块与 capsule，直至全部 Rust 线程退出；私有 ABI 版本不匹配时启动报错，不切换实现。此次功能验收使用 NumPy 2.5.3、PyTorch 2.14.0+cu130。

训练运行方式见[训练入口](../../training/README.md)，完整线程、数组和恢复契约见[运行时架构](../../docs/ai/TRAINING-RUNTIME.md)。
