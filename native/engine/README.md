# Rust 原生规则引擎

crate 输出 `haojie-engine` 可信开发验证程序及 `haojie_native` 动态库/rlib。`learning/` 提供普通局合法参数解码树、实体/局部记录/几何编码和 Python 进程内接口，`ai/` 等价移植网页三档手工 AI 的普通局路径。

| 位置                                      | 职责                                     |
| ----------------------------------------- | ---------------------------------------- |
| `src/main.rs`、`modules.rs`、`library.rs` | CLI、库模块与构建身份                    |
| `src/development.rs`                      | 可信开发协议与规则差分                   |
| `src/core/`、`commands/`、`setup/`        | 正式规则、原子提交、状态所有权与公开观察 |
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

训练运行方式见[训练入口](../../training/README.md)。
