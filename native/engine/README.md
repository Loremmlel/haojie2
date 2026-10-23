# Rust 原生引擎与训练运行时

同一 crate 输出 `haojie-engine` 开发验证程序和 `haojie_native` 共享库。Python 训练使用进程内共享库；旧 `--training` 子进程宿主已删除。

| 位置                                  | 职责                                           |
| ------------------------------------- | ---------------------------------------------- |
| `src/main.rs`、`modules.rs`           | 开发 CLI、模块声明和构建身份                   |
| `src/library.rs`、`online/runtime.rs` | 批量 FFI、多局原生 worker、可保存动作状态机    |
| `src/development.rs`                  | 可信开发协议、规则与公开输入差分、历史记录审核 |
| `src/core/`、`commands/`、`setup/`    | 正式规则、原子提交、状态所有权与公开观察       |
| `src/training/`                       | 公开输入边界、惰性动作树、编码与旧记录审核     |

## 构建与验证

需要 Rust 1.98.1 和平台链接器。Windows/Linux 使用同一实现。

```sh
cargo build --locked --release --manifest-path native/engine/Cargo.toml --target-dir artifacts/native-target
cargo test --locked --manifest-path native/engine/Cargo.toml --target-dir artifacts/native-target
cargo clippy --locked --all-targets --manifest-path native/engine/Cargo.toml --target-dir artifacts/native-target -- -D warnings
```

输出位于 `artifacts/native-target/release`：Windows 为 `haojie_native.dll`，Linux 为 `libhaojie_native.so`。独立安装时通过 `HAOJIE_NATIVE_LIB` 指定共享库。`haojie-engine --version` 返回规则、源码内容哈希和新训练协议身份。

`hj_create` 在 worker 内创建规则目录和全部游戏状态，不将 `Rc` 所有权强行声明为线程安全。`hj_submit` / `hj_poll` 交换连续二进制请求与结果；控制边界可保存或恢复正式状态、两类 RNG、根前缀、排除集合、回溯及事件历史。`hj_close` 释放 worker，`hj_memory_bytes` 提供活动分配字节计数。

所有正式命令继续通过共享规则内核提交。公开树只接收白名单观察；正式 RNG 仅供规则执行，采样 RNG 独立保存。模型版本由 Python 所有者绑定根，训练数据不持有行为权重槽。

TS/Rust 的状态、候选及公开编码对应继续由 `scripts/training/native/validate.ts` 和 `sampling/validate.ts` 验证。历史记录只审核，不恢复旧采样和训练入口。真实硬件测量与未验收项见[验收记录](../../training/online/VALIDATION.md)。
