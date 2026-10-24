# Rust 原生引擎与训练运行时

同一 crate 输出 `haojie-engine` 开发验证程序和 `haojie_native` 共享库。独立 C++ / LibTorch 宿主使用进程内共享库；旧 Python 和 `--training` 子进程宿主已删除。

| 位置                               | 职责                                           |
| ---------------------------------- | ---------------------------------------------- |
| `src/main.rs`、`modules.rs`        | 开发 CLI、模块声明和构建身份                   |
| `src/library.rs`、`online/`        | 逐局类型化 FFI、可停放 actor 与动作状态机      |
| `src/development.rs`               | 可信开发协议、规则与公开输入差分、历史记录审核 |
| `src/core/`、`commands/`、`setup/` | 正式规则、原子提交、状态所有权与公开观察       |
| `src/training/`                    | 公开输入边界、惰性动作树、编码与旧记录审核     |

## 构建与验证

需要 Rust 1.98.1 和平台链接器。Windows/Linux 使用同一实现。

```sh
cargo build --locked --release --manifest-path native/engine/Cargo.toml --target-dir artifacts/native-target
cargo test --locked --manifest-path native/engine/Cargo.toml --target-dir artifacts/native-target
cargo clippy --locked --all-targets --manifest-path native/engine/Cargo.toml --target-dir artifacts/native-target -- -D warnings
```

输出位于 `artifacts/native-target/release`：Windows 为 `haojie_native.dll`，Linux 为 `libhaojie_native.so`。C++ 宿主在构建时链接该库，运行时由系统动态库路径定位。`haojie-engine --version` 返回规则、源码内容哈希和训练协议身份。

`hj_actor_create` 在调用它的 C++ lane 上创建目录与持久 actor 句柄，不将 `Rc` 所有权声明为线程安全。`hj_actor_call` 每次只操作指定一局：创建、推进、快照、停放、恢复、删除。停放返回自包含快照并释放整局及树；恢复保留正式状态、两类 RNG、路径和每层排除 frontier。`hj_actor_close` 在拥有线程释放句柄；`hj_memory_bytes` 提供活动分配计数。返回字节视图只在下次调用前有效，panic 和错误不会越过 FFI。

所有正式命令通过共享规则内核提交。公开树只接收白名单观察；正式 RNG 仅供规则执行，采样 RNG 独立保存。模型发布由 C++ 宿主统一管理，不绑定长根或训练数据。

TS/Rust 的状态、候选及规范公开编码对应由 `scripts/training/native/validate.ts` 和 `sampling/validate.ts` 验证。类型化训练行在编码源头保留整数、布尔与存在位；连续值才转为 BF16。历史记录只审核；本机测量见[验收记录](../../training/native/VALIDATION.md)。
