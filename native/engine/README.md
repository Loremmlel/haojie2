# haojie-engine 维护入口

`native/engine/` 是正式维护的 Rust 引擎，crate 与默认二进制均为 `haojie-engine`。现有单 crate、CLI 与运行内核继续维护，内部 Rust 模块不承诺永久稳定。浏览器使用 TS；旧教师/PUCT 不由原生训练入口调用。

## 构建与支持范围

在仓库根目录执行，Rust/Cargo 1.98.1、链接器；Python 主线要求 3.12。Windows x64 与 Linux x64 纳入 CI，macOS 与其他架构尚未验收。CUDA 由现有 Python 模型负责，Rust 不绑定 GPU；XPU 不在本轮验证范围。

```sh
cargo build --locked --release --manifest-path native/engine/Cargo.toml --target-dir artifacts/native-target
cargo test --locked --manifest-path native/engine/Cargo.toml --target-dir artifacts/native-target
cargo clippy --locked --all-targets --manifest-path native/engine/Cargo.toml --target-dir artifacts/native-target -- -D warnings
cargo fmt --manifest-path native/engine/Cargo.toml --check
```

等价开发入口为 `npm run native:build`、`native:test`、`native:check`；npm 只用于宿主开发验收。Windows 程序为 `artifacts/native-target/release/haojie-engine.exe`，Linux 无扩展名。训练安装、启动、记录、编码和恢复只依赖交付包、Python 与 Rust，不调用 Node/Bun/tsx。

`haojie-engine --version`（或 `--capabilities`）输出引擎版本、源码内容 SHA256、编译器、目标平台、构建配置、探针状态、规则包哈希、规则/编码/记录/协议版本及能力。源码身份覆盖 Cargo 文件、build.rs、全部 Rust 和规则数据；不是 Git 基点的猜测。宿主应核对身份与协议，名称变化没有改变存档或模型格式。默认 release 不带 `kernel-profile`；显式探针构建放另一 target 目录，不能用于正式速度结论。

## 能力与信任边界

| 入口                          | 正式维护能力                                                                   | 输入责任                                                                                 |
| ----------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| `--training`                  | 原生创建、连续采样、公开视图/编码、记录写入与审核；Python 完整学习阶段恢复接线 | 校验种子、规则、前缀命令、权限、记录链、模型响应和版本；接受外部记录，不接受任意权威状态 |
| `--development`（无参数兼容） | create/observe、只读查询/预检、执行、驻留状态及差分/诊断                       | 可信开发接口；仅接收 TS 重建或原生创建的规范局面，不能替代存档解析、联网鉴权或训练边界   |
| Rust 模块                     | 类型化内核、共享查询、事务、编码与出口                                         | 内部接口；改动须维护 TS 对应和失败原子性                                                 |

训练协议 `haojie-training-binary-v1`：JSON 行控制加 LE 张量；握手附完整引擎身份。记录为 `haojie-native-record-v1`，编码为 `haojie-entities-factorized-v1`，规则包为 `haojie-rules-package-v1`。具体安装、命令、张量所有权和恢复约束见 [training/NATIVE.md](../../training/NATIVE.md)。

审核器 `haojie-native-audit-v1` 的完成回执绑定同一次读取的原始字节 SHA256、规则包、完整编码 schema 和审核器源码身份。Python 的已审核对象只在当前可信进程内复用，不能从外部文件加载为缓存凭据；复用前还核对实际二进制内容与当前记录内容。来源替换之后不重新打开文件生成张量。`audit --prepare` 是推荐的单遍出口；单独 `audit` 保留，独立 TS 规则审核继续在宿主 CI 执行。

开发协议仍为 `haojie-native-engine-v4`，先 `init` 提交规则、图鉴、战斗参数、抽取池、配方与可选编码 schema。保留 `run/load/bench`、`create/observe/reset/step/export`、`training-nodes`、`sample-game`；客户端在 [scripts/training/native/client.ts](../../scripts/training/native/client.ts)。`step` 检查 revision，逐条提交成功前缀，非法命令不提交资源/RNG；规则拒绝为 invalid，预检抵达随机边界为 uncertain，unsupported 必须使完整规则验收失败。权威 export 绝不能作为模型输入。

## 日常验收与交付

[原生 CI](../../.github/workflows/native.yml) 在原生、规则、编码、训练宿主及测试变动时运行锁定构建、fmt/test/Clippy、release 身份、代表性双端规则/动作树/公开信息/编码/规范哈希差分，以及交付包独立安装、真实非空更新、另进程恢复和新权重采样。原有 TS 与离线浏览器 CI 保留。手动 workflow 的 endurance 开启多种子自然终局双端验收，不把长基准或毫秒阈值塞进每个 PR。

```sh
python scripts/training/native/pipeline/package.py --output artifacts/delivery/source.tar.gz
node --import tsx scripts/training/native/validate.ts --fixtures-only --output artifacts/native-rules
node --import tsx scripts/training/native/sampling/validate.ts --output artifacts/native-encoding
```

两个差分命令可用 `--executable` 显式指定程序。Python 原生测试必须设置 `HAOJIE_NATIVE`；缺程序立即失败。包内自带已核验的晚盘前缀和规则/哈希向量，不依赖开发机 artifacts；前缀在仓库以 gzip 保存，打包时解压为 `fixtures/late-starts.json`。包的来源及成员哈希在旁边 manifest；交付包不含 Node、TS、模型或训练产物。Linux 隔离验收见 `scripts/training/native/ci/Dockerfile`，容器只挂证据输出目录、关闭网络；扫描整个根并核对实际 execve，不允许绝对路径或回退脚本绕回宿主。

## 维护、升级与排错

- 唯一规则数值源仍是 TS catalog/combat/recipes/pools/schema。变动后运行 `scripts/training/native/rules-package.ts` 并提交规则包；`npm run train:rules:check` 拒绝过期。安装和运行不自动生成规则包。
- 按 [双端运行内核对应](../../docs/ai/performance/kernel/ARCHITECTURE.md) 同步规则、公开白名单、字段所有权和编码。新增冷字段须说明分离/失效边界；差分覆盖失败、连锁、反应、同长度替换和历史快照。
- 升级先保存原始记录、模型、数据与来源，再核对 `--version` 和检查点契约。旧模型/记录按原版本验收；不改写历史哈希，不因改名升级协议。完整学习阶段可恢复；半局递归栈不可精确恢复。
- 规则包/协议/编码不匹配时重建匹配的源码版本；不回退 TS。命令超时或模型响应错误保留成功记录前缀，关闭本次子进程，再从报告定位错误。
- 默认开发验证只用随仓库夹具；`validate.ts --all-workers` 是显式历史研究入口，依赖原始历史轨迹，缺数据不得跳过或补造。
- 本轮状态与性能证据见 [唯一任务记录](../../docs/ai/performance/engine/README.md)；旧 prototype/kernel 报告及其路径是历史证据，不批量改写。引擎转正与提速不等于模型棋力提高。
