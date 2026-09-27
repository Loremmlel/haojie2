# Rust 训练引擎原型

目标、覆盖、实测及后续计划统一维护在 [RUST-PROTOTYPE.md](../../docs/ai/performance/RUST-PROTOTYPE.md)。这是有显式能力边界的训练侧原型，尚不能独立跑一局游戏。浏览器仍调用 TS。

在仓库根目录运行（已验证 Rust 1.98.1、Node 22.23.2）：

```powershell
cargo build --manifest-path native/engine-prototype/Cargo.toml --target-dir artifacts/native-target --release --locked
cargo test --manifest-path native/engine-prototype/Cargo.toml --target-dir artifacts/native-target --locked
cargo clippy --manifest-path native/engine-prototype/Cargo.toml --target-dir artifacts/native-target --all-targets --locked -- -D warnings
cargo fmt --manifest-path native/engine-prototype/Cargo.toml --check
node --import tsx scripts/training/native/validate.ts --fixtures-only --output artifacts/training/rust-fixtures-new
node --import tsx scripts/training/native/validate.ts --output artifacts/training/rust-full-new
node --import tsx scripts/training/native/validate.ts --all-workers --output artifacts/training/rust-expanded-new
```

输出目录必须不存在。默认回放使用本机已有的 `economics-20260926` 四份 `worker-0.jsonl.gz` 原始轨迹，`--all-workers` 扩大到四组各8份、共32份；它们不跟踪进 Git。专项和驻留协议验证不依赖这些数据。Linux/macOS 可用 `--executable artifacts/native-target/release/haojie-engine-prototype` 指定无扩展名程序；尚未实测这些平台。构建目录显式放入已忽略的 `artifacts/`，不要把 Cargo 产物提交到源码目录。

## 维护对应关系

| Rust            | TS 参照                                                                     | 约束                                                                       |
| --------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `model.rs`      | `types.ts`、`core/traits.ts`、`core/state.ts` 的 PRNG                       | 热点字段有类型，其余 JSON 字段完整保留；图鉴及 `COMBAT_RULES` 只从 TS 接收 |
| `geometry.rs`   | `core/geometry.ts` 非部署落位、移动和攻击路径、部署行                       | 相同邻居顺序、克隆叠放、地标层、2×2、分数上限与入射方向                    |
| `movement.rs`   | `commands/game.ts`、`commands/movement.ts`、`setup/shrines.ts`              | 阶段检查、移动、操作预算、收尾、举旗刷新                                   |
| `stats.rs`      | `core/state.ts`、`core/traits.ts`、`commands/combat.ts` 的虹吸刷新          | 完整属性、独立蓄力、光环、虹吸连线是否仍有效                               |
| `combat.rs`     | `commands/combat.ts`、`core/attack-profile.ts`、`commands/game.ts` 攻击分支 | 目标准备、攻击随机分界、攻击后效果、反击、攻击操作记账                     |
| `damage.rs`     | `commands/combat.ts`、`setup/shrines.ts` 伤害和死亡钩子                     | 伤害包、免伤、名刀、反伤、人头、死亡反应队列；未移植能力明确拒绝           |
| `resolution.rs` | `core/state.ts`、`core/protection.ts`、`core/event-facts.ts`                | 事件因果范围、序号、身份/坐标快照、保护来源与衍生物创建                    |
| `reactions.rs`  | `commands/movement.ts`、`commands/combat.ts`                                | 冲撞、弹出、小屋/王城、死后射击、反射、命中牵引                            |
| `main.rs`       | 实验驱动                                                                    | JSONL 常驻进程、单局驻留与修订号，不是生产或联网入口                       |

`movement_stats` 是 `getStats` 的移动依赖投影，省去本切片不读取的攻击属性计算；因此倍率包含这项成本变化，不能称作纯语言收益。改动对应 TS 规则时须同时维护此映射并重新运行差分；没有另外一套 AI 评分或策略。

## 协议

stdin/stdout 每行一个 JSON，stdout 不混日志。第一个请求必须为 `{op:"init", protocol:"haojie-native-engine-v2", ruleset, catalog, combat}`；规则版本与 TS `RULESET_ID` 一致，`combat` 为 TS 的 `COMBAT_RULES`。只接受正式 TS 引擎重建、可 JSON 序列化的规范局面和已解析命令，不替代 `parseSession` / `parseCommand`，不支持 JS 的环、共享引用或 `undefined` 属性身份。

- `run`：`jobs: [{state, probes: Command[], command?: Command}]`。返回各预检结果及可选命令的完整新状态。
- `load`：只加载验证过的 jobs，供后续核心计时使用。
- `bench`：对加载的 jobs 重复 `repeats` 次，返回内部耗时；不含解析与通信，报告中单列。
- `reset`：`{state}` 原子替换驻留局面，返回递增的 `revision`。验证失败保留旧局面和修订号。
- `step`：`{revision, commands, trace?, clearHistory?}`。只提交成功命令，修订号逐条增加；遇到首个非法/未支持命令停止，保留成功前缀。整批解析错误或过期修订号不会执行任何命令。默认只返回每条结果；`trace:true` 额外返回清理前完整状态用于差分，`clearHistory:true` 在每条成功命令后清理事件和日志，与训练环境一致。
- `export`：返回 `{revision,state}`。它保存权威规则状态，不是 AI 的公开观察。
- 规则拒绝为 `invalid`，未移植为 `unsupported`，只读预检抵达随机边界为 `uncertain`，协议错误为 `error`。进程不会回调 TS；驱动遇协议错误失败，60秒超时会终止子进程。预期的错误可以在验证驱动中捕获，随后检查状态未变。

真实轨迹仍从种子＋命令经共享读取器校验后重建，只在内存传送临时局面。产物保存源码/可执行文件指纹、冻结运行器、计数与报告；不生成逐步 Observation 训练文件。随机伤害、免疫等使用与 TS 相同的 xorshift32、分界及消耗顺序；预检不得消费正式 RNG。权威状态只供规则执行/差分，不能直接传给未来的网络或搜索。

当前支持移动/收尾、非穿透攻击、伤害/死亡和部分反应链。召唤、部署、法术、技能命令、充能、回合切换、合成与神龛暗选尚未移植；攻击中的穿透/显式路径、击退、处决、策反、强夺、保护塔及部分入场/跨回合反应仍会返回 `unsupported`。任意深度遇到这些分支，都丢弃当前命令的副本，包括已经产生的伤害、事件、序号与 RNG 消耗。它们是覆盖缺口，不能当作非法候选裁掉。

连续段从真实轨迹中相邻的成功命令形成，不越过缺口；每段最多64条，仅从 worker-0 的长度至少2的段中均匀选8段/组。扩展回放不改变计时集。驻留计时分别报告批次执行，以及加载＋执行＋最终导出的完整通信成本；没有候选生成、选招或模型推理，不代表完整采样经济性。
