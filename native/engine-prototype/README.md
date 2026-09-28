# Rust 训练引擎

当前范围和验收入口见[状态入口](../../docs/ai/STATUS.md)；阶段目标、原始验证证据和当时计划保存在 [RUST-PROTOTYPE.md](../../docs/ai/performance/RUST-PROTOTYPE.md)。当前实现完整规则、公开动作树、实体编码及经济性实验的小网络连续采样。浏览器继续使用 TS。原生策略尚未训练，不把采样吞吐当作棋力或完整训练倍率。

在仓库根目录运行（本机 Windows、Rust 1.98.1、Node 22.23.2）：

```powershell
cargo build --manifest-path native/engine-prototype/Cargo.toml --target-dir artifacts/native-target --release --locked
cargo test --manifest-path native/engine-prototype/Cargo.toml --target-dir artifacts/native-target --locked
cargo clippy --manifest-path native/engine-prototype/Cargo.toml --target-dir artifacts/native-target --all-targets --locked -- -D warnings
cargo fmt --manifest-path native/engine-prototype/Cargo.toml --check
node --import tsx scripts/training/native/validate.ts --fixtures-only --output artifacts/training/rust-fixtures-new
node --import tsx scripts/training/native/validate.ts --all-workers --output artifacts/training/rust-full-new
node --import tsx scripts/training/native/sample-benchmark.ts --commands 1000 --output artifacts/training/rust-sampler-new
node --import tsx scripts/training/native/sampling/validate.ts --output artifacts/training/rust-tree-new
node --import tsx scripts/training/native/sampling/benchmark.ts --commands 1000 --output artifacts/training/rust-native-sampler-new
node --import tsx scripts/training/native/sampling/complete.ts --references artifacts/training/rust-fresh-games-20260927 --output artifacts/training/rust-native-complete-new
```

输出目录必须不存在。验证默认读取本机 `economics-20260926` 的四份 `worker-0.jsonl.gz` 原始轨迹，`--all-workers` 扩大到四组各8份、共32份；这些数据不跟踪进 Git。专项和驻留协议验证不依赖历史数据。Linux 可通过 `--executable` 指定无扩展名程序；2026-09-28 已在 Linux 通过 1101 规则夹具与 2202 根动作树/编码差分，macOS 尚未实测。推荐将 Cargo 产物放入已忽略的 `artifacts/`；默认 `target/` 也已排除版本管理和维护目录检查。

## 维护对应关系

图鉴、战斗参数、普通/终极/神龛抽取池和合成配方由 TS 初始化注入，不在 Rust 手工维护第二份数值表。下表路径相对各自的引擎目录。

| Rust                         | TS 参照                                                      | 约束                                                                        |
| ---------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------- |
| `model.rs`                   | `types.ts`、`core/traits.ts`、`core/state.ts`                | 热点字段有类型，其余 JSON 字段保留；相同 PRNG                               |
| `model.rs`、`core/shared.rs` | `core/branch.ts`、`core/clone.ts`                            | 实体/顶层字段写时复制；小型核心容器即时隔离；规则快照与正式成功结果独立拥有 |
| `core/geometry.rs`           | `core/geometry.ts`                                           | 邻居和路径顺序、叠放、地标层、2×2、部署行、入射方向                         |
| `core/stats.rs`              | `core/state.ts`、`core/traits.ts`                            | 属性、独立蓄力、光环、虹吸刷新                                              |
| `core/resolution.rs`         | `core/state.ts`、`core/protection.ts`、`core/event-facts.ts` | 保护来源、事件因果/坐标/身份快照、衍生物                                    |
| `commands/movement.rs`       | `commands/game.ts`、`commands/movement.ts`                   | 阶段、移动、完整操作预算、举旗                                              |
| `commands/combat.rs`         | `commands/combat.ts`、`core/attack-profile.ts`               | 路径/穿透、逐目标攻击、随机边界、反击、攻击后被动                           |
| `commands/damage.rs`         | `commands/combat.ts`、`setup/shrines.ts`                     | 逐份伤害、保护/免疫、死亡/人头、强夺、反应排队                              |
| `commands/reactions.rs`      | `commands/movement.ts`、`commands/combat.ts`                 | 冲撞、弹出、小屋/王城、死后射击、反射、牵引、结束后切回合                   |
| `commands/preparation.rs`    | `core/state.ts`、`setup/summoning.ts`、`setup/shrines.ts`    | 抽牌/整批克隆/改判、部署/装备/光环、费用和 RNG 原子性                       |
| `commands/lifecycle.rs`      | `commands/lifecycle.ts`                                      | 实际和个人时钟、结束效果、火焰/冰层、持续虹吸、下一召唤窗口                 |
| `commands/abilities.rs`      | `commands/abilities.ts`                                      | 全部主动技能、继承能力独立次数、巨大化与复活                                |
| `commands/spells.rs`         | `commands/abilities.ts`、`commands/combat.ts`                | 法术参数校验、反制顺序、基地来源、延迟效果                                  |
| `setup/synthesis.rs`         | `setup/synthesis.ts`                                         | 合成材料与落点原子移除，移除不当死亡                                        |
| `setup/shrines.rs`           | `setup/shrines.ts`                                           | 暗选、时钟快照、玉碎、强夺、地标重建                                        |
| `setup/runtime.rs`           | `commands/game.ts`、`ai/observation.ts`                      | 原生开局，公开观察显式白名单，暗选按观察方脱敏                              |
| `training/actions.rs`        | `commands/options.ts`、`ai/training/queries.ts`              | 公开动作说明、操作者权限、预检；全部参数域不做评分裁剪                      |
| `training/tree.rs`           | `ai/training/action-tree.ts`                                 | 惰性节点、选材/路径/方向顺序、合法叶子与回溯                                |
| `training/encoding.rs`       | `ai/training/encoding/`                                      | 词表、实体/引用顺序、存在掩码、未知字段报错                                 |
| `training/policy.rs`         | `scripts/training/economics/policy.ts`                       | 固定小网络、Float32 写回、独立策略随机流                                    |
| `training/sampler.rs`        | `scripts/training/economics/sample.ts`、`run.ts`             | 4096节点预算、可选回合外干预、自然终局与截断                                |
| `main.rs`                    | 实验驱动                                                     | 常驻 JSONL 协议、失败回滚、修订号，不是联网鉴权入口                         |

`movement_stats` 是 `getStats` 的移动依赖投影，省去移动入口不读取的攻击属性计算；因此收益不全来自语言。按 ID 查询目标时两端均只包装命中项。修改 TS 规则须同时维护对应 Rust 模块并重跑差分，不在 Rust 引入另一套 AI 评分。

TS 已有的攻击、部署、时钟、钩子、献祭、复活和小屋只读预检也已移植：先拒绝明显非法参数，再创建内部状态分支执行完整预检。目标/落点准备与正式结算共用函数；小屋落点仅在同一公开局面树内缓存。Rust 用 `Rc<Node>` 复用不可变节点，编码借用树的公开局面及棋子，避免重复反序列化；这些对应 TS 原有的对象引用语义，不改变候选范围或策略。

TS 生产引擎与本原型已统一采用状态分支算法。只有 `State::fork` / `Unit::fork` / `ValueMap::fork` 创建共享分支；普通 `Clone` 仍生成独立快照，不能为单端提速改成浅复制。每个实体、每个局面扩展字段分别作为写入隔离单元；`turns / bases / deployRows / pending / siphons / events` 在创建分支时立即复制。规则内部快照继续深复制，正式成功由 `transition` 导出独立状态；预检和失败直接丢弃分支。TS 用写入拦截，Rust 用引用计数和可变借用实现同一流程。算法映射、原型计时与生产接入验收见[状态分支记录](../../docs/ai/performance/STATE-SHARING-2026-09-27.md)，Node 性能下降不构成保留算法分叉的理由。

## 协议与信任边界

stdin/stdout 每行一个 JSON，stdout 不混日志。先调用 `{op:"init", protocol:"haojie-native-engine-v4", ruleset, catalog, combat, summonPools, recipes}`。字段分别来自 TS `RULESET_ID / CATALOG / COMBAT_RULES / SYNTHESIS_RECIPES`；`summonPools` 包含 `normal / ultimate / shrine`。拒绝旧协议和不完整初始化，不静默补数值。握手返回完整命令清单及 `completeEngine:true`；它表示实现范围，不是所有规则组合都已穷尽证明。

动作树/采样扩展在同一初始化额外接收 `encoding:{...ENCODING_SCHEMA, decision_stages:DECISION_STAGES}`。旧规则请求保持 v4 兼容；原生采样拒绝未提供或结构不符的编码词表。印刷属性的可选字段保留原始存在性，不能把缺失 `mage/aura` 补成 false 后送入旧模型。`serde_json` 开启 `preserve_order` 并锁定依赖；编码枚举对象键时仍遵守 JS 数字键优先、其余键按插入顺序的规则。

只接受正式 TS 引擎重建或原生创建的规范局面和已解析命令，不替代 `parseSession / parseCommand`。权威状态和正式 seed/rng 只用于规则结算与重放，不能送入策略。JSON 表示不保留 JS 环、共享引用或 `undefined` 属性身份。

- `create`：`{seed, rules:"classic"|"shrine", clearHistory?}` 原生创建新局，返回递增 `revision` 和当前决策方的公开 `observation`。
- `observe`：`{viewer?:1|2}` 返回公开白名单，不含 seed/rng、日志、事件；未共同揭示时只包含观察方自己的神龛选择。
- `run`：`{jobs:[{state, probes:Command[], command?:Command}]}` 返回预检和可选命令的完整结果，用于独立差分。
- `load / bench`：加载验证工作集后内部重复计算；内部计时不含解析与通信，必须单列。
- `reset`：`{state}` 原子替换驻留局面，递增修订号；失败保留旧局面。
- `step`：`{revision, commands, trace?, clearHistory?, observe?}` 逐条提交成功命令并递增修订号。遇首个失败停止并保留成功前缀；整批解析错误或过期修订号不执行。`trace` 返回清理前的逐步状态供差分；`clearHistory` 在每步成功后清理事件/日志；`observe` 在批次结束后返回决策方公开观察。
- `export`：返回 `{revision,state}`，仅供差分和权威持久化，不能当公开观察。
- `training-nodes`：`{observation,actor,cursors,encode?}`，返回惰性节点；开启 `encode` 同时返回实体/候选张量和固定 TinyPolicy 的 logits，空分支返回 null。此入口用于逐项跨语言验证。
- `sample-game`：`{seed,rules,maxCommands,maxPlies,policy?:"tiny"|"uniform",policySeed?,samplerSeed?}`。原生进程内部连续采样，返回实际命令、状态、分项计数及验收用完整终态/双方公开观察。与 `reset/step` 驻留会话独立；计数或回合截断时收益为 null，异常保留成功前缀并报告。正式种子只交给权威开局，树/编码器只接收脱敏观察，策略随机流独立。宿主取消时终止此独立进程，不在候选选择中读墙钟。

规则拒绝为 `invalid`；只读预检抵达随机边界为 `uncertain`。未知命令、非规范状态或未知反应仍可返回 `unsupported`，完整引擎验收遇到它直接失败，不跳过样本。协议错误返回 `error`；客户端默认60秒超时，完整采样验收显式设为600秒，超时终止子进程，没有 TS 回退。单条失败丢弃命令副本，费用、事件、序号和 RNG 都不提交。

## 验证与计时

`validate.ts` 先保存运行器、全部依赖及 Rust 源码指纹和可执行文件，再运行该冻结二进制。专项比较全部117格、错误原因、随机边界和完整结算；真实记录通过共享读取器校验后重建。每条命令独立比较完整状态，另从每局开局在 Rust 连续推进，每64条核对驻留终态，不能通过每步重置掩盖漂移。对来源中的真实终局另保存整局回放计时；原来中断的记录仍为 unknown。

核心微基准、32段连续命令和完整终局回放分别计时。驻留总时间包含加载、命令通信和最终导出；这些工作量不包含选招或网络。每组先预热，三轮交错，报告中位数。不同阶段连续段长度变化时不能直接相除归因于优化。

`sample-benchmark.ts` 使用完全相同的 TS 公开动作树、编码和随机初始化小网络，只切换权威环境；包括每步 JSON IPC、公开观察、选招和记录指纹。进程启动及静态表初始化在计时前，创建新局在计时内；文件压缩和差分审计在计时后。两种规则各预热一次、三轮交错，核对所有选中命令、最终权威局面及双方观察。达到命令/ply上限仍为截断，不填胜负标签，也不外推终局吞吐或30天训练成本。

`sampling/validate.ts` 对专项的双方视角比较根节点、第一层参数和合法命令的完整深层路径，并核对编码及网络输出。`sampling/benchmark.ts` 在同一批次内完成原生采样，与 TS 原算法预热后交错测量，逐命令和工作量计数必须相等。两端均暂存诊断命令，计时不含记录指纹、压缩或验收重放；Rust 的一批 IPC 和终态导出计入总时间。`sampling/complete.ts` 从种子独立采样到终局，再与先前保存的 TS 选招完整轨迹比较；参照命令不会送入原生选择器，此工具不报告加速倍率。

正式保存的对照轨迹使用 TS 记录协议并经过共享读取器校验。即使保留 JSON 插入顺序，Rust 状态重建的键序仍不能冒充旧格式 Observation 指纹。当前通过计时外的 TS 重放生成并审核 `.jsonl.gz`；它是可验证的兼容出口，仍有额外成本。正式模型批量推理、搜索用可枚举随机源、生产记录性能和长期并行经济性需继续验收。
