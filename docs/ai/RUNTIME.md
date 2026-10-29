# 运行内核与双端对应

本分支保留 TS / Rust 游戏规则内核，重写普通局训练，并完整对应三档手工 AI。训练采用单进程、Rust 固定线程池与 PyTorch；方法见[普通局训练](TRAINING.md)，线程及恢复契约见[运行时架构](TRAINING-RUNTIME.md)。

## 规则内核

| 责任                 | TypeScript                    | Rust                                            |
| -------------------- | ----------------------------- | ----------------------------------------------- |
| 状态、实体与工作分支 | `src/engine/runtime/`         | `src/core/entities.rs`、`shared.rs`、`model.rs` |
| 几何、占位和路径     | `src/engine/core/geometry.ts` | `src/core/geometry.rs`                          |
| 命令预检和提交       | `src/engine/commands/`        | `src/commands/`、`core/command.rs`              |
| 召唤、合成和模式设置 | `src/engine/setup/`           | `src/setup/`                                    |
| 公开观察             | `src/ai/observation.ts`       | `src/setup/runtime.rs`                          |

TS 外部快照与内部拥有型局面分开；工作分支按写入需要分离实体和冷数据。Rust 使用 `Rc` 与写入前分离，正式快照导出保持隔离。语言内存机制可以不同，规则、命令顺序、候选域及随机消耗须保持对应。

输入不变、失败不部分提交。单位身份、原字段顺序、扩展字段、缺失与 null、历史回放与正式 RNG 语义继续保留。冲撞的 `transitUnitId` 记录实际穿行者；死亡记录、延迟效果、保护来源和时钟快照不能在分支之间互相污染。

公开观察只包含白名单，禁止泄露 seed/rng、整份 Session 或未揭示选择。查询缓存只在当前只读局面内使用；新局面重新建立缓存。手工 AI 继续共用当前规则与独立概率模拟。

## 验证

三档手工 AI 与训练接口对应如下，Rust 路径以 `native/engine/src/` 为根：

| 责任                      | TypeScript                                   | Rust / Python                                                 |
| ------------------------- | -------------------------------------------- | ------------------------------------------------------------- |
| 局面评价、几何与威胁      | `src/ai/evaluation/`                         | `ai/evaluation.rs`、`spatial.rs`、`threats.rs`                |
| 候选排序与分组            | `src/ai/planning/candidates.ts` 等           | `ai/candidates.rs`、`ranking.rs`、`settings.rs`               |
| 束搜索、回应搜索          | `src/ai/planning/search.ts`                  | `ai/search.rs`                                                |
| 概率展开与采样            | `src/ai/simulation.ts`                       | `ai/distribution.rs`、`core/chance.rs`                        |
| 观察键与决策种子          | `src/ai/observation.ts`                      | `ai/keys.rs`                                                  |
| 回合预算与计划缓存        | `src/ai/budget.ts`、`planning/plan-cache.ts` | `ai/search.rs` 的 `teacher_next`                              |
| 无评分合法参数域          | `src/ai/classic/options.ts`                  | `commands/options.rs`                                         |
| 合法参数展开              | `src/ai/classic/parameters.ts`               | `learning/parameters.rs`                                      |
| 分层合法参数解码          | `src/ai/classic/decoder.ts`                  | `learning/decoder.rs`                                         |
| 实体关系几何              | `src/ai/classic/relations.ts`                | `learning/encoding.rs`                                        |
| 公开张量编码与进程内接口  | 新训练专用                                   | `learning/encoding.rs`、`ffi.rs`，`training/haojie/native.py` |
| 神经网络 PUCT 与搜索 RNG  | 新训练专用                                   | `learning/search.rs`、`random.rs`                             |
| 固定线程池与批量请求      | 新训练专用                                   | `learning/pool/`、`training/haojie/native_pool.py`            |
| BF16 推理、学习与运行协调 | 新训练专用                                   | `training/haojie/model.py`、`runtime/`                        |

手工 AI 的字段顺序、种子、候选顺序、预算与概率切点对应 TS。Rust 保留原输入字段顺序，新增对象按 TS 构造顺序生成决策键；命令内部顺序仅服务算法键，不改变对外命令。只读几何缓存随局面作用域释放，预览对象不误用原实体缓存。普通局老千仍参与手工评价、召唤候选和训练编码。

训练运行不需要 TS/Node；TS 用于网页和双端开发验收。神经网络搜索是两端手工 AI 之外的新算法，不替换网页三档行为。训练使用条件解码与渐进扩宽，参数前缀不计入游戏树深度。普通局夹具逐前缀比较首末合法路径，并比较全部实体关系特征。

原生搜索从提交 `7cc4ba9` 的 Python 实现整体迁入 Rust。默认预算仍为 25% 决策 64 次模拟，其余 16 次，深度 16 条完整命令；策略、价值、辅助目标和 PyTorch 原生 BF16 数学保持不变。Python 正式入口只调度批量请求、学习和记录；旧 Python 搜索由正确性测试从 Git 读取。差分核对逐条请求输入、命令、根访问和先验、监督以及最终 RNG。

每局的规则局面、搜索图、PCG64 和手工 AI 计划固定归属一个 Rust 线程，默认 4 线程 × 16 局。Python GPU 主线程借用原生请求数组并返回 BF16 位模式，协调/I/O 线程复制正式记录、抽取回放及保存。保存先暂停、排空并通过屏障；恢复保留 v2 格式和 worker/lane 映射，从原命令起点及固定模型重算未完成搜索。

搜索随机分布及 f32 exp 直接复用 NumPy 原生函数，保持原 Python 搜索的随机消耗与舍入。当前桥固定 NumPy 2.5.3，并检查私有分布符号和 `numpy_1.24_ufunc_call_info` capsule 的使用条件；此次功能验收搭配 PyTorch 2.14.0+cu130。

`scripts/native/engine/fixtures/` 保存规则夹具，`validation/` 比较两端预检、成功/失败命令、完整后继、输入不可变和驻留状态恢复。已有神龛规则夹具仍用于保证原游戏不退化；新训练只做普通局。

```sh
npm run native:rules:check
npm run native:test
npm run native:check
npm run native:build
npm run native:validate -- --output artifacts/native-rules
node --import tsx scripts/native/learning/validate.ts --evaluation --candidates --search --decoder
```
