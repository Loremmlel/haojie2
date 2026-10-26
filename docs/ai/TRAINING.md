# 训练与规则验证维护入口

训练只使用[独立 C++ / LibTorch 系统](../../training/README.md)。C++ 宿主持有 CUDA、模型、优化器、调度与检查点；Rust `online/runtime.rs` 和 `online/actor.rs` 持有正式状态、采样随机流和逐局动作状态机。

TypeScript 保留游戏规则参照、公开观察、惰性动作树、编码以及历史记录审核。它们用于双端语义差分和存档重放，不提供另一套训练宿主。旧教师生成、纠错再标注、旧模型解码和常驻 JSONL 训练服务已删除。

| 路径                                           | 职责                                           |
| ---------------------------------------------- | ---------------------------------------------- |
| `src/match/training.ts`                        | 规则测试与历史审核的权威环境；截断结果保持未知 |
| `src/ai/training/action-tree.ts`、`queries.ts` | 完整公开候选域与公开预检                       |
| `src/ai/training/encoding/`                    | 与 Rust 对应的公开特征编码                     |
| `src/ai/training/simulation.ts`                | 公开观察上的规则模拟和概率查询                 |
| `scripts/training/records/`                    | 历史增量记录的重放、校验与压缩读取             |
| `scripts/training/native/`                     | 规则包生成、双端差分、独立安装验收             |

历史审计仍可使用 `npm run train:inspect -- ...`。`train:import-save` 将已验证游戏存档转为可审核的命令记录；这些旧格式记录不会送入新连续轨迹学习器，也不补造行为概率。缺少终局的历史记录保留 `returns=null`。

原生验证工具按职责分组：`fixtures/` 构造规则边界，`validation/` 运行正式状态／命令差分，`sampling/` 比较公开候选和编码，`pipeline/` 审核历史规范记录，`ci/` 提供安装、规范哈希及源码打包入口。正式训练性能工具集中在 `training/native/diagnostics/`。

新轨迹保存学习席位的每个随机微步、类型化公开根、候选支持集、实际行为概率、版本及后继。当前路径和每级精确排除 frontier 取代完整事件历史；对手动作之间暂存转换，直到同席位下一状态或真实终局。训练依据当前参数重新编码，推理缓存不作为训练特征。详细契约见[原生说明](../../training/native/README.md)。
