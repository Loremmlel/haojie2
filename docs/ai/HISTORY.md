# AI / 训练历史索引

当前用法见[状态入口](STATUS.md)。已完成的调查、阶段报告、测量数据和一次性实验脚本从下面的固定版本查阅；原文中的设备、耗时、下一步和工作区路径属于记录当时。

| 内容                                             | 固定版本                                                                                                                                                                                                                         |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MC-Q v2 交付、机制验证和自然局验收               | [2026-10-04 交付记录](https://github.com/Loremmlel/haojie2/blob/d920e181dccb425c0a77d57b473fb98f45163b69/docs/ai/console/DELIVERY-2026-10-04.md)                                                                                 |
| Rust 内核、Python 执行、常驻采样的测量及失败证据 | [性能报告](https://github.com/Loremmlel/haojie2/tree/d920e181dccb425c0a77d57b473fb98f45163b69/docs/ai/performance)                                                                                                               |
| 早期 CPU / GPU / 浏览器基准数据                  | [原始基准](https://github.com/Loremmlel/haojie2/tree/d920e181dccb425c0a77d57b473fb98f45163b69/docs/ai/benchmarks)                                                                                                                |
| 采样经济性协议及性能实验脚本                     | [研究协议](https://github.com/Loremmlel/haojie2/tree/d920e181dccb425c0a77d57b473fb98f45163b69/docs/ai/research)、[实验代码](https://github.com/Loremmlel/haojie2/tree/d920e181dccb425c0a77d57b473fb98f45163b69/scripts/training) |
| 早期 AI 反馈调查、预算与回合收尾修复             | [AI 调查记录](https://github.com/Loremmlel/haojie2/tree/d920e181dccb425c0a77d57b473fb98f45163b69/docs/ai)                                                                                                                        |
| 早期训练路线与逐轮实验进度                       | [历史进度](https://github.com/Loremmlel/haojie2/blob/fcf37e73f91629965255b6fa4a5da1c80e7fa71a/docs/ai/history/TRAINING-PROGRESS.md)                                                                                              |
| 已执行的本地产物回收范围                         | [2026-10-03 记录](https://github.com/Loremmlel/haojie2/blob/d920e181dccb425c0a77d57b473fb98f45163b69/docs/maintenance/CLEANUP-2026-10-03.md)                                                                                     |
| 早期 UI 工程选型                                 | [2.0.1 工程记录](https://github.com/Loremmlel/haojie2/blob/d920e181dccb425c0a77d57b473fb98f45163b69/docs/ENGINEERING.md)                                                                                                         |
| 2026-10-21 规则引擎采样优化                      | [收束记录](https://github.com/Loremmlel/haojie2/blob/e91856fde94075c0f18e1ba8789d34d697ee8c49/docs/ai/experiments/ENGINE-SAMPLING-2026-10-21.md)                                                                                 |
| 独立流水线前的训练性能与已撤回实验               | [阶段测量](https://github.com/Loremmlel/haojie2/blob/e91856fde94075c0f18e1ba8789d34d697ee8c49/training/PERFORMANCE.md)                                                                                                           |

作者[原始 AI 实战观察](AI-HARD-PLAYTEST-2026-09-07.md)、[规则反馈](../feedback/)和版本对应的 `docs/playtests/` 回放保留原内容。历史回放使用各自原规则提交验证，当前规则回放在 `docs/playtests/current/`。模型、原始轨迹、优化器和其他本地产物由使用者保管，文档清理不会删除它们。

[历史回放工作流](../../.github/workflows/history.yml)可手动运行全部原规则指纹检查；默认 CI 运行当前规则回放。
