# TypeScript 训练接口与数据工具

日常自对弈和持续学习使用[本地训练控制台](../../training/CONSOLE.md)或[云端服务](../../training/COLAB.md)。这里维护教师采样、旧记录编码、模型 CLI 对战和独立的 TS 规则参照，保留各自的数据语义；共享原生执行层及旧记录审核边界见[原生支持](../../training/NATIVE.md)。

网页或 CLI 增量存档可用 `npm run train:import-save -- 输入.json 输出.jsonl.gz` 转为训练轨迹，再交给已有审计和编码器。起点、未结束局标签与公开信息边界见[存档与训练转换](../session/SAVES.md)。

## 模块职责

| 文件                            | 责任                                                      |
| ------------------------------- | --------------------------------------------------------- |
| `src/match/training.ts`         | 常驻权威环境，轻量status、公开observation/frame和原子step |
| `src/ai/training/queries.ts`    | 未经策略裁剪的动作描述/参数域，公开预检和共享几何查询     |
| `src/ai/training/simulation.ts` | 公开观察上的独立随机模拟及概率分布查询                    |
| `src/ai/training/teacher.ts`    | 只接收Observation的手工教师，复用预算和plan-cache         |
| `scripts/training/serve.ts`     | 支持多环境/batch的常驻JSONL标准输入输出进程               |
| `scripts/training/self-play.ts` | 热启动教师对弈及可选流式公开样本记录                      |
| `scripts/training/benchmark.ts` | 规则、复制、编码、预检、动作描述、对弈和多进程基准        |

环境用 `applyRuntimeCommand` 接续拥有型工作局面，与公开 `applyCommand` 共用原子规则内核；成功后清理 events/log，正式随机数、serial 和规则状态保持一致。它不保留 Session 的撤销历史。`observation` 导出独立快照，内部 `view` 提供同一决策中的公开只读视图。

`status()`是轻量状态；`step(actor, command)`只返回status，需要网络输入时显式调用`observation(viewer)`或`frame(viewer)`，避免重复观察复制。fromState仅供宿主导入可信引擎局面，策略和JSONL协议没有权威局面导出接口。

`terminated`表示规则终局，`truncated`表示训练命令数/相对ply上限；二者分开。只有规则终局提供returns，值是玩家1/2的终局收益（胜+1、负−1、平0）。returns不是应随轮询重复累加的奖励；截断的returns为null，不按基地血量判胜，也不生成价值监督标签。上限可能停在反应/部署中，局面保留，必须明确reset，不自动吞掉强制动作。

`toPlay`表示主要决策方，不能作为全部权限的锁。客户端也可以显式给另一席位查询观察/动作并提交合法回合外巨大化；共同揭示前，两席位各自查询的观察只包含自己可知的选择。协议是本地受信训练工具，不是联网身份认证服务。

## 常驻JSONL协议

使用一个长期运行的进程，每行一条JSON请求/响应。直接启动可避免npm的普通运行横幅混入stdout：

```sh
node --import tsx scripts/training/serve.ts
# 等价的静默npm入口
npm run --silent train:serve
```

示例请求（连续发给同一进程）：

```jsonl
{"id":1,"op":"reset","env":"game-a","options":{"seed":7,"rules":"classic","maxCommands":3000,"maxPlies":100}}
{"id":2,"op":"actions","env":"game-a","actor":1}
{"id":3,"op":"inspect","env":"game-a","actor":1,"command":{"type":"summon"}}
{"id":4,"op":"step","env":"game-a","actor":1,"command":{"type":"summon"},"observe":false}
{"id":5,"op":"observe","env":"game-a","viewer":1}
{"id":6,"op":"sample","env":"game-a","actor":1,"command":{"type":"summon"},"sampleSeed":31415}
{"id":7,"op":"close","env":"game-a"}
```

成功返回`{id,ok:true,result}`，失败返回`{id,ok:false,error}`，close没有result。step默认返回下一决策方的frame，可显式指定viewer；actor缺省取当前toPlay，与返回观察的viewer无关，训练客户端宜显式传入actor。`observe:false`仅返回status。inspect返回available/uncertain/invalid，uncertain在第一次未知随机请求前停止，不能当作权威接受保证。

`sample`返回一次独立模拟后的公开观察；`distribution`返回`{sampled,attempts,outcomes:[{observation,weight}]}`。sampleSeed是独立实验采样编号，不是实局seed。神龛暗选阶段明确拒绝这两种转移模拟，不能偷偷补造对手提交。真实对弈暗选仍可用step正常完成。

批量用于摊薄进程通信开销，可以混合环境；逐项独立返回错误，不提供跨命令/跨环境事务：

```text
{"op":"reset","env":"a","options":{"seed":7}}
{"op":"reset","env":"b","options":{"seed":9}}
{"op":"batch","requests":[{"op":"step","env":"a","actor":1,"command":{"type":"summon"},"observe":false},{"op":"observe","env":"b","viewer":2}]}
```

单进程最多128个环境、每批最多256项、不支持嵌套batch，单行协议限制4MiB。一个进程串行处理命令，需要并行采样时启动多个独立进程；不能把批量请求误当成并行执行。stdout只输出JSONL，错误不关闭整个服务。EOF/进程退出释放所有环境，不自动存盘。

## 动作表示与覆盖边界

actions提供`factorized-unpruned`表示，包括共享unitActions/cardActions、常驻/人头召唤、两抽选择、神龛选择（玉碎奇偶均包含）、反应、时钟、合成窗口。每项包含command草稿与SelectionStep；有自选召唤能力时额外给chosenKinds，合成给全部materialIds与materialCount=3。action id沿用能力标识，跨棋子不保证唯一，应结合command中的身份或当前数组索引。

返回targetIds、117格points、四方向、deathIds作为完整参数域。它们不是全量合法动作掩码，也不保证每个草稿都有合法完成方式。策略逐项选择参数，随后inspect/step验证；路径保留为显式坐标序列。旧craft命令是firelord合成的兼容别名，统一用synthesize表示即可，原始JSON命令仍接受该别名。

geometry复用共享查询：`{type:"attack",unitId,targetId}`返回可选方向路径；`{type:"skill",targetId}`返回巨大化落点；`{type:"synthesize",recipeId,materialIds}`返回指定三材料移除后的落点。方向路径查询不能代替手绘路径的完整空间，手绘path仍由step中的validAttackRoute校验。

当前教师仍使用原有启发式候选，不会探索全部路径/组合，也不会自动新增回合外响应窗口。这是热启动基线的策略限制，不是引擎权限限制。共享公开状态编码与分步动作树已实现，覆盖及参数/完整命令的区别见[编码说明](../../training/ENCODING.md)。模型 CLI 可显式选择有限候选的 beam / MCTS / Gumbel 解码；控制台使用独立的 MC-Q 分解采样。两者的预算、样本和评测结果分别记录。

纯网络CLI对战已由`npm run train:match`提供，复用分步动作树的贪心排序/空分支回溯；Node负责引擎，常驻Python只接收公开编码张量。网络预算耗尽和推理异常明确暂停，单候选免推理，最终命令仍交给引擎原子裁定。用法见[Python 训练包](../../training/README.md)，既往运行结果见[历史索引](HISTORY.md)。

## 生成热启动样本

```sh
npm run train:selfplay -- --games 10 --seed 20260922 --rules classic --difficulty easy --nodes 100 --plies 100 --commands 3000 --output artifacts/teacher-classic.jsonl.gz
npm run train:selfplay -- --games 10 --rules shrine --difficulty hard --output artifacts/teacher-shrine.jsonl.gz
```

不指定output只输出摘要，便于测量不含记录IO的吞吐；指定后流式写出并遵守背压，拒绝覆盖已有文件。游戏种子依次递增；难度easy/medium/hard，不给nodes时共享现有production-work预算，指定时为每次决策固定节点目标，均不使用墙钟停止。

格式为`haojie-training-record-v1`，记录分三种：game保存种子、规则、实际截断上限与教师设置；sample只保存操作者、实际命令、前后指纹及教师统计；outcome保存真实结果和耗时摘要。没有逐步Observation或异常快照。`.jsonl.gz`启用流式gzip，`.jsonl`可用于调试；大批量集合和必要统计仍随命令数量增长，不保证所有文件小于1MB。

编码、数据检查、预算审计和网络报告共用重放入口，严格检查版本、命令顺序、权限、指纹及结果。操作者Observation在内存中生成并经原有编码器输入网络，宿主种子/初始局面不进入特征。teacherStats不是MCTS访问分布；计时不参与预算或选招。缺少结束行的完整命令前缀显式标记中断，不能产生价值标签；损坏JSON/gzip报错。prepare通过管道直接写二进制`.pt`，训练阶段不再重放。旧快照训练JSONL不兼容，历史报告保留，重新采样使用新格式。

SIGINT在命令边界取消，同步教师搜索尚不能在调用内部抢占。取消后丢弃本次未执行命令，不自动结束回合。规则版本改变时冻结旧数据与代码，禁止改写历史指纹。

## 性能评估与验收

```sh
npm run bench:training -- --output artifacts/training-benchmark.json
# 快速重测纯环境，不含教师对弈
npm run bench:training -- --skip-selfplay --iterations 500 --repeats 30 --rounds 3 --workers 4
# 延长教师测试，仍明确保留未决结果
npm run bench:training -- --games 2 --plies 100 --commands 3000 --nodes 100
```

报告包含CPU/内存/Node版本、基准参数、当前提交、相关源码SHA256、固定语料SHA256与最终指纹。源码hash覆盖未提交文件内容，可区分同提交上不同工作副本。微基准预热后取多轮中位数，多进程基准在子进程预热完成后计时。

区分clone、observe、JSON编码、applyCommand、独立随机转移并输出观察、公开预检、动作空间查询，以及相同79条实录的引擎/训练/Session序列化/协议编码开销。Session测试不含磁盘IO和教师trace，协议编码测试不含OS管道传输，均明确标记。演示局是压力样例，不是自然对局分布。教师测试另测经典/神龛实际对弈，报告完成、截断、命令数与模拟数，不计算对作者胜率。

TS 行为测试位于 `tests/training`，随 `npm test` 运行；Python 测试见训练包说明。当前原生差分验证位于 `scripts/training/native/`，核对规则、候选、编码、哈希和 MC-Q 选择。历史性能实验、阶段报告和原始测量见[历史索引](HISTORY.md)。
