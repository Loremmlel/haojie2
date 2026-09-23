# AI训练路线与高速对弈接口

2026-09-22讨论与实现记录。目标设备为普通桌面/笔记本，CPU推理也须实用，可使用WebGPU。目标棋力是对熟练作者取得超过80%的胜率；这是后续实战验收目标，不是当前能力或训练规模保证。

持续更新的阶段状态、实验结果和下一步见[训练进度](TRAINING-PROGRESS.md)。

网页/CLI增量存档可通过`npm run train:import-save -- 输入.json 输出.jsonl`转换为现有训练样本，并沿用审计与编码器；起点、未结束局标签和隐私边界见[存档与训练转换](../session/SAVES.md)。

## 已讨论的训练方向

采用小型实体Transformer策略/价值网络、显式处理随机事件的MCTS、现有手工AI热启动，再持续自对弈。棋子价值不均不妨碍强化学习；真正困难的是多实体能力关联、组合动作、长期收益、随机性和罕见场景覆盖。

网络读取结构化Observation：棋子/手牌/地标、坐标与尺寸、当前有效能力（含继承能力）、装备、状态来源、实际与个人时钟、独立行动预算、基地、人头、阶段和反应。数值及规则身份继续来自catalog与共享引擎查询，不复制UI文案或让模型靠解析中文规则恢复属性。

实体注意力是候选结构，不保证比简单网络更省样本或更强。精确规则已知，优先直接调用引擎模拟，不先训练一个近似规则转移模型。

搜索必须遵守：

- 抽牌、暴击、免疫等按真实概率枚举或独立采样，随机分支不能选最幸运结果。
- 不读取正式seed/rng，不从历史推算未来随机序列。实局PRNG和模拟随机源独立。
- 同一玩家可连续发出多个命令，价值回传按实际决策所有者处理，不能每层自动变号。
- 反应由所属玩家决策；回合外巨大化须保留相应权限和明确的响应调度。
- 目前手牌公开；神龛共同揭示前的暗选才有额外隐藏信息。普通公开局面搜索不能补造对手选择；未来需要信息集/混合策略处理。
- 路径、三材料合成、多目标技能适合分层选择、逐步扩展，不永久限制在手工AI裁剪后的候选内。

训练顺序：

1. 用不同种子、先后手、规则模式和教师难度产生公开观察/动作样本，模仿策略热启动。
2. 价值主要学习真实终局结果，手工评分最多作辅助目标，不能直接当作实测胜率。
3. 接入网络与MCTS，将搜索访问次数分布作为策略目标，终局结果作为价值目标，持续迭代。
4. 保留历史模型、手工AI和不同风格对手，检查策略互相克制和遗忘。
5. 用合法的局面、罕见组合及作者实战补齐薄弱点，训练集和最终实战测试集分离。

几万至几十万局是实验规模，是否足够由验证曲线决定。10万局×每局300个决策×每次200次搜索模拟为60亿次模拟；一条模拟可能需要多次规则转移，缓存、批处理和网络成本也必须计入。不能用裸引擎命令吞吐直接估算完整训练耗时。

## 模型大小与浏览器部署

| 候选规模 | FP16原始权重 | 定位                                           |
| -------- | ------------ | ---------------------------------------------- |
| 10–15M   | 20–30MB      | 正式首版优先，兼顾普通笔记本CPU与搜索吞吐      |
| 20–30M   | 40–60MB      | WebGPU及棋力对照实验                           |
| 50M      | 100MB        | 后续大模型/教师候选；默认CPU＋MCTS偏重，需实测 |

首个正式候选为6层Encoder、隐藏维度384、6个注意力头、FFN维度1536，共享策略/价值主干。现已实现11,547,266参数的[PyTorch骨架与训练子项目](../../training/README.md)，采用候选评分头和价值头，并比较CPU/XPU与训练精度。完整游戏特征编码、候选生成及浏览器部署尚待接入；合成数据训练仅验证计算闭环，不代表已有棋力。早期更小配置只用于快速验证。

参数量不是延迟：实体数量、输出动作表示、算子支持、批大小、CPU/GPU传输和搜索次数都会影响速度。分档填充实体，不因固定token长度截断克隆、叠放或必要状态。一次决策若串行评估200个新叶子，单次10ms就仅网络耗时2秒；本游戏一回合有多个决策，须评估完整回合等待时间。

WebGPU可评估FP16，但要检测shader-f16与实际算子支持。CPU/WASM不统一要求FP16，应比较FP32和受支持的8位量化版本，重新验收量化后的策略/价值与棋力。稳定形状、兼容算子和批量叶子推理值得测量；批次过大也可能削弱树搜索的及时反馈。

长期训练前还应在目标浏览器测约12M/25M模型的加载、编译预热、batch=1及小批量推理、峰值内存。该浏览器模型基准尚未实现；现有Node规则基准和PyTorch训练基准不能代替浏览器推理测试。

保持单个离线HTML。50M FP16权重约100MB，Base64内联约133MB，另有运行时、解码和激活内存。实际file://环境必须验证WASM多线程能力；不能套用开启跨源隔离的网页多线程数据。模型训练在独立工具中进行，不给发行HTML加入训练服务或隐式下载。

## 本轮已实现的接口

| 文件                            | 责任                                                      |
| ------------------------------- | --------------------------------------------------------- |
| `src/match/training.ts`         | 常驻权威环境，轻量status、公开observation/frame和原子step |
| `src/ai/training/queries.ts`    | 未经策略裁剪的动作描述/参数域，公开预检和共享几何查询     |
| `src/ai/training/simulation.ts` | 公开观察上的独立随机模拟及概率分布查询                    |
| `src/ai/training/teacher.ts`    | 只接收Observation的手工教师，复用预算和plan-cache         |
| `scripts/training/serve.ts`     | 支持多环境/batch的常驻JSONL标准输入输出进程               |
| `scripts/training/self-play.ts` | 热启动教师对弈及可选流式公开样本记录                      |
| `scripts/training/benchmark.ts` | 规则、复制、编码、预检、动作描述、对弈和多进程基准        |

环境每步仍调用applyCommand，结算期间完整生成事件，成功后才清理events/log。不保存Session的60步历史，不逐步解析/保存整份存档，不计算trace；正式随机数、serial、规则状态与原版一致。没有修改引擎、旧CLI、存档格式或浏览器行为。

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

当前教师仍使用原有启发式候选，不会探索全部路径/组合，也不会自动新增回合外响应窗口。这是热启动基线的策略限制，不是引擎权限限制。共享公开状态编码与分步动作树已实现，覆盖及参数/完整命令的区别见[编码说明](../../training/ENCODING.md)。未来MCTS需实现渐进扩展及响应调度；当前接口尚未接入MCTS、完整命令的精确合法掩码、浏览器网络推理或信息集求解器。

纯网络CLI对战已由`npm run train:match`提供，复用分步动作树的贪心排序/空分支回溯；Node负责引擎，常驻Python只接收公开编码张量。网络预算耗尽和推理异常明确暂停，单候选免推理，最终命令仍交给引擎原子裁定。用法与统计边界见[网络CLI对战](../../training/README.md#网络cli对战)，实际运行结果见[训练进度](TRAINING-PROGRESS.md)。

## 生成热启动样本

```sh
npm run train:selfplay -- --games 10 --seed 20260922 --rules classic --difficulty easy --nodes 100 --plies 100 --commands 3000 --output artifacts/teacher-classic.jsonl
npm run train:selfplay -- --games 10 --rules shrine --difficulty hard --output artifacts/teacher-shrine.jsonl
```

不指定output只输出摘要，便于测量不含记录IO的吞吐；指定后流式写出并遵守背压，拒绝覆盖已有文件。游戏种子依次递增；难度easy/medium/hard，不给nodes时共享现有production-work预算，指定时为每次决策固定节点目标，均不使用墙钟停止。

记录分三种：game为宿主复现元数据（种子、规则与教师设置）；sample为观察、操作者、实际命令及教师统计；outcome为终局/截断与returns，另含观察、教师、落子的耗时分解及出现过的最大棋子数。计时只用于性能报告，不参与预算分配或选招。只将sample.observation作为网络输入，game中的种子不可送给策略。teacherStats不是MCTS访问分布；不伪造visits或将手工分数当作胜率。按game分组连接终局标签，截断/中断且无outcome的样本不能冒充已完成比赛。

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

本机结果与Node是否足够的判断见[Node性能评估](TRAINING-NODE-2026-09-22.md)；11.55M网络的CPU/XPU与精度对比见[PyTorch训练性能](TRAINING-PYTORCH-2026-09-22.md)。不在CI中用机器耗时写脆弱阈值；TypeScript行为测试在tests/training，沿用npm test自动运行，Python测试命令见训练子项目说明。

最终80%目标需冻结规则、模型和推理预算，使用未参与训练的种子、平衡先后手，并让作者熟悉模型打法。近似独立且条件稳定时，100局80胜的95% Wilson区间约71%–87%，不足以可靠断言真实胜率至少80%。不得把短局、自对弈或固定种子回放等同于对作者胜率。

## 研究与实现参考

- [AlphaStar论文](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)：实体注意力、模仿学习和历史对手联盟的参考；其强化学习算法不是MCTS。
- [AlphaZero论文](https://arxiv.org/pdf/1712.01815)：策略/价值网络、搜索分布与自对弈迭代。
- [ReBeL论文](https://arxiv.org/abs/2007.13544)：隐藏信息下不能直接照搬完整信息搜索。
- [ONNX Runtime Web性能指南](https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html)：CPU精度选择、多线程条件、WebGPU图捕获与传输开销。
- [Chrome WebGPU半精度说明](https://developer.chrome.com/blog/io24-webassembly-webgpu-2)：shader-f16的可用性与性能影响。
- [Node 22流文档](https://nodejs.org/docs/latest-v22.x/api/stream.html)：常驻管道背压。
