# 无 Node 的 Rust＋Python 训练入口

复用 `entity-transformer-v2`、候选交叉熵和终局价值损失、AdamW、分片与检查点。Rust 负责环境、公开观察、参数树、执行、记录、审核与编码；Python 组织批量前向与学习。功能验收的 `ModelConfig.tiny()` 有151938个可训练参数，不是固定 TinyPolicy；去掉 `init --tiny` 使用原默认模型。

最小主线是模型采样 → 已执行动作监督及真实终局价值 → 训练/验证分片 → 更新 → 恢复 → 新权重采样。`sampled-action-imitation` 表示对已执行动作的监督，不是策略梯度、搜索访问分布或行为概率；回溯后不输出未经推导的路径概率。不宣称此闭环已获得有效强化学习或棋力提升。

TS 手工教师、restricted PUCT、旧记录转换和历史改善/连续实验仍是独立的 TS 工具，未迁移且不由本入口调用。搜索轨迹不能静默变成 one-hot。缺少新规则包哈希的旧权重继续由旧入口使用，新采样入口明确拒绝。

## 独立安装与运行

需要 Rust 1.98.1、链接器、Python 3.12，不需要 Node/npm/Bun/tsx/node_modules 或 TS 源码。源码包提供 `native/engine-prototype/{Cargo.toml,Cargo.lock,src,data}` 及 Python 包。CPU 安装：

```sh
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
.venv/bin/pip install ./training
cargo build --release --manifest-path native/engine-prototype/Cargo.toml --locked
```

Windows 使用 `.venv/Scripts/python.exe` 和 `.exe` 引擎。下文 `PYTHON` / `ENGINE` 是需替换的可执行文件路径。

```sh
PYTHON -m haojie_training.native --engine ENGINE init --tiny --checkpoint initial.pt
PYTHON -m haojie_training.native --engine ENGINE sample --checkpoint initial.pt --starts starts.json --output sample --commands 12000 --plies 500
PYTHON -m haojie_training.native --engine ENGINE audit sample/game-0.jsonl sample/game-1.jsonl
PYTHON -m haojie_training.native --engine ENGINE prepare sample/game-0.jsonl sample/game-1.jsonl --output data
PYTHON -m haojie_training.train --data data/train.pt --validation data/validation.pt --initialize-from initial.pt --checkpoint updated.pt --steps 2 --batch-size 4 --threads 1 --device cpu
PYTHON -m haojie_training.train --data data/train.pt --validation data/validation.pt --resume updated.pt --checkpoint restored.pt --steps 2 --batch-size 4 --threads 1 --device cpu
PYTHON -m haojie_training.native --engine ENGINE sample --checkpoint restored.pt --starts starts.json --output evaluation --commands 12000 --plies 500
```

`starts.json` 为1–8个环境，例如 `[{"seed":71,"rules":"classic"},{"seed":72,"rules":"shrine"}]`。可选 `prelude` 是 `{actor,command}` 数组，须从原种子按权限逐条重放；不接受任意内部状态。晚盘前缀只用于功能验收。截断、取消和未完成尾部没有价值标签；正式功能验收须实际取得非空终局监督。

## 版本与边界

| 契约           | 版本                                             |
| -------------- | ------------------------------------------------ |
| 规则           | `3.0-feedback5-live-deployment-2026-09-23`       |
| 提交规则包     | `haojie-rules-package-v1`                        |
| 记录           | `haojie-native-record-v1`                        |
| 常驻二进制接口 | `haojie-training-binary-v1`                      |
| 编码           | `haojie-entities-factorized-v1`                  |
| 模型 / 检查点  | `entity-transformer-v2` / `haojie-checkpoint-v1` |

- 单一权威仍是 TS 的 catalog/combat/recipes/pools/schema。开发时显式运行 `node --import tsx scripts/training/native/rules-package.ts` 生成提交包，`npm run train:rules:check` 检查过期。安装、缺缓存启动和新数据预处理不调用生成器。`data/rules.json` 固定字节 SHA256、规则/编码版本随 manifest 提交并嵌入二进制；Git 不转换规则包换行。
- `haojie-training-binary-v1` 使用 JSON 行控制，随后发送 LE f32 特征、LE i64 索引、u8 掩码。顺序是 entities/globals/candidates/kinds/sources/targets/entity_mask/candidate_mask，维度与旧模型一致。Python 复制为独立拥有张量，异步批处理不借用可被覆盖的内存。Rust 保存局面与递归上下文。
- 模型请求带递增 id 和检查点文件 SHA256；核对规则/完整编码/规则包、有限权重及 logits。错误/过期响应明确失败，不回退教师或随机策略。换权重创建新模型及环境，不复用 TinyPolicy 投影。
- `haojie-native-record-v1` 使用逐行 SHA256 链和执行前后状态哈希。规范编码：类型标签 null=0/bool=1/number=2/string=3/array=4/object=5；长度 u64 LE，字符串 UTF-8，对象按 UTF-8 键排序，数值有限 f64 LE、负零归零，布尔0/1，缺失不等于 null。哈希链计算 `{previous,body}` 的规范编码；旧记录和旧指纹不改写。
- 训练入口验证命令字段、范围、身份与权限。状态只来自原生新局及合法命令，权威 RNG/暗选不进入模型输入。原开发差分协议不作为不可信训练入口。
- 原始完整命令行写入 `.partial`；完成行写出、sync后使用不覆盖目标的同目录硬链接发布。完整JSON行损坏、校验失败及压缩损坏均报错。未换行尾部单列 `incompleteTail`，不视为已提交命令。新原始出口为未压缩 JSONL，不接受 gzip 冒充普通文本。
- 原生同实现审核不等于独立规则验证。开发/CI 保留 TS 规则、候选、编码、规范哈希和实际模型前向差分。

## 恢复与来源

检查点原子保存模型、AdamW、AMP、累计步骤/更新、CPU/设备随机源和来源。固定学习率；数据索引由种子和累计步骤确定。续训严格要求相同数据哈希、划分、损失与参数；行为模型哈希列表进入数据和检查点。训练/验证按规则和原始开局种子族隔离，拒绝重复游戏身份。身份由规则包、起点及命令前缀、模型、策略随机源和采样算法组成；改变命令或回合预算不能把旧前缀变成新数据。

支持最近完整学习阶段恢复和已记录成功前缀审核，**不支持从半局精确恢复采样递归栈**。重启采样须新输出路径，旧前缀保留为 unknown；合并相同游戏身份时拒绝重复。取消不自动结束回合。需要新采样阶段时显式选择种子与输出，不把半成品标作成功。

纯包验收脚本检查0→2→4更新、参数变化、有限优化器、另进程恢复与连续4步逐张量相等，再用恢复权重实际采样和审核：

```sh
python scripts/training/native/pipeline/package.py --starts <已验证前缀.json> --output <新源码包.tar.gz>
python scripts/training/native/pipeline/accept.py --engine ENGINE --starts fixtures/late-starts.json --output <新验收目录>
```

阶段状态和证据见[任务记录](../docs/ai/performance/native-training/README.md)。

## 并发与测量边界

每个环境持有一个Rust进程和最多一个在途模型请求。Python收到首个完成请求后合并已经就绪的节点，不等待其他环境的规则查询；相同环境的响应与后续请求仍串行。前向异常、版本错误或取消会关闭本批自己启动的进程，成功命令前缀保留。这里没有共享可变张量、跨权重缓存或新的Rust专用查询算法。

`scripts/training/native/pipeline/measure.py` 在独立Linux PID命名空间测量冻结检查点，`--parallel` 批量运行全部起点，省略时逐局单环境运行；`--prepare` 加入独立审核、再次审核并编码分片、加载、两步AdamW和全验证集损失评估。独立审核与准备中的审核都计费，不能把减少审核算作优化。输出包含工作量、文件哈希、阶段墙钟、累计CPU和100ms采样RSS；RSS不是连续精确峰值，Rust推理等待包含Python前向及传输，不能与前向耗时相加。

性能报告只适用于注明的检查点、尺寸、设备与环境并发数。小模型两步更新的占比不代表长期训练；并发切片也不代表自然开局或整机最大吞吐。
