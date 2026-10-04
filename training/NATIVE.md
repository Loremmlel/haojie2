# 无 Node 的 Rust＋Python 训练入口

日常长期训练优先使用[本地训练控制台](CONSOLE.md)：内存样本、持续更新、暂停继续、统一配额。本文保留完整记录/严格复现研究模式，独立 `resident` 采样固定检查点，不自动升级为控制台数据路径。

复用 `entity-transformer-v2`、候选交叉熵和终局价值损失、AdamW、分片与检查点。Rust 负责环境、公开观察、参数树、执行、记录、审核与编码；Python 组织批量前向与学习。功能验收的 `ModelConfig.tiny()` 有151938个可训练参数，不是固定 TinyPolicy；去掉 `init --tiny` 使用原默认模型。

最小主线是模型采样 → 已执行动作监督及真实终局价值 → 训练/验证分片 → 更新 → 恢复 → 新权重采样。`sampled-action-imitation` 表示对已执行动作的监督，不是策略梯度、搜索访问分布或行为概率；回溯后不输出未经推导的路径概率。不宣称此闭环已获得有效强化学习或棋力提升。

TS 手工教师、restricted PUCT、旧记录转换和历史改善/连续实验仍是独立的 TS 工具，未迁移且不由本入口调用。搜索轨迹不能静默变成 one-hot。缺少新规则包哈希的旧权重继续由旧入口使用，新采样入口明确拒绝。

## 当前运行内核

2026-10-04 已接通 Python 执行优化：推理专用输入/策略头、拥有型帧视图、连续页锁定组批、直接就绪队列、分片长度索引、单批预取和 CUDA fused AdamW。模型、损失及原生协议不变。默认 11,695,874 参数模型的 CUDA 实测、数值误差、失败路线和复现入口见[执行链路报告](../docs/ai/performance/execution/README.md)。

采样使用 `--device cuda --precision fp32`、1 个 PyTorch 宿主线程，`--batch-wait-ms` 默认 0；旧 `sample` 保留最多 8 起点短测，新 `resident` 的环境数独立配置。32 环境的固定默认模型自然局实测见[常驻采样报告](../docs/ai/performance/resident/README.md)。CLI 的 `--device auto` 保留无加速器时 CPU 回退。训练推荐 `--device cuda --precision bf16 --threads 1`，`--optimizer auto` 在 CUDA FP32/BF16 使用融合 AdamW，FP16/CPU/XPU 使用 foreach。`--prefetch` 默认开启，`--no-prefetch` 可做参考对照；不改变样本权重或按累计步骤确定的随机源。

要求同设备逐位恢复时，新阶段使用 `--deterministic`；续训省略此开关会继承检查点的确定性和矩阵精度设置。非确定性 CUDA 的连续/恢复结果允许存在舍入差异。融合优化器检查点须在 CUDA 原执行模式恢复；`--optimizer auto` 恢复实际保存的优化器执行方式，显式不匹配会失败。训练报告另列加载、前后全量指标评估及保存耗时，不用两步烟雾测试代表更新吞吐。

环境内部使用类型化实体、命令、反应与时钟快照；公开只读视图直接进入动作树与编码，连续模拟不再经过 `Observation Value → State`。外部存档、观察、记录及控制消息仍遵守原协议。原生发送缓冲在同步写入完成后复用，Python 接收的张量仍独立拥有。现有二进制协议与就绪调度没有换版。

2026-10-03 的正式原生 CI 从源码交付包在独立 Linux 容器构建并安装，运行根不挂载宿主仓库、关闭网络、扫描整个文件系统并核对安装与运行的实际 execve；不是仅从 PATH 删除 Node。非空标签、0→2→4 更新、45 个参数张量变化、另进程恢复与连续四步逐张量相等及新权重采样/审核均通过。Windows/Linux 的规则、候选、编码、RNG、私有信息和规范哈希继续由 TS 独立参照验证。

基础表示对应见[运行内核](../docs/ai/performance/kernel/ARCHITECTURE.md)；当前优化、性能边界、原始证据与复现入口统一见[正式引擎任务记录](../docs/ai/performance/engine/README.md)。固定实体编码区、节点追加区及最终发送帧在同步消费完成后复用；类型化状态直接进入既有规范哈希，成功提交后的哈希才可用于下一条记录。模型、优化器、监督语义和训练安装依赖保持不变。

## 独立安装与运行

正式维护入口为 [haojie-engine](../native/engine/README.md)。需要 Rust 1.98.1、链接器、Python 3.12，不需要 Node/npm/Bun/tsx/node_modules 或 TS 源码。源码包提供 `native/engine/{Cargo.toml,Cargo.lock,build.rs,src,data}` 及 Python 包。CPU 安装：

```sh
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
.venv/bin/pip install ./training
cargo build --release --manifest-path native/engine/Cargo.toml --locked
```

Windows 使用 `.venv/Scripts/python.exe` 和 `native/engine/target/release/haojie-engine.exe`；Linux 的程序无 `.exe`。下文 `PYTHON` / `ENGINE` 是需替换的可执行文件路径。`ENGINE --version` 输出构建身份、规则/编码/记录/协议版本与能力清单；`--engine` 必须显式指定，原生测试同样要求 `HAOJIE_NATIVE`，缺引擎直接失败。

```sh
PYTHON -m haojie_training.native --engine ENGINE init --tiny --checkpoint initial.pt
PYTHON -m haojie_training.native --engine ENGINE sample --checkpoint initial.pt --starts starts.json --output sample --commands 12000 --plies 500
PYTHON -m haojie_training.native --engine ENGINE audit sample/game-0.jsonl sample/game-1.jsonl --prepare data
PYTHON -m haojie_training.train --data data/train.pt --validation data/validation.pt --initialize-from initial.pt --checkpoint updated.pt --steps 2 --batch-size 4 --threads 1 --device cpu
PYTHON -m haojie_training.train --data data/train.pt --validation data/validation.pt --resume updated.pt --checkpoint restored.pt --steps 2 --batch-size 4 --threads 1 --device cpu
PYTHON -m haojie_training.native --engine ENGINE sample --checkpoint restored.pt --starts starts.json --output evaluation --commands 12000 --plies 500
```

`starts.json` 为1–8个环境，例如 `[{"seed":71,"rules":"classic"},{"seed":72,"rules":"shrine"}]`。可选 `prelude` 是 `{actor,command}` 数组，须从原种子按权限逐条重放；不接受任意内部状态。晚盘前缀只用于功能验收。截断、取消和未完成尾部没有价值标签；正式功能验收须实际取得非空终局监督。

## 版本与边界

### 常驻工作池

```sh
PYTHON -m haojie_training.native --engine ENGINE --threads 1 resident --checkpoint initial.pt --output resident-new --environments 32 --target 64 --seconds 3600 --drain-seconds 120 --seed 2026100407 --rules mixed --commands 20000 --plies 1000 --device cuda --precision fp32
PYTHON -m haojie_training.native --engine ENGINE --threads 1 audit-pool --source resident-new --output resident-data-new --shard-size 32
```

`resident` 不读取 starts.json。`--tasks` 可限制总任务编号数量；`--target` 是累计真实终局目标，达到后停止接新任务，在途局仍可完成，因此可能略超目标。每个槽位独立补位，模型、推理缓冲、规则宿主和调度器持续复用。默认经典/神龛按任务编号各半，游戏种子和策略随机源不依赖并发或完成顺序。

在输出目录创建 `STOP` 文件或按一次 Ctrl+C 停止接新任务，最多用 `--drain-seconds` 排空且不超过原总预算；再次 Ctrl+C 强制退出，保留前缀。移除 STOP 后，用相同参数和输出目录追加 `--resume`。可调整环境数、任务/终局目标和本次运行时限；模型、实际引擎哈希、规则模式、主种子、精度和单局上限必须相同。半局会从原种子重跑，使用新尝试编号，旧前缀不覆盖；已完成身份跳过。账本出现规则/协议错误时拒绝恢复，不能靠换槽位或不断重开绕过错误。

SQLite 的 tasks/attempts/runs 保存任务清单、完成身份、下一编号所需状态、尝试记录、配置与哈希；独占进程锁防止同目录双写。完成文件发布与账本提交之间的崩溃通过原生审核补记。仅偶发 EOF/断管可重建槽位，每次运行最多两次；规则和协议错误立即中止。`telemetry-*.jsonl` 每约5秒记录有界窗口和累计量，`report-*` 或 `failure-*` 保存结束状态。错误退出留下的数据库 running 状态表示待恢复任务，不表示子进程仍存活。

`audit-pool` 串行审核全部尝试，每次最多32个样本（以上命令）和客户端双帧队列；消费不过来时读取线程背压，不积存整局张量。输出目录的 `progress.json` 原子更新累计计数及 `pending_attempts`，积压记录留在磁盘。旧尝试和错误局只审核，不重复进入训练；当前有效尝试逐片准备，只有原始字节身份及结束回执确认后才发布。未完成前缀只保留策略标签。每局 `shards.jsonl` 列出带哈希的独立 `.pt` 分片，兼容 `load_dataset`；集合由 `audits.jsonl` 索引，按种子族稳定划分训练/验证。这里没有把百万局合成一个常驻内存索引；训练器的跨集合取数不在本轮扩展。

采样算法标记为 `model-gumbel-backtracking-v2`：公开 `uncertain` 候选被权威执行明确拒绝后，沿当前 Gumbel 排序继续回溯，不重新抽样，不预读私有 RNG，不排除整个 uncertain 动作域。失败原子性保证旧状态和正式 RNG 不变；每次参数决策最多64次拒绝。拒绝尝试单独写入 `rejected` 行，原生审核重新验证公开路径、权限、真实拒绝与前后状态哈希，不生成已执行动作监督。4,096节点/256深度解码保护耗尽时记 `decode-budget` 截断并补位，绝不计完整局或补终局收益。权限、available 候选执行失败、未实现或协议错误仍中止。旧 v1 记录继续按原语义审核，历史文件不改写。TS 参照沿同一算法同步。

| 契约           | 版本                                             |
| -------------- | ------------------------------------------------ |
| 规则           | `3.0-feedback5-live-deployment-2026-09-23`       |
| 提交规则包     | `haojie-rules-package-v1`                        |
| 记录           | `haojie-native-record-v1`                        |
| 常驻二进制接口 | `haojie-training-binary-v1`                      |
| 编码           | `haojie-entities-factorized-v1`                  |
| 模型 / 检查点  | `entity-transformer-v2` / `haojie-checkpoint-v1` |

- 单一权威仍是 TS 的 catalog/combat/recipes/pools/schema。开发时显式运行 `node --import tsx scripts/training/native/rules-package.ts` 生成提交包，`npm run train:rules:check` 检查过期。安装、缺缓存启动和新数据预处理不调用生成器。`data/rules.json` 固定字节 SHA256、规则/编码版本随 manifest 提交并嵌入二进制；Git 不转换规则包换行。
- `haojie-training-binary-v1` 使用 JSON 行控制，随后发送 LE f32 特征、LE i64 索引、u8 掩码。顺序是 entities/globals/candidates/kinds/sources/targets/entity_mask/candidate_mask，维度与旧模型一致。Python 每帧拥有独立接收字节区，NumPy/Tensor 视图共同持有该帧，后续请求不覆盖；主机组批容量只在结果回传完成后复用。Rust 保存局面与递归上下文。
- 模型请求带递增 id 和检查点文件 SHA256；核对规则/完整编码/规则包、有限权重及 logits。错误/过期响应明确失败，不回退教师或随机策略。换权重创建新模型及环境，不复用 TinyPolicy 投影。
- `haojie-native-record-v1` 使用逐行 SHA256 链和执行前后状态哈希。规范编码：类型标签 null=0/bool=1/number=2/string=3/array=4/object=5；长度 u64 LE，字符串 UTF-8，对象按 UTF-8 键排序，数值有限 f64 LE、负零归零，布尔0/1，缺失不等于 null。哈希链计算 `{previous,body}` 的规范编码；旧记录和旧指纹不改写。
- 训练入口验证命令字段、范围、身份与权限。状态只来自原生新局及合法命令，权威 RNG/暗选不进入模型输入。原开发差分协议不作为不可信训练入口。
- 推荐 `audit --prepare`：一次原生重放完成校验与编码，准备阶段消费已经拥有的张量。单独 `audit` 继续可用；`prepare --output` 自行执行一次审核。先单独 audit 再 prepare 属于刻意执行两次完整审核的严格工作流，性能比较分别记账。
- 已审核结果只在当前可信 Python 进程内使用，不从外部文件反序列化为缓存凭据。完成回执绑定本次实际读取的全部原始字节（含未完成尾部）、规则包、完整编码 schema、审核器版本与源码身份；复用还检查实际二进制和当前记录内容。路径、mtime 或以前成功不能授权复用。输出来自同一份已确认张量，核验后不重新读取来源生成特征。
- 新单遍入口需要 `ENGINE --version` 包含 `auditor: haojie-native-audit-v1` 与 `audited-encoding` 能力。宿主和引擎一起升级；记录、规则、编码、模型和二进制传输协议没有换版。
- 原始完整命令行写入 `.partial`；完成行写出、sync后使用不覆盖目标的同目录硬链接发布。完整JSON行损坏、校验失败及压缩损坏均报错。未换行尾部单列 `incompleteTail`，不视为已提交命令。新原始出口为未压缩 JSONL，不接受 gzip 冒充普通文本。
- 原生同实现审核不等于独立规则验证。开发/CI 保留 TS 规则、候选、编码、规范哈希和实际模型前向差分。

## 恢复与来源

检查点原子保存模型、AdamW、AMP、累计步骤/更新、CPU/设备随机源和来源。固定学习率；数据索引由种子和累计步骤确定。续训严格要求相同数据哈希、划分、损失与参数；行为模型哈希列表进入数据和检查点。训练/验证按规则和原始开局种子族隔离，拒绝重复游戏身份。身份由规则包、起点及命令前缀、模型、策略随机源和采样算法组成；改变命令或回合预算不能把旧前缀变成新数据。

支持最近完整学习阶段恢复和已记录成功前缀审核，**不支持从半局精确恢复采样递归栈**。单批 `sample` 重启须新输出路径；常驻 `resident --resume` 在原目录以新尝试编号从原种子重跑，旧前缀保留为 unknown，完成身份不重复计数。取消不自动结束回合。新采样阶段显式选择种子与输出，不把半成品标作成功。

纯包验收脚本检查0→2→4更新、参数变化、有限优化器、另进程恢复与连续4步逐张量相等，再用恢复权重实际采样和审核：

```sh
python scripts/training/native/pipeline/package.py --starts <已验证前缀.json> --output <新源码包.tar.gz>
python scripts/training/native/pipeline/accept.py --engine ENGINE --starts fixtures/late-starts.json --output <新验收目录>
```

当前阶段状态和证据见[正式引擎任务记录](../docs/ai/performance/engine/README.md)。此前训练宿主接线的历史结果保留于[原生训练报告](../docs/ai/performance/native-training/README.md)。

## 并发与测量边界

每个环境持有一个Rust进程和最多一个在途模型请求。Python收到首个完成请求后合并已经就绪的节点，不等待其他环境的规则查询；相同环境的响应与后续请求仍串行。前向异常、版本错误或取消会关闭本批自己启动的进程，成功命令前缀保留。这里没有共享可变张量、跨权重缓存或新的Rust专用查询算法。

`scripts/training/native/pipeline/measure.py` 在独立 Linux PID 命名空间测量冻结检查点，`--parallel` 批量运行全部起点，省略时逐局单环境运行；`--prepare` 加入独立审核、再次审核并编码分片、加载、两步 AdamW 和全验证集损失评估。这一严格流程的两次审核都计费；推荐单遍流程另记账，不能把减少阶段算作等价内核加速。输出包含工作量、文件哈希、阶段墙钟、累计 CPU 和 100ms 采样 RSS；RSS 不是连续精确峰值，Rust 推理等待包含 Python 前向及传输，不能与前向耗时相加。本轮交错成对测量使用 `scripts/training/performance/kernel/application.py`，CPU/RSS 的具体边界见任务记录。

性能报告只适用于注明的检查点、尺寸、设备与环境并发数。小模型两步更新的占比不代表长期训练；并发切片也不代表自然开局或整机最大吞吐。
