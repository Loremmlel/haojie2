# AI / 训练当前状态

2026-10-03补充：独立Rust＋Python实际模型闭环的安装、记录、编码、更新和安全阶段恢复见[新入口](../../training/NATIVE.md)及[分阶段任务记录](performance/native-training/README.md)。该入口已在无Node隔离包中完成真实更新/恢复；TS手工教师/PUCT及旧数据工具仍单独保留，不能当成这些搜索训练路线已完整迁移。共同就绪调度在实际小模型两环境自然局上三轮中位采样1.121×、含审核/准备/评估链1.063×；单环境无收益，未保留Rust专用算法分叉。按收益拐点收口，没有棋力结论或后续自动训练。下文9月29日记录仍为原性能基线背景。

核对基线：2026-09-29，本轮从 `db9beb0` 开始，规则仍为 `3.0-feedback5-live-deployment-2026-09-23`。本页是唯一当前状态入口；[训练路线](TRAINING.md)说明接口与研究背景，[历史进度](history/TRAINING-PROGRESS.md)和各实验报告保留当时的结论、数字、命令与版本，其中的“下一步”不是现行任务。此前整理提交只调整位置、导入和文档；本轮固定编码复用见[性能切片](performance/ENCODING-REUSE-2026-09-29.md)，不改变规则、选招、训练方法或模型。

## 位置与维护用途

| 类别             | 位置                                                                                                                                                                                        | 边界                                                                                                                                                              |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 生产代码         | `src/engine/`、`src/ai/`、`src/ui/`、`src/match/`、`src/index.ts`                                                                                                                           | 浏览器规则、本地 AI、单 HTML、公开联机接入；Rust 和网络模型未进入浏览器                                                                                           |
| 可复用基础设施   | `src/ai/training/`、`src/match/training.ts`、`scripts/training/records/`、`scripts/training/encode.ts`、`training/haojie_training/`、`native/engine-prototype/`、`scripts/training/native/` | 公开观察与动作、编码、记录读写/重放、Python 数据/模型/训练、完整 Rust 规则和跨语言差分；Rust 的目录名保留历史，代码仍在维护                                       |
| 当前可用研究入口 | `scripts/training/{self-play,benchmark,neural-match}.ts`、`scripts/training/performance/`、`scripts/training/native/`、`scripts/training/search/`                                           | 有界采样、对照和诊断；运行一次也可能产生新实验数据，按明确目标使用                                                                                                |
| 历史实验         | `scripts/training/improvement/`、`scripts/training/counterfactual/`、`scripts/training/search/{bootstrap,continuous,leaf,recovery,value-cycle,probes}/` 与 `docs/ai/experiments/`           | 阶段性协议仍可由当前代码尝试运行；能否重现原报告另由原输入、冻结源码及版本决定。共享的 `puct.ts`、`positions.ts`、`reference.ts` 仍留在 `search/`，不能随探针归档 |
| 本地产物         | `artifacts/`                                                                                                                                                                                | 原始轨迹、模型、优化器、报告、冻结源码、缓存和 Cargo 结果均不由 Git 跟踪；旧历史产物在本机缺失，本轮新产物保存在 `artifacts/training/sampling-20260929/`          |

同名工具有不同语义：`scripts/training/records/` 验证增量轨迹，`training/haojie_training/data.py` 读取已编码张量，Rust `training/` 负责原生动作树/编码/采样；这些不合并。连续实验和改善实验原先共用的分片写入实现现位于 `haojie_training.prepare`，两处都从该共享组件导入。空间研究模型原文件移至 `haojie_training/research/spatial.py`，由恢复和改善实验共用；参数和计算代码逐字未变。独立短程探针的 CLI 移至 `scripts/training/search/probes/`；旧报告的原始指纹只能在对应提交及输入上核验。

## 已证实到哪一步

| 状态                                     | 事实、提交与证据限制                                                                                                                                                                                                                                                                                                                                                                     |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 已进入生产                               | `3ab6ac9` 将 TS/Rust 同一状态分支算法接入 TS 生产引擎；网页仍用原本地束搜索 AI、规则和公开联机适配，单 HTML。提交时通过 404 项行为测试和离线浏览器验收；证据见[状态分支记录](performance/STATE-SHARING-2026-09-27.md)。这不是神经网络棋力证明。                                                                                                                                          |
| 已实现并作功能验证，性能或棋力未充分验证 | `9e5e930` 完成 Rust 规则，`859cb27` 完成原生动作树、编码与 TinyPolicy 连续采样；`3ab6ac9` 对齐双端分支。报告记录 1101 专项、32 份旧记录 71955 条命令，以及经典/神龛共 13270 条自然终局命令选招差分。证据见[Rust 记录](performance/RUST-PROTOTYPE.md)和[状态分支记录](performance/STATE-SHARING-2026-09-27.md)。历史计时不能拼接为当前完整训练倍率，Linux/Python 环境也未由这些报告验收。 |
| 仅实验原型                               | Python 策略/价值训练、网络 CLI、教师候选域搜索、连续自对弈和短程 PUCT 均有实现及阶段报告，未接入发行 AI；未验证对熟练作者胜率。浏览器 ONNX 只在本地 HTTP 做过前向试验，不等于离线发行模型。见[Python 入口](../../training/README.md)、[搜索实验](experiments/search/)、[历史进度](history/TRAINING-PROGRESS.md)。                                                                        |
| 已否定或暂停                             | [组合模型对照](experiments/search/COMPARISON-2026-09-26.md)未得到可靠棋力收益；[隔夜闭环](experiments/search/OVERNIGHT-2026-09-26.md)未晋级模型；[经济性评估](research/ECONOMICS-2026-09-26.md)未通过累计 30 天扩训依据。负结果及原始报告不改写。                                                                                                                                        |
| 尚未实施的建议                           | 完整神经网络＋搜索发行、生产模型批量推理、R9c 长局/并行/恢复经济性复核，以及新训练方案均未获本轮授权。可作为以后单独研究题目，不能从旧报告的“下一步”自动启动。                                                                                                                                                                                                                           |

## 当前可用的短命令

从仓库根目录运行。Node 22 与 `npm ci`、Rust 1.98.1 是对应工具前提；Python 训练需另装 PyTorch/项目依赖和原始数据。命令存在不代表历史产物在本机存在。

| 用途                | 命令                                                                                                                                                                              | 前提与边界                                                                                     |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| 静态与行为          | `npm run check && npm test && npm run check:structure`                                                                                                                            | 当前源码；行为测试不是完整棋力评测                                                             |
| 发行一致性          | `npm run build && npm run deploy:check`                                                                                                                                           | `dist/` 可重建；页面改动另运行 `npm run deploy`                                                |
| 公开环境最小交互    | `npm run --silent train:serve`                                                                                                                                                    | stdin/stdout JSONL，协议见[训练路线](TRAINING.md#常驻jsonl协议)                                |
| 增量记录审计        | `npm run train:inspect -- <实际存在的记录> --output <新报告.json>`                                                                                                                | 原始轨迹与对应规则版本；旧指纹不修补                                                           |
| Rust 编译/单测/静态 | `cargo test --manifest-path native/engine-prototype/Cargo.toml --locked`、`cargo clippy --manifest-path native/engine-prototype/Cargo.toml --all-targets --locked -- -D warnings` | Cargo 依赖缓存；跨语言差分另见[Rust 接口](../../native/engine-prototype/README.md)，需相应输入 |
| Python 组件测试     | `PYTHONPATH=training python -m unittest discover -s training/tests`                                                                                                               | `python` 为已装 PyTorch 的环境；无依赖时不能算通过                                             |
| 短程探针            | `node --import tsx scripts/training/search/probes/probe.ts --output <新目录>`                                                                                                     | 完整 30 夹具/480 决策，属于研究复跑；小型验证优先用 `tests/search/`                            |

历史实验分两类：A，当前版本仍支持启动，例如上述探针及 `search/continuous` 的冻结配置入口，但结果必须重新标注当前规则/源码，不等于原数值复现；B，需检出报告记录的历史提交并取得原始 `artifacts/` 或冻结 `source/` 后才能重现原哈希/胜负。旧实验产物在本机缺失，本轮新产物不替代旧证据。旧报告保留旧命令作为历史证据，移动映射见下文。

| 旧路径                                                                    | 新路径 / 复现方式                                                                                     |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| 根 `AGENTS.md` 的详细条款                                                 | `docs/maintenance/ENGINEERING-CONSTRAINTS.md`；根文件保留全局硬约束和必读索引                         |
| `docs/ai/TRAINING-PROGRESS.md`                                            | `docs/ai/history/TRAINING-PROGRESS.md`；其相对链接已改到原目标，数字未改                              |
| `scripts/training/search/{probe,audit,teacher}.ts`                        | `scripts/training/search/probes/{probe,audit,teacher}.ts`；现行新输出用新路径，旧源码指纹在原提交核验 |
| `scripts/training/search/continuous/dataset.py` 的 `digest`、`save_split` | 实现移至 `training/haojie_training/prepare.py`；旧实验自身其余逻辑仍在原处                            |
| `scripts/training/search/recovery/spatial.py`                             | `training/haojie_training/research/spatial.py`；旧检查点格式不变，旧源码哈希在原提交核验              |
| `native/engine-prototype/`                                                | 路径不变；名称不表示废弃，仍是双端算法对应实现与验证对象                                              |

## 2026-09-28 整理验收

- TypeScript：404 项行为测试、类型、格式和目录检查通过；当前规则 CLI 的 79 条命令指纹全部匹配。单 HTML 构建为 833324 字节，`deploy:check` 与根发行页逐字节一致。
- Python：本机另在 `/tmp` 安装 CPU 版 PyTorch 2.14.0 后，23 项测试通过，1 项 Windows 进程树测试按平台跳过。抽取的分片函数与基线 AST 相同，移动的空间模型与基线文件逐字节相同；可从训练包和原实验入口导入。未进行历史训练或模型棋力评测。
- Rust：release 构建、2 项单测、格式与 Clippy 通过；TS/Rust 规则夹具 1101 场景、原生动作树/编码 2202 根与 5376 节点通过。20 命令的经典/神龛小型采样在两端选招、终态和工作量一致；两局均截断，不能推导终局率或长期吞吐。
- 浏览器：实际离线 `file://` 生产页 35 项、Pages 嵌入及 AI/联机/图鉴/特效/神龛/反馈浏览器套件全部通过。所需 Chromium 共享库装在 `/tmp`，没有改变浏览器管理策略或发行 HTML。

以上是 9 月 28 日整理后的功能回归，当时未重跑历史训练、原始 32 份轨迹或完整 13270 命令终局；其旧结论仍以对应冻结提交和原始输入为准。当时工作区没有 `artifacts/` 原始产物，新生成的验证临时文件在 `/tmp`，不作为已保留的历史证据。9 月 29 日重新生成的长局及固定编码对照另见[性能切片](performance/ENCODING-REUSE-2026-09-29.md)。

## 2026-09-29 固定编码复用

从 `db9beb0` 冻结基线，只在 TS/Rust 连续采样中接入同一公开观察的固定编码复用；保留原候选、编码、回溯、策略随机消耗和规则。移动批量准备仍是未合入的历史实验，没有启动训练或第二个热点。完整依据与复现产物见[性能切片](performance/ENCODING-REUSE-2026-09-29.md)。

50 个含开局、中盘及高复杂度后段的固定局面，TS 三轮交错中位数 1.14×；Rust 自然长局三轮为经典 1.16×、神龛 1.24×，全部 13270 条命令及终态/观察/工作量一致。这些数字包含编码和 TinyPolicy 前向，仍不含正式记录指纹、压缩写盘、TS 兼容复核、训练或恢复，不能称作完整训练倍率。

新增隔离测试后 406 项行为测试通过；冻结旧版规则/候选/张量差分、当前双端 1101 规则夹具和 2202 根/5376 节点编码差分通过。类型、Rust 单测/Clippy、格式、结构、构建及发行一致性通过，根 HTML 无变化，离线 `file://` 浏览器 35 项通过。Python 本轮因缺少 PyTorch 无法导入测试模块，未声称功能验收通过。

当前 TS/Rust 又独立完成两局自然终局，正式压缩记录通过共享读取器的指纹/权限/标签审核。首次采样至最后证据冻结的保守总跨度 39.64 分钟（含中间回归和等待），未超过 60 分钟；本轮到此停止，不开启训练或第二个热点。

## 2026-09-29 内部运行架构

当前工作分支 `codex/sampling-runtime` 从 `6cd5bf4` 冻结旧TS和旧Rust，先完成TS（`3fcf888`、补充别名兼容 `efa29d0`），再迁移相同查询、分支提交、编码和TinyPolicy复用流程（`37f8e2b`）。完整取舍、失败试验、原始路径和复现命令见[任务记录](performance/runtime/README.md)。这是A类等价优化，没有改变规则、候选、预算、策略随机流或模型，也没有启动训练。

同机111固定局面各8命令，预热后八轮交错合并：旧/新TS为31.184/10.470秒（2.978倍），旧/新Rust为10.621/6.595秒（1.610倍）。包含观察、候选/回溯、同一TinyPolicy前向和真实结算；正式记录成本另计。TS没有达到3倍目标，不将历史倍率相乘或向上取整。外部apply及原预算浏览器AI对照有收益，选招一致。

最终TS通过410项行为测试、旧版1101夹具/31990候选差分；Rust通过1101规则夹具、2202根/5376节点和编码/网络差分，原生单测、Clippy与release构建通过。离线file://35场景、浏览器AI和联机13场景通过；Python23项组件测试使用本机已有环境通过。四条自然长局共28826命令在四端一致，正式记录逐步指纹一致；含写盘和独立复核的单次成本账TS为1443.315→575.965秒，Rust为887.459→435.755秒，后者含相应TS记录出口，不冒充纯原生加速。根HTML只在本地生成。

本轮按收益拐点收口：有价值部分已双端迁移，三项剩余结构假设未获得稳定净收益；TS的3倍目标未达成。代码及证据保留在独立分支和任务工作树，未合并、发布或启动训练。

本页只记已核对状态和入口，不追加新实验计划。以后改变状态时同步更新证据提交与验收限制。
