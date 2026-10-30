# 测试范围与维护

按实际依赖选择测试，不因“提交”就运行整仓。多模块改动取各范围的并集，同一检查只执行一次；测试、构建通过后，只有后续修改或失败涉及该范围才重跑。`npm test` 保留为主动全仓审计入口，日常使用下表的目标范围。

## 改动与验收

| 改动范围                                  | 运行什么                                                                                                                   | 无需默认运行                       |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| 文档、注释                                | 修改文件的格式、链接；增删/移动文件时 `check:structure`                                                                    | 行为测试、浏览器、训练             |
| Python 模型、目标、对手池                 | `training.tests.test_training`；模型张量/回放格式改变时增加 `test_storage`                                                 | 游戏 TS、Rust 引擎单测、游戏浏览器 |
| Python 训练调度、GPU 执行、发布、保存恢复 | `test_training`、`test_pipeline`、`test_storage`；诊断路径涉及 `test_profiling`                                            | 游戏 TS、Rust 规则差分、游戏浏览器 |
| Python 回放/存储                          | `test_storage`；影响学习或运行时恢复时增加 `test_training`、`test_pipeline`                                                | 无关游戏和 UI 测试                 |
| Web 控制协议与配置                        | `test_web`；配置也影响运行时则加对应训练测试                                                                               | 引擎测试、全量训练、游戏浏览器     |
| 训练 UI、图表、页面构建                   | `check`、`test:training-ui`、`training:web:check`、`training:web:views`                                                    | GPU 训练、Rust 构建、游戏测试      |
| TS 游戏规则、状态/存档或 AI               | 对应 `tests/core`、`ultimate`、`economics`、`shrine`、`kernel`、`session`、`ai` 文件；共享规则/算法改动加相关 TS/Rust 差分 | 训练 UI、无关 Python 训练          |
| Rust 规则或手工 AI                        | 对应 `cargo test`、`native:check`、`native:rules:check`、原生构建及相关规则/AI差分                                         | 无关游戏浏览器                     |
| Rust 神经搜索、编码、池/FFI               | Rust 对应单测、`test_pipeline`；编码/合法域改变加 TS/Rust 解码差分，学习张量改变加 `test_training`                         | 无关页面交互                       |
| Python 原生桥                             | `test_pipeline`；学习张量改变加 `test_training`，跨端协议/编码语义变化核对对应差分                                         | 未修改的 Rust 引擎单测、游戏浏览器 |
| 游戏 UI、交互、联机宿主适配               | 相关 TS 单测、`check`、`deploy:check` 和对应浏览器入口                                                                     | 训练后端、Rust 引擎测试            |
| 共用 tokens、打包入口、依赖               | 实际受影响的页面构建与浏览器入口；共用 tokens 同时覆盖游戏和训练台                                                         | 未受影响的规则矩阵                 |

Python 命令前缀为 `training/.venv/Scripts/python.exe -B -m unittest`，Linux 使用 `training/.venv/bin/python`。例如只改训练回放时：

```powershell
training/.venv/Scripts/python.exe -B -m unittest training.tests.test_storage
```

训练后端的完整回归为以下 4 个模块；Web 的 2 项轻量协议/配置测试单独运行：

```powershell
training/.venv/Scripts/python.exe -B -m unittest training.tests.test_training training.tests.test_pipeline training.tests.test_storage training.tests.test_profiling
training/.venv/Scripts/python.exe -B -m unittest training.tests.test_web
```

需要原生库时可先构建 `cargo build --locked --release --lib --manifest-path native/engine/Cargo.toml --target-dir artifacts/native-target`。这是准备训练依赖，不要求额外运行引擎测试。搜索差分从 Git 读取固定参考版本，不能用浅克隆缺失为由跳过恢复和搜索保护。

TS 单文件示例：`node --import tsx --test tests/session/recording.test.ts`。游戏全套用 `npm run test:game`，训练图表独立用 `npm run test:training-ui`。格式检查使用仓库的 `config/prettier.json`；Python 使用 Black，Rust 使用 cargo fmt；`modules.rs` 经 include 引入，另以 rustfmt 检查。

## 浏览器与构建

游戏主流程为 `test:browser`；图鉴使用 `test:browser:library`，AI/布局/节奏使用 `test:browser:ai`，神龛、特效、联机分别使用对应 `test:browser:*`。feedback3/feedback4 保护各自的选点、召唤与状态交互。只改图鉴时无需启动训练台，只改训练台时无需跑这些游戏入口。

`deploy:check` 本身会构建并核对发行 HTML，无需再先运行一次 `build`。页面源码变化后用 `deploy` 同步生成根 HTML；测试或文档改动无需重新生成页面。生产浏览器验收使用离线 `file://`；内存渲染只能报告渲染验收。

训练台 `training:web:views` 使用明确标记的合成观测，只验证图表、键盘、布局、草稿和重连。Python 可通过 `PYTHON` 环境变量指定，仅读取标准库 Config；`test_web` 仅需 psutil，不加载 PyTorch 或原生引擎。

`training:web:test` 是本机 CUDA 的真实生命周期验收，保留开始、刷新、保存停止、继续、扩缩容、断线和实际训练数据接入。只有修改这些跨层行为、GPU/设备实现或准备发布训练运行时才运行；纯样式或图表改动不启动它。性能基准和历史回放均显式按需执行，不属于日常提交门禁。

## CI 边界

- `ci.yml`：游戏源码/测试/构建发生变化时运行游戏检查和浏览器；不匹配 Python 训练或训练台文件。当前 79 命令回放及真实 CLI 入口由 arena 测试负责，流程不再重复运行一次 CLI 回放。
- `native.yml`：原生、TS 规则/AI、差分夹具及其依赖变化时，执行 Windows/Linux 原生和跨端验证；不笼统匹配 training、UI 或全部 tests。
- `training.yml`：训练后端及原生依赖变化时，双平台构建运行库并执行 4 个训练测试模块，覆盖模型、存储、搜索恢复和诊断；不运行引擎测试。
- `training-web.yml`：训练台、控制服务、配置和样式依赖变化时执行轻量控制、图表与视图检查，无 PyTorch/CUDA/Rust 安装。
- `history.yml`：只有手动触发时按记录原版本验证历史回放。

自动运行采用 PR 和 main push，避免同一功能分支同时触发 push/PR 两套。工作流可手动执行；共用依赖改动会触发所有受影响入口。CI 的范围比局部开发回归稍宽，正式规则、原生平台、存档和恢复边界保持保护。

## 删减原则与归属

只删已有等价保护的重复、无用或恒真检查。断言较少、测试名字相似、跨语言执行相同规则，都不是单独的删除理由；跨层测试应验证该层独有的接线、资源或失败行为。

本轮核实后删除两个旧 U13 用例、旧部署行用例和重复 96 步历史用例。U13 的实际卡牌种类检查合入 feedback-september17，旧 Session 往返合入 recording 的 79 步用例；实时部署由 feedback4/feedback5 保护。优化器恢复精度归 test_training，阶段屏障归 test_pipeline；并发重分配从 test_web 移到 test_pipeline。图鉴操作归 library 浏览器入口，图表交互归轻量 views。确定性、隐私、随机流、非法命令原子性、BF16、暂停和存储恢复测试保留。
