# 训练记录精简计划

用户确认不需要旧训练JSONL兼容，之后重新生成。游戏存档兼容和历史规则回放不属于此次删除范围。

1. 教师、存档转训练和网络对战统一版本化训练轨迹：每局种子/模式/实际上限，逐步操作者、命令、前后指纹及必要统计，结尾真实结果；不逐步落盘Observation，也不在异常记录重复局面。
2. 一个流式读取入口验证版本、顺序、权限、指纹和终局，并在内存重建操作者Observation。编码、轨迹审计、预算审计和网络报告使用该入口，旧格式明确拒绝并提示重新生成。
3. 用Node内置gzip支持压缩JSONL读写；推荐压缩轨迹，编码张量经管道直接进入Python并保存现有.pt，不引入大数组JSON中间文件或新依赖。
4. 验证经典/神龛、交换席位、终局/截断/中断、错误指纹/版本/压缩数据、取消与背压；证明重建输入与决策未改变，实测长局字节数，跑TS和Python预处理闭环。
5. 更新训练文档和性能轨迹指标版本，保留历史报告/文件，不自动删除或改写它们。本轮不宣称所有多局集合都会小于1MB，也不以短局吞吐作为棋力结论。

## 实施结果

上述入口已统一到`scripts/training/records`。教师和存档转换支持`.jsonl.gz`，网络对战与性能采样默认输出gzip；旧快照格式明确拒绝。异常只保留索引、指纹和原因，缺少结束行的完整前缀统一标记`missing-outcome`。现有Python管道直接生成`.pt`，无需新增Python格式或依赖。

性能采样报告新增`trajectoryFormat: haojie-training-record-v1`和`compression: gzip`。摘要现在覆盖增量轨迹，不能直接与包含逐步Observation的历史`trajectorySha256`比较；计时也新增压缩成本。冻结的旧教师模块会被拒绝，应重新构建；重现历史测量须使用当时的脚本和代码，不改写历史报告。

真实预处理发现并修复了训练动作树的中立墓地过滤缺口：目标域过去直接比较保留的来源`owner`，会排除引擎认可的牵引。现在复用`allegiance`；两棋子回归用例先失败后通过，规则结算和教师选招未改。

## 体积测量与闭环

本机目录`artifacts/training/incremental-20260923/`，经典easy/40固定工作量，种子2026092307、2026092308，每局100个全局回合或600条命令。第一局405步真实终局，第二局600步显式截断。旧体积由同一命令轨迹重建每步Observation，按旧sample字段紧凑序列化计算；没有把它另存成冗余文件。新旧均保留同一教师统计与耗时，以下为字节数：

| 命令数 | 旧快照结构 | 增量JSONL | 增量gzip | 旧快照gzip对照 |
| ------ | ---------- | --------- | -------- | -------------- |
| 405    | 4,943,521  | 181,577   | 17,548   | 81,358         |
| 600    | 8,883,811  | 270,296   | 26,043   | 135,638        |

实际两局合并文件为451,873字节，gzip为43,021字节。压缩块跨局共享，所以合并值略小于分局值之和。收益主要来自不重复保存局面，gzip进一步压缩命令和统计；此测量不是任意模式/局长的大小上限。

验证结果：

- 364项Node测试、7项Python测试通过，覆盖经典/神龛、换边、终局/截断/中断、旧格式拒绝、错误指纹、损坏gzip和生成失败后的前缀恢复。
- 1,005条真实教师命令全部重放、审计并编码，得到1,831个分步样本；训练集600条决策无价值标签，验证集405条决策保留真实收益，种子族没有交叉。
- 真实`.pt`上完成CPU小模型4个样本、1步优化，参数/梯度有限；用该检查点完成换边各4命令的网络CLI烟雾验证，gzip与报告通过重放，无非法命令。另验证Python启动失败仍产生有效空gzip和失败报告。
- 预算审计从压缩轨迹选取1个位置，实际完成三个预算与重复决策检查；这不是棋力评估。
- 格式、TypeScript、目录结构、构建、`deploy:check`通过；发行HTML与本轮构建逐字节相同，无需更新页面产物。

复现主要入口（再次运行需更换输出目录）：

```powershell
node --import tsx scripts/training/self-play.ts --games 2 --seed 2026092307 --rules classic --difficulty easy --nodes 40 --plies 100 --commands 600 --output artifacts/training/incremental-20260923/teacher.jsonl.gz --report artifacts/training/incremental-20260923/teacher.json
node --import tsx scripts/training/inspect-teacher.ts artifacts/training/incremental-20260923/teacher.jsonl.gz --encode --output artifacts/training/incremental-20260923/inspection.json
training/.venv/Scripts/python.exe -X utf8 -m haojie_training.prepare artifacts/training/incremental-20260923/teacher.jsonl.gz --output artifacts/training/incremental-20260923/encoded
```
