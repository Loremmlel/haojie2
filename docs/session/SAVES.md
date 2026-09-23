# 增量存档与训练转换

网页导出、浏览器自动保存和CLI的JSON统一通过`serializeSession`写出`haojie-record-v1`，读取统一通过`parseSession`。这不是游戏规则或`GameState.version`的升级；浏览器仍使用`haojie.session.v2`，不删除旧档。

## 文件内容

- `ruleset`：引擎回放规则版本（`RULESET_ID`）。
- `origin`：`opening`表示初始局面与该种子、模式的标准开局完全一致；`position`表示从中途或演示局面开始。
- `initial`：记录起点的完整权威局面，包含正式随机状态。
- `commands`：从起点到当前局面的全部成功命令；不受60步缓存上限影响。
- `present`：当前完整局面，便于读取与终态核对。
- `match`：可选的同屏/人机设置。

不导出`past`、`future`或人类决策点快照。默认紧凑JSON，不加缩进。导入仍受24MB限制，单份记录最多10万条命令；超限明确报错，不截断记录。

## 导入、悔棋和兼容

运行时保留最近60步完整局面用于即时悔棋，并保存整条命令路线与当前游标。重做仅在当前会话有效；悔棋后另走截断后续命令，导出也只输出当前游标之前的路线。长AI回应额外保留最近人类决策点，载入新格式后仍可整段撤销和重做。

导入先验证规则版本、起点、命令结构，再顺序重放一次，核对全部当前局面字段（包括rng、战报和事件），同时重建最近60步及人类决策点。校验完成才替换棋局；不会播放历史事件。字段顺序不影响校验。导入耗时随命令数和局面复杂度增长，目前没有检查点；`present`并不表示跳过历史校验。

规则版本不同的增量文件明确拒绝，不自动用新规则重放；需保留对应规则代码读取旧文件。历史CLI回放文件和其固定规则提交不变。

旧`haojie-session-v2`仍可读取、执行既有迁移，并在当前会话使用其原有悔棋/重做快照。由于没有原命令，转存新格式以当时当前局面为起点，旧快照历史不写入新文件；界面明确提示不包含此前过程。在旧历史中撤销到记录起点之前，也会从已知局面重新起录，不虚构丢失命令。v1继续拒绝自动迁移。

浏览器容量不足时仍降级为当前局面存档并提示“历史请导出”；完整记录仍留在当前内存会话中。此时刷新后不能恢复未成功写入的历史。导出的是本地完整权威存档，不是可发送给对手的PlayerView；受控联机入口仍由宿主保存。

## 转为训练样本

```sh
npm run train:import-save -- artifacts/match.json artifacts/match-samples.jsonl
npm run train:inspect -- artifacts/match-samples.jsonl --encode --output artifacts/match-inspection.json
npx tsx scripts/training/encode.ts artifacts/match-samples.jsonl > artifacts/match-encoded.jsonl
```

转换先通过同版重放校验，再从起点依序执行，以实际操作者生成`observation + command`。暗选、对方反应与回合外巨大化不能一律使用active阵营。记录的权威种子和起点仅留在整局元数据，网络输入仍经原有Observation白名单与编码器。

输出沿用现有`game/sample/outcome` JSONL，标记`source: saved-game`，保留记录起点和真实胜负。未结束存档标记`interrupted`、`returns: null`，只能用于策略标签。不会伪造教师搜索分数、概率、耗时或工作量；审计以`saved-game:unrated`分类，并单列缺少搜索统计的样本。

训练头的`ruleset`沿用训练/公开协议的`HAOJIE_RULESET`，`recordRuleset`保留存档的`RULESET_ID`，二者现有命名不同，不替换旧训练数据版本。转换后的样本可以预先编码供反复训练，无需每轮重放原始存档；空间节省不代表棋力或训练吞吐提升。

输出文件独占创建，不覆盖已有数据。若后续编码发现尚未支持的动作会明确失败，不静默丢样本；保留原始存档便于修复编码器后重试。

PowerShell中可使用`npm.cmd`替代`npm`，确保`--encode`等参数经过`--`传给脚本。

训练包装基准现在使用`record-json`测量实际增量序列化；历史报告的`session-json`测量完整快照，不直接作为同名指标比较，也不改写旧报告。
