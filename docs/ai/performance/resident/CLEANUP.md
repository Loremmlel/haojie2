# 2026-10-04 本地产物清理

用户在 PR 交付后授权清理本地文件，并建议移入回收站。PR 为 [#11](https://github.com/Loremmlel/haojie2/pull/11)，实现提交 `c91bbca` 的 Windows/Linux 原生、独立安装及 HTML 验证 CI 已通过。本次只调整产物保留状态及文档，不改变源码、模型、记录或实测结果。

## 清理范围与保留证据

在 `artifacts/resident-20261004/` 内检查绝对路径和重解析点后，使用 Windows `SendToRecycleBin` 移入 10 个目标，共 25,350 个文件、75,306,981,140 字节（约 75.31 GB / 70.14 GiB）：

- `audit/000000/`：首次尝试的派生训练分片，约 2.56 GB。
- `repaired/audit/000000/`：修复验收的派生训练分片，约 72.74 GB。
- 重复的 `baseline.tar`、三个复算检查摘要、三个复算控制台日志及 `summary-console.txt`。

回收前将每局 `report.json`、`shards.jsonl` 复制到 `retained-audit/` 下的对应路径；另保留经典任务12、神龛任务67各一份32样本分片，分别为 `retained-audit/classic-32.pt` 和 `shrine-32.pt`。这些副本共5,914,833字节，样本哈希匹配原分片索引。

原始自然局记录及失败前缀、SQLite账本和迁移备份、遥测、审核总索引、模型、冻结引擎、筛选数据、失败复现与正式报告均保留。清理前后737个保留文件SHA256一致；本轮产物目录现约308.54 MB。其他训练实验、优化器、依赖环境、编译目录、工作树归档及未提交工作未纳入清理。

## 回收站状态变化

初查 E 盘回收站有535项、约82.12 GB；实际执行前快照已变为119项、约57.45 GB。该变化发生在本轮回收动作之前，未确认来源。

为避免容量上限淘汰旧内容，只将 E 盘回收站 `MaxCapacity` 从99,722 MiB提高至204,800 MiB（200 GiB），其他盘不变。旧值和对应卷注册表路径保存在清理清单中；没有自动缩回上限，以免影响现存可恢复内容。没有调用清空回收站或永久删除。

2026-10-04 15:17（日本时间）执行完成时，10项均匹配回收站元数据，回收内容的文件数和字节数与清单相等，执行前119项仍存在。然而15:19再次复核时，这10项的回收元数据和内容均已消失；磁盘空闲空间约819.17 GB。无法确认是外部清空还是其他清理动作，**不能再承诺这些分片可从回收站恢复**。原始证据及保留小分片再次核对仍全部一致，完整数据可从原记录重新准备。

本地凭据在 `artifacts/maintenance/resident-cleanup-20261004/`：`manifest.json` 保存路径、容量设置、保留哈希及副本；`recycled.jsonl` 记录执行结果；`verified.json` 记录执行时的回收站映射；`postcheck.json` 保存后续消失状态。执行脚本为 `artifacts/maintenance/resident-cleanup.ps1`，有已执行目录保护，不应重复运行。

## 清理后的复现

采样启动、停止和恢复命令不变；账本、原记录、固定模型及实际引擎仍在原路径。`report.py` 使用账本、审核总索引和测量摘要，复算不依赖被移走的 `.pt`。`results.json` 的存储量及719.38 GB空闲空间保留为验收时快照，不改写为清理后的数值。

若需要完整训练分片，使用新的输出目录重新执行有界审核；本轮清理没有重新生成大数据或启动学习：

```powershell
$env:PYTHONPATH = 'training'
$env:PYTHONUTF8 = '1'
training/.venv/Scripts/python.exe -m haojie_training.native --engine artifacts/resident-20261004/engine-budget.exe --threads 1 audit-pool --source artifacts/resident-20261004/repaired/formal --output artifacts/resident-regenerated --shard-size 32
```

短更新成本脚本依赖完整分片；要重跑它必须先恢复或再生成分片。两份保留小样本仅供加载排错，不代表完整数据集或新的性能测量。

清理后重新运行报告脚本，成本字段与已提交结果一致；修改文档格式、`git diff --check` 和目录结构检查通过。`deploy:check` 仍复现已记录的根HTML换行差异，构建843,066字节；没有改写发行文件或部署。没有追加模型或长局测试。
