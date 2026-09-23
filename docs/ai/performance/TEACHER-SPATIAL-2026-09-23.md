# 攻击距离场追加优化

上一轮几何优化已提交为 `80890de`。本轮以它为基线，规则仍为 `3.0-feedback5-live-deployment`，只修改AI内部的攻击距离场遍历，不修改引擎、预算或策略。

## 新热点与取舍

用上一轮冻结模块、固定种子2026092301的一对完整对局重新剖析，共633条命令。V8默认采样中，`attackField`自耗时约12.12%，`structuredClone`约10.91%，垃圾回收约6.78%，`hitDistance`约6.31%，`attackPath`约4.56%，`positionKey`约4.07%。这是本次短批次的线索，不能与9月22日单个61棋子局面的比例直接比较；带剖析的57.78秒不计入速度对照。

只保留一个局部优化：固定9×13棋盘的邻接表在模块初始化时从引擎`neighbors`生成一次，搜索队列使用格号；阻挡、阵营、穿透和旧占位排除仍按每个局面查询。邻接顺序、起始覆盖格顺序、距离赋值时机和终点处理均保持原样。武器穿透判断在单次查询中复用。

没有新增局面缓存、修改候选或裁剪预算。状态深拷贝仍承担输入不可变与模拟分支隔离，改写它需要更大验证范围；寻路完整路径保存、指纹序列化和多任务调度留作后续独立剖析，不为小收益扩大本轮改动。

## 单进程复测

沿用上一轮测量脚本及冻结模块，classic、hard/800对medium/320、同种子换边，每局上限40回合或600命令。执行顺序为优化、基线、基线、优化；每次独立进程，包含真实采样、JSONL写盘与轨迹摘要，不含模块加载。测量期间没有运行本任务的测试或构建。

| 实现           |   第一次 |   第二次 |         平均 |
| -------------- | -------: | -------: | -----------: |
| 80890de基线    | 61.234秒 | 58.313秒 | **59.773秒** |
| 追加距离场优化 | 49.100秒 | 45.696秒 | **47.398秒** |

本组任务追加减少耗时 **20.70%**，吞吐提高 **26.11%**。四次都是同两局633条命令，均真实终局，逐步观察、命令、计划统计及结果摘要均为 `c72ff5880fcc2184a2f2c5bfd5bf1e624f8c9e786ecbe38ddd680efd85fc51e7`。两局模拟次数仍分别为20300、9880。

未锁定频率，四次重复不是四组独立棋力样本。本轮没有重复四进程批次或小时级测试，因此不把20.70%叠加到上一轮四进程结果，也不据此预测40局长局需要多久。当前局部改动已取得明确收益，本轮在此停止。

[精确测量与剖析摘要](spatial-20260923.json)保存原始数值、模块摘要及运行时间。原始文件位于`artifacts/training/perf-round2-20260923/`。

## 复现

先创建输出目录 `artifacts/training/perf-round2-20260923/`。

```sh
# 基线使用上一轮生成的 optimized.mjs；也可在80890de源码下重新构建。
node --cpu-prof --cpu-prof-dir=artifacts/training/perf-round2-20260923 --cpu-prof-name=current.cpuprofile scripts/training/performance/sample.mjs --module artifacts/training/perf-20260923/optimized.mjs --output artifacts/training/perf-round2-20260923/profile.json
npx esbuild scripts/training/self-play.ts --bundle --platform=node --format=esm --outfile=artifacts/training/perf-round2-20260923/spatial.mjs
node scripts/training/performance/sample.mjs --module artifacts/training/perf-20260923/optimized.mjs --output artifacts/training/perf-round2-20260923/baseline-1.json
node scripts/training/performance/sample.mjs --module artifacts/training/perf-round2-20260923/spatial.mjs --output artifacts/training/perf-round2-20260923/spatial-1.json
```

重复时使用新的输出名，不覆盖旧实验。新增行为测试用引擎路径查询核对5616个逐格距离，覆盖双方、1×1/2×2、穿透、叠放、中立、友方占位、忽略旧占位、零射程与基地终点。自身覆盖格保持原先距离0的语义；未改规则路径对自身覆盖格的返回行为。

## 验收

344项行为测试、TypeScript、目录结构及全仓格式检查通过（格式使用 `--end-of-line auto` 兼容既有Windows行尾）。基础34项与AI浏览器流程以真实离线 `file://` 通过，覆盖Worker、协作回退、节奏、取消、悔棋与恢复。当前79条CLI指纹匹配。根 `index.html` 已通过 `npm run deploy` 更新，812782字节，与 `deploy:check` 新构建逐字节一致。
