# 运行内核复现入口

从仓库根目录执行。Windows 使用 PowerShell 7；每组输出必须是不存在的新目录，脚本拒绝覆盖。正式计时期间只运行一组测量，不并行编译、测试或训练。默认 release 不带 `kernel-profile`。完整原始证据在本工作树 `artifacts/kernel-20261003/`；这些忽略产物不是可删除缓存。

## 重新冻结程序

当前任务保留的冻结入口为 `baseline/api.mjs`、`baseline/target/release/haojie-engine-prototype.exe` 与 `delivery/{api.mjs,engine.exe,manifest.json}`。新实验可以直接复用这些文件。若需要从 Git 重建，以下流程只向新目录提取旧 Rust 源码；TS 冻结器直接读取旧 Git 对象，不能间接导入新引擎。

```powershell
$bench = 'artifacts/kernel-recheck'
New-Item -ItemType Directory $bench, "$bench/baseline"
git archive --format=tar --output="$bench/baseline.tar" d6df36f975fbddf785b318d7a280e266ccd1cfe0 native/engine-prototype
tar -xf "$bench/baseline.tar" -C "$bench/baseline"
cargo build --locked --release --manifest-path "$bench/baseline/native/engine-prototype/Cargo.toml" --target-dir "$bench/old-target"
cargo build --locked --release --manifest-path native/engine-prototype/Cargo.toml --target-dir "$bench/new-target"
node --import tsx scripts/training/performance/kernel/freeze.ts --ref d6df36f975fbddf785b318d7a280e266ccd1cfe0 --executable "$bench/old-target/release/haojie-engine-prototype.exe" --output "$bench/old"
node --import tsx scripts/training/performance/kernel/freeze.ts --executable "$bench/new-target/release/haojie-engine-prototype.exe" --output "$bench/new"
```

冻结器记录全部 TS 依赖、Rust 源文件和规则数据的 SHA256、硬件、Node、Git 基点及程序哈希。尚未提交时，manifest 的 `head` 是基点，须同时查看源码哈希，不能把它误认成改动后的提交。

## 工作集与语义差分

固定四个种子 `classic:731270001/731270031`、`shrine:741270001/741270037`。冻结旧 Rust 先独立完成自然对局，冻结旧 TS 逐条重放核对；取预定时间点、最大密度点和规则夹具，产生 111 个工作局面。新种子在实现前已经固定。

```powershell
node --import tsx scripts/training/performance/kernel/workset.ts --baseline "$bench/old/api.mjs" --executable "$bench/old/engine.exe" --output "$bench/workset"
node --import tsx scripts/training/performance/kernel/differential.ts --baseline "$bench/old/api.mjs" --candidate "$bench/new/api.mjs" --output "$bench/frozen-differential"
node --import tsx scripts/training/native/validate.ts --executable "$bench/new/engine.exe" --fixtures-only --output "$bench/rules"
node --import tsx scripts/training/native/sampling/validate.ts --executable "$bench/new/engine.exe" --output "$bench/encoding"
node --import tsx scripts/training/performance/kernel/complete.ts --api "$bench/new/api.mjs" --executable "$bench/new/engine.exe" --workset "$bench/workset" --output "$bench/complete"
```

`complete.ts` 让 TS 与 Rust 各自构造动作树、编码、采样到自然终局，然后比较冻结命令、权威终态和双方观察。不是仅执行预给定命令的性能基准。全规则差分保护非法失败、连锁结果、候选顺序、存在掩码及编码；`tests/kernel/position.test.ts` 另保护分支隔离、同长度替换、重排、缺失身份与几何边界。

## A 层：引擎工作

```powershell
node --import tsx scripts/training/performance/runtime/measure.ts --baseline "$bench/old/api.mjs" --candidate "$bench/new/api.mjs" --workset "$bench/workset/workset.json" --commands 8 --rounds 5 --output "$bench/ts-accepted-paired"
node --import tsx scripts/training/performance/runtime/native.ts --baseline "$bench/old/engine.exe" --candidate "$bench/new/engine.exe" --api "$bench/old/api.mjs" --workset "$bench/workset/workset.json" --commands 8 --rounds 5 --output "$bench/rust-accepted-paired"
```

两组依次执行，各自预热后交错五轮。每轮 111×8=888 条连续命令，保留真实动作树、精确预检、结算、观察及编码，固定 TinyPolicy 前向另列。`commands` 模式不做记录哈希；哈希在 B 层实测。TS 的局面导入在计时外；Rust 内部计时含一次初始导入，`requestMs` 另含外部 JSON 请求、完整终态及双方观察输出。因此只比较同语言旧/新，不能用两张表的绝对值计算跨语言倍率。

## B 层：同一实际模型与完整链

`PYTHON` 替换为已安装训练包的 Python 3.12 可执行文件。本机使用 `E:/Code/haojie2/training/.venv/Scripts/python.exe`。`PYTHONPATH` 指向当前工作树，避免环境中的旧 editable 安装覆盖新代码。

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'training'
PYTHON scripts/training/performance/kernel/starts.py --workset "$bench/workset" --slices --output "$bench/mixed-starts.json"
PYTHON -m haojie_training.native --engine "$bench/new/engine.exe" init --tiny --seed 20261003 --checkpoint "$bench/model-initial.pt"
PYTHON scripts/training/performance/kernel/application.py --baseline "$bench/old/engine.exe" --candidate "$bench/new/engine.exe" --checkpoint "$bench/model-initial.pt" --starts "$bench/mixed-starts.json" --commands 128 --concurrency 4 --rounds 3 --output "$bench/application-accepted-paired"
```

同一实际 PolicyValueNet、151,938 参数、CPU 单线程、并发 4。12 个开/中/晚盘前缀，128 命令/60 ply 预算；先分别预热，再交错三轮。包括真实前向、记录、独立审核、准备时再次审核和编码、模型/分片加载、两步 AdamW、全验证集评估。未知尾部不造终局标签。逐行规范哈希相等才通过；原始 JSON 字段书写顺序不是规则语义。严格复用本次权重时使用 `artifacts/kernel-20261003/model-initial.pt`；重新初始化应记录自己的检查点文件哈希。

## 独立成本探针与无 Node 功能闭环

```powershell
cargo build --locked --release --features kernel-profile --manifest-path native/engine-prototype/Cargo.toml --target-dir "$bench/profile-target"
node --import tsx scripts/training/performance/kernel/profile.ts --executable "$bench/profile-target/release/haojie-engine-prototype.exe" --workset "$bench/workset/workset.json" --output "$bench/rust-separated-cost-profile"
PYTHON scripts/training/performance/kernel/application.py --baseline "$bench/old/engine.exe" --candidate "$bench/profile-target/release/haojie-engine-prototype.exe" --checkpoint "$bench/model-initial.pt" --starts "$bench/mixed-starts.json" --commands 128 --concurrency 4 --profile --output "$bench/application-separated-cost-profile"
node --import tsx scripts/training/performance/kernel/freeze.ts --executable "$bench/new/engine.exe" --output "$bench/delivery"
node --import tsx scripts/training/performance/kernel/report.ts --root $bench --output "$bench/results.json"
PYTHON scripts/training/performance/kernel/starts.py --workset "$bench/workset" --output "$bench/late-starts.json"
PYTHON scripts/training/native/pipeline/accept.py --engine "$bench/new/engine.exe" --starts "$bench/late-starts.json" --output "$bench/no-node-acceptance"
```

最后一条在 PATH 不含 Node 的独立 shell 中执行，并先确认 `Get-Command node -ErrorAction SilentlyContinue` 无结果；使用 Python 和引擎的绝对路径。仅修改该子进程环境，不删除 Node 安装。功能验收检查 0→2→4 更新、恢复等于连续四步、新权重再采样及审核。安装入口仍见 [training/NATIVE.md](../../../../training/NATIVE.md)，这些 TS 开发差分工具不是训练运行依赖。

工程收口执行 `npm run format:check`、`npm run check`、`npm test`、`npm run build`、`npm run deploy:check`、`npm run check:structure`、`npm run train:rules:check`、Python 全测试、Cargo fmt/test/clippy，以及 package.json 的离线浏览器验收入口。页面源码变化执行 `npm run deploy` 更新本地根 `index.html`，该命令不发布远端。
