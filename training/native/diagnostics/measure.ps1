param(
    [Parameter(Mandatory)][string]$Output,
    [ValidateSet('actor', 'learner', 'mixed', 'inference', 'service')][string]$Mode = 'actor',
    [int[]]$Concurrency = @(1, 2, 4, 8, 16, 32, 64, 128),
    [int[]]$Batch = @(1000000),
    [string]$Checkpoint = '',
    [int]$Seconds = 20,
    [int]$Warmup = 5,
    [int]$Samples = 4096,
    [int]$TimeoutSeconds = 120,
    [int]$RootSlots = 2,
    [int]$InferenceBatch = 128,
    [int]$DeferLearning = 0,
    [string]$Executable = '',
    [string]$SourceArchive = '',
    [string]$BehaviorReference = '',
    [switch]$KeepCheckpoints
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$root = [IO.Path]::GetFullPath($Output, $repo)
if (Test-Path -LiteralPath $root) { throw '测量目录必须不存在。' }
New-Item -ItemType Directory -Path $root | Out-Null
$torch = [IO.File]::ReadAllText((Join-Path $repo 'artifacts/cpp-training-build/libtorch-path.txt')).Trim()
$env:PATH = "$torch/lib;$repo/artifacts/native-target/release;$env:PATH"
$exe = Join-Path $repo 'artifacts/cpp-training-build/Release/haojie-train.exe'
if ($Executable) { $exe = [IO.Path]::GetFullPath($Executable, $repo) }
$sources = @(Get-ChildItem (Join-Path $repo 'training/native/src') -Recurse -File | Get-FileHash | Select-Object Path, Hash)
if ($Executable) { $sources = @() }
@{ blas_preference = $env:TORCH_BLAS_PREFER_CUBLASLT; tunable_enabled = $env:PYTORCH_TUNABLEOP_ENABLED; tuning_enabled = $env:PYTORCH_TUNABLEOP_TUNING; tuning_file = $env:PYTORCH_TUNABLEOP_FILENAME } | ConvertTo-Json | Set-Content (Join-Path $root 'device-environment.json')
if ($BehaviorReference) {
    @{ path = $BehaviorReference; files = @(Get-ChildItem $BehaviorReference -File | Get-FileHash | Select-Object Path, Hash) } | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $root 'behavior-reference.json')
}
@{ executable = (Get-FileHash $exe).Hash; sources = $sources; source_archive_sha256 = if ($SourceArchive) { (Get-FileHash $SourceArchive).Hash } else { $null }; mode = $Mode; root_slots = $RootSlots; inference_batch = $InferenceBatch; defer_learning = $DeferLearning; allocator_config = $env:PYTORCH_ALLOC_CONF; seconds = $Seconds; warmup = $Warmup; samples = $Samples; checkpoint = $Checkpoint; checkpoint_sha256 = if ($Checkpoint) { (Get-FileHash $Checkpoint).Hash } else { $null } } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $root 'manifest.json')
foreach ($b in $Batch) {
    foreach ($c in $Concurrency) {
        $name = "c$c-b$b"
        $directory = Join-Path $root $name
        $gpu = Start-Process nvidia-smi.exe -ArgumentList @('--query-gpu=timestamp,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu', '--format=csv,noheader,nounits', '-lms', '200') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $root "$name.gpu.csv") -RedirectStandardError (Join-Path $root "$name.gpu.stderr")
        $arguments = @('diagnose', '--mode', $Mode, '--directory', ('"' + $directory + '"'), '--concurrency', $c, '--workers', 8, '--batch', $b, '--memory-mb', 16384, '--warmup', $Warmup, '--seconds', $Seconds, '--save-seconds', 0, '--seed', 1, '--shrine')
        $arguments += @('--root-slots', $RootSlots, '--inference-batch', $InferenceBatch,
                        '--defer-learning', $DeferLearning)
        if ($BehaviorReference) {
            if ($Mode -ne 'actor' -or $Warmup -ne 0 -or $Seconds -ne 0) { throw '固定 actor 工作集要求 actor、Warmup=0、Seconds=0。' }
            $arguments += @('--games', 16, '--game-samples', 256, '--replay', ('"' + $BehaviorReference + '"'))
        }
        if ($Checkpoint) { $arguments += @('--resume', ('"' + $Checkpoint + '"'), '--samples', $Samples) }
        $cpu = [Collections.Generic.List[object]]::new()
        $timer = [Diagnostics.Stopwatch]::StartNew()
        try {
            $p = Start-Process $exe -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $root "$name.stdout") -RedirectStandardError (Join-Path $root "$name.stderr")
            while (-not $p.WaitForExit(500)) {
                if ($timer.Elapsed.TotalSeconds -gt $TimeoutSeconds) {
                    Stop-Process -Id $p.Id
                    "诊断超过 $TimeoutSeconds 秒，主动停止；没有完整测量窗口。" | Set-Content (Join-Path $root "$name.timeout.txt")
                    $p.WaitForExit()
                    break
                }
                $p.Refresh()
                if (-not $p.HasExited) {
                    $frequency = Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation | Where-Object Name -eq '_Total'
                    $cpu.Add(@{ elapsed_s = $timer.Elapsed.TotalSeconds; cpu_s = $p.TotalProcessorTime.TotalSeconds; rss = $p.WorkingSet64; base_mhz = $frequency.ProcessorFrequency; performance_percent = $frequency.PercentProcessorPerformance; utility_percent = $frequency.PercentProcessorUtility })
                }
            }
        } finally {
            if (-not $gpu.HasExited) { Stop-Process -Id $gpu.Id; $gpu.WaitForExit() }
        }
        @{ exit_code = $p.ExitCode; wall_s = $timer.Elapsed.TotalSeconds; cpu = $cpu } | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $root "$name.host.json")
        if ($p.ExitCode -ne 0) { Write-Warning "$name 失败，原始 stderr 已保留。" }
        else { Get-Content (Join-Path $directory 'window.json') -Raw | Write-Output }
        # 运行目录由本轮新建；保留测量记录，仅删除明确不再使用的本轮检查点。
        $checkpointPath = Join-Path $directory 'checkpoint.pt'
        if (-not $KeepCheckpoints -and (Test-Path -LiteralPath $checkpointPath)) {
            Remove-Item -LiteralPath $checkpointPath
        }
    }
}
