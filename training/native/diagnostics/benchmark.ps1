param(
    [string]$Output = 'artifacts/native-benchmark',
    [int[]]$Concurrency = @(12),
    [int]$Repeats = 1,
    [int]$Batch = 32,
    [int]$InferenceBatch = 12,
    [int]$Chunk = 16,
    [int]$Publish = 8,
    [string]$Executable = '',
    [int]$Games = 16,
    [int]$GameSamples = 256,
    [string]$BehaviorReference = '',
    [string]$Device = 'cuda',
    [switch]$Shrine
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$outputRoot = [IO.Path]::GetFullPath($Output, $repoRoot)
if (Test-Path -LiteralPath $outputRoot) { throw '基准输出目录必须不存在。' }
if ($Games -le 0 -or $GameSamples -le 0 -or $Repeats -le 0 -or $Batch -le 0 -or $Chunk -le 0) { throw '工作量必须为正数。' }
New-Item -ItemType Directory -Path $outputRoot | Out-Null
$torchRoot = if ($env:LIBTORCH) { $env:LIBTORCH } else { [IO.File]::ReadAllText((Join-Path $repoRoot 'artifacts/cpp-training-build/libtorch-path.txt')).Trim() }
$env:PATH = "$(Join-Path $torchRoot 'lib');$(Join-Path $repoRoot 'artifacts/native-target/release');$env:PATH"
$exe = Join-Path $repoRoot 'artifacts/cpp-training-build/Release/haojie-train.exe'
if ($Executable) { $exe = (Resolve-Path -LiteralPath $Executable).Path }
$results = [Collections.Generic.List[object]]::new()
$reference = if ($BehaviorReference) { [IO.Path]::GetFullPath($BehaviorReference, $repoRoot) } else { $null }
$Samples = $Games * $GameSamples
$Publish = [Math]::Max($Publish, [int][Math]::Ceiling($Samples / [double]$Batch))
@{ executable = $exe; sha256 = (Get-FileHash -LiteralPath $exe).Hash.ToLowerInvariant(); external_snapshot = [bool]$Executable; source_hashes_scope = 'current-worktree'; publish = $Publish; batch = $Batch; inference_batch = $InferenceBatch } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputRoot 'executable.json') -Encoding utf8
for ($repeat = 1; $repeat -le $Repeats; $repeat++) {
    foreach ($count in $Concurrency) {
        $name = "concurrency-$count-repeat-$repeat"
        $directory = Join-Path $outputRoot $name
        $gpuPath = Join-Path $outputRoot "$name.gpu.csv"
        $gpuSampler = if ($Device.StartsWith('cuda')) {
            Start-Process -FilePath 'nvidia-smi.exe' -ArgumentList @('--query-gpu=timestamp,utilization.gpu,memory.used','--format=csv,noheader,nounits','-lms','200') -WindowStyle Hidden -PassThru -RedirectStandardOutput $gpuPath -RedirectStandardError (Join-Path $outputRoot "$name.gpu.stderr")
        } else { $null }
        $timer = [Diagnostics.Stopwatch]::StartNew()
        try {
            $runArguments = @('workset', '--directory', ('"' + $directory + '"'), '--device', $Device, '--concurrency', $count, '--workers', '8', '--games', $Games, '--game-samples', $GameSamples, '--batch', $Batch, '--inference-batch', $InferenceBatch, '--chunk', $Chunk, '--publish', $Publish, '--save-seconds', '0')
            if ($reference) { $runArguments += @('--replay', ('"' + $reference + '"')) }
            if ($Shrine) { $runArguments += '--shrine' }
            $process = Start-Process -FilePath $exe -ArgumentList $runArguments -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $outputRoot "$name.stdout") -RedirectStandardError (Join-Path $outputRoot "$name.stderr")
            $peakRss = 0L
            $lastCpu = $null
            while (-not $process.WaitForExit(50)) {
                $process.Refresh()
                if (-not $process.HasExited) {
                    $peakRss = [Math]::Max($peakRss, $process.WorkingSet64)
                    $lastCpu = $process.TotalProcessorTime.TotalSeconds
                }
            }
            $timer.Stop()
        } finally {
            if ($gpuSampler -and -not $gpuSampler.HasExited) { Stop-Process -Id $gpuSampler.Id; $gpuSampler.WaitForExit() }
        }
        if ($process.ExitCode -ne 0) { throw "基准失败（退出码 $($process.ExitCode)）：$(Get-Content (Join-Path $outputRoot "$name.stderr") -Raw)" }
        $window = Get-Content (Join-Path $directory 'window.json') -Raw | ConvertFrom-Json
        $worksetPath = Join-Path $directory 'workset'
        $workset = Get-Content (Join-Path $worksetPath 'index.json') -Raw | ConvertFrom-Json
        $actual = ($workset.games | Measure-Object -Property steps -Sum).Sum
        if (-not $workset.complete -or $workset.games.Count -ne $Games -or $window.consumed -ne $actual) { throw '固定对局工作集没有完整消费。' }
        if ($reference) {
            foreach ($original in Get-ChildItem -LiteralPath $reference -File) {
                $replayed = Join-Path $worksetPath $original.Name
                if ((Get-FileHash -LiteralPath $original.FullName).Hash -ne (Get-FileHash -LiteralPath $replayed).Hash) { throw "工作集轨迹或正式状态不同：$($original.Name)" }
            }
        } else { $reference = $worksetPath }
        $gpu = $null
        if ($gpuSampler) {
            $readings = @(Import-Csv -LiteralPath $gpuPath -Header Time, Utilization, Memory | ForEach-Object {
                [pscustomobject]@{
                    unix_ms = [DateTimeOffset]::ParseExact($_.Time.Trim(), 'yyyy/MM/dd HH:mm:ss.fff', [Globalization.CultureInfo]::InvariantCulture).ToUnixTimeMilliseconds()
                    utilization = [double]$_.Utilization
                    memory = [double]$_.Memory
                }
            })
            $inside = @($readings | Where-Object { $null -ne $window.window_start_unix_ms -and $_.unix_ms -ge $window.window_start_unix_ms -and $_.unix_ms -le ($window.window_start_unix_ms + 1000 * $window.seconds) })
            $gpu = @{
                process_memory_peak_mib = ($readings | Measure-Object memory -Maximum).Maximum
                window_samples = $inside.Count
                window_utilization_mean = if ($inside.Count) { ($inside | Measure-Object utilization -Average).Average } else { $null }
                window_memory_peak_mib = if ($inside.Count) { ($inside | Measure-Object memory -Maximum).Maximum } else { $null }
            }
        }
        $results.Add(@{
            name = $name
            concurrency = $count
            fixed_games = $Games
            window = $window
            wall = (Get-Content (Join-Path $directory 'wall.json') -Raw | ConvertFrom-Json)
            process_wall_seconds = $timer.Elapsed.TotalSeconds
            periodic_peak_rss_bytes = $peakRss
            last_observed_cpu_seconds = $lastCpu
            gpu = $gpu
        })
        $results | ConvertTo-Json -Depth 20 | Set-Content (Join-Path $outputRoot 'results.json') -Encoding utf8
        Write-Host "$name consumed=$($window.consumed) maximum=$Samples window=$($window.seconds) process=$($timer.Elapsed.TotalSeconds)"
    }
}
$hashes = @{}
foreach ($file in @(Get-ChildItem (Join-Path $PSScriptRoot '../src') -Recurse -File) + @(Get-ChildItem (Join-Path $repoRoot 'native/engine/src') -Recurse -Filter '*.rs' -File)) {
    $hashes[[IO.Path]::GetRelativePath($repoRoot, $file.FullName).Replace('\', '/')] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
}
$hashes | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $outputRoot 'source-hashes.json') -Encoding utf8
