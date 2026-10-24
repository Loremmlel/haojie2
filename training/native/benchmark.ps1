param(
    [string]$Output = 'artifacts/native-benchmark',
    [int]$Samples = 4096,
    [int[]]$Concurrency = @(64, 256),
    [int]$Repeats = 2,
    [int]$Batch = 512,
    [int]$Chunk = 16,
    [string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$outputRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot $Output))
if (Test-Path -LiteralPath $outputRoot) { throw '基准输出目录必须不存在。' }
if ($Samples -le 0 -or $Repeats -le 0 -or $Batch -le 0 -or $Chunk -le 0) { throw '工作量必须为正数。' }
New-Item -ItemType Directory -Path $outputRoot | Out-Null
$torchRoot = if ($env:LIBTORCH) { $env:LIBTORCH } else { [IO.File]::ReadAllText((Join-Path $repoRoot 'artifacts/cpp-training-build/libtorch-path.txt')).Trim() }
$env:PATH = "$(Join-Path $torchRoot 'lib');$(Join-Path $repoRoot 'artifacts/native-target/release');$env:PATH"
$exe = Join-Path $repoRoot 'artifacts/cpp-training-build/Release/haojie-train.exe'
$results = [Collections.Generic.List[object]]::new()
for ($repeat = 1; $repeat -le $Repeats; $repeat++) {
    foreach ($count in $Concurrency) {
        $name = "concurrency-$count-repeat-$repeat"
        $directory = Join-Path $outputRoot $name
        $timer = [Diagnostics.Stopwatch]::StartNew()
        $process = Start-Process -FilePath $exe -ArgumentList @('run', '--directory', ('"' + $directory + '"'), '--device', $Device, '--concurrency', $count, '--workers', '8', '--samples', $Samples, '--batch', $Batch, '--chunk', $Chunk, '--save-seconds', '0') -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $outputRoot "$name.stdout") -RedirectStandardError (Join-Path $outputRoot "$name.stderr")
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
        if ($process.ExitCode -ne 0) { throw "基准失败：$(Get-Content (Join-Path $outputRoot "$name.stderr") -Raw)" }
        $window = Get-Content (Join-Path $directory 'window.json') -Raw | ConvertFrom-Json
        if ($window.consumed -ne $Samples) { throw '实际消费量与预算不同。' }
        $results.Add(@{
            name = $name
            window = $window
            wall = (Get-Content (Join-Path $directory 'wall.json') -Raw | ConvertFrom-Json)
            process_wall_seconds = $timer.Elapsed.TotalSeconds
            periodic_peak_rss_bytes = $peakRss
            last_observed_cpu_seconds = $lastCpu
        })
        $results | ConvertTo-Json -Depth 20 | Set-Content (Join-Path $outputRoot 'results.json') -Encoding utf8
        Write-Host "$name consumed=$Samples window=$($window.seconds) process=$($timer.Elapsed.TotalSeconds)"
    }
}
$hashes = @{}
foreach ($file in @(Get-ChildItem (Join-Path $PSScriptRoot 'src') -File) + @(Get-ChildItem (Join-Path $repoRoot 'native/engine/src') -Recurse -Filter '*.rs' -File)) {
    $hashes[[IO.Path]::GetRelativePath($repoRoot, $file.FullName).Replace('\', '/')] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
}
$hashes | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $outputRoot 'source-hashes.json') -Encoding utf8
