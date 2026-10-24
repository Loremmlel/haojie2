param([string]$Output = 'artifacts/native-rewrite/control', [string]$Device = 'cuda')
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$targetRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot $Output))
if (Test-Path -LiteralPath $targetRoot) { throw '验收输出目录必须不存在。' }
New-Item -ItemType Directory -Path $targetRoot | Out-Null
$env:LIBTORCH = if ($env:LIBTORCH) { $env:LIBTORCH } else { [IO.File]::ReadAllText((Join-Path $repoRoot 'artifacts/cpp-training-build/libtorch-path.txt')).Trim() }
$env:PATH = "$(Join-Path $env:LIBTORCH 'lib');$(Join-Path $repoRoot 'artifacts/native-target/release');$env:PATH"
$exe = Join-Path $repoRoot 'artifacts/cpp-training-build/Release/haojie-train.exe'
& $exe check --device $Device
if ($LASTEXITCODE) { throw '数值和原生状态检查失败。' }
$live = Join-Path $targetRoot 'live'
$process = Start-Process -FilePath $exe -ArgumentList @('run','--directory',$live,'--device',$Device,'--concurrency','16','--workers','4','--batch','128','--chunk','16','--save-seconds','0','--seconds','60') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $targetRoot 'live.stdout') -RedirectStandardError (Join-Path $targetRoot 'live.stderr')
$readings = [Collections.Generic.List[object]]::new()
function Wait-State([scriptblock]$Predicate) {
    $timer = [Diagnostics.Stopwatch]::StartNew()
    while ($timer.Elapsed.TotalSeconds -lt 30) {
        if ($process.HasExited) { throw "训练提前退出：$(Get-Content (Join-Path $targetRoot 'live.stderr') -Raw)" }
        $statusPath = Join-Path $live 'status.json'
        if (Test-Path -LiteralPath $statusPath) {
            $status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
            if (& $Predicate $status) { $readings.Add($status); return $status }
        }
        Start-Sleep -Milliseconds 100
    }
    throw '等待训练状态超过 30 秒。'
}
try {
    $null = Wait-State { param($s) $s.active -eq 16 -and $s.consumed -ge 128 }
    & $exe resize --directory $live --concurrency 1
    $null = Wait-State { param($s) $s.active -eq 1 -and $s.parked -eq 15 }
    & $exe resize --directory $live --concurrency 0
    $null = Wait-State { param($s) $s.active -eq 0 -and $s.parked -eq 16 }
    & $exe save --directory $live
    $null = Wait-State { param($s) (Test-Path -LiteralPath (Join-Path $live 'checkpoint.pt')) }
    & $exe resize --directory $live --concurrency 16
    $null = Wait-State { param($s) $s.active -eq 16 -and $s.parked -eq 0 }
    & $exe stop --directory $live
    if (-not $process.WaitForExit(30000) -or $process.ExitCode -ne 0) { throw '停止保存失败。' }
    $saved = Get-Content -LiteralPath (Join-Path $live 'saved.json') -Raw | ConvertFrom-Json
    if ($saved.admitted_total -ne (16 + $saved.completed_games)) { throw '缩放重开了在途对局。' }
    $restored = Join-Path $targetRoot 'restored'
    & $exe run --directory $restored --resume (Join-Path $live 'checkpoint.pt') --device $Device --concurrency 16 --workers 4 --samples 128 --save-seconds 0
    if ($LASTEXITCODE) { throw '恢复后的实际训练失败。' }
    $window = Get-Content -LiteralPath (Join-Path $restored 'window.json') -Raw | ConvertFrom-Json
    $continued = Get-Content -LiteralPath (Join-Path $restored 'saved.json') -Raw | ConvertFrom-Json
    if ($window.consumed -ne 128 -or $continued.consumed -lt ($saved.consumed + 128)) { throw '恢复消费或窗口计数错误。' }
    if ($null -ne $continued.consumed_per_second) { throw '累计保存状态不得混入本轮速率。' }
    if ($continued.admitted_total -ne ($saved.admitted_total + $continued.completed_games - $saved.completed_games)) { throw '恢复重开了在途对局。' }
    @{ readings = $readings; saved = $saved; restored = $window } | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $targetRoot 'validation.json') -Encoding utf8
} finally {
    if (-not $process.HasExited) { & $exe stop --directory $live; $null = $process.WaitForExit(30000) }
}
