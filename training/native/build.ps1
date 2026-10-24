param([string]$LibTorch = $env:LIBTORCH)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
if (-not $LibTorch) {
    $LibTorch = Join-Path $repoRoot 'training/.venv/Lib/site-packages/torch'
}
if (-not (Test-Path -LiteralPath (Join-Path $LibTorch 'include/torch/csrc/api/include/torch/torch.h'))) {
    throw '请通过 -LibTorch 或 LIBTORCH 指定 LibTorch 解压目录。'
}
$buildPath = Join-Path $repoRoot 'artifacts/cpp-training-build'
& cargo build --locked --release --lib --manifest-path (Join-Path $repoRoot 'native/engine/Cargo.toml') --target-dir (Join-Path $repoRoot 'artifacts/native-target')
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& cmake -S $PSScriptRoot -B $buildPath "-DLIBTORCH=$LibTorch"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& cmake --build $buildPath --config Release -j 3
if ($LASTEXITCODE) { exit $LASTEXITCODE }
[IO.File]::WriteAllText((Join-Path $buildPath 'libtorch-path.txt'), (Resolve-Path $LibTorch).Path, [Text.UTF8Encoding]::new($false))
