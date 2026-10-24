param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$buildPath = Join-Path $repoRoot 'artifacts/cpp-training-build'
$torchRoot = if ($env:LIBTORCH) { $env:LIBTORCH } else { [IO.File]::ReadAllText((Join-Path $buildPath 'libtorch-path.txt')).Trim() }
$env:PATH = "$(Join-Path $torchRoot 'lib');$(Join-Path $repoRoot 'artifacts/native-target/release');$env:PATH"
& (Join-Path $buildPath 'Release/haojie-train.exe') @Arguments
exit $LASTEXITCODE
