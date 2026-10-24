param([string]$LibTorch = $env:LIBTORCH, [string]$CudaRuntime = $env:CUDA_RUNTIME_ROOT, [string]$CudaCompilerHeaders = $env:CUDA_CRT_ROOT)
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
$runtimeOption = @()
if (-not $CudaRuntime -and (Test-Path -LiteralPath (Join-Path $LibTorch 'lib/cudart64_13.dll'))) {
    # Windows cu130 只下载官方 Runtime 与 CRT 头包，不安装或调用 nvcc。
    $dependencyPath = Join-Path $repoRoot 'artifacts/native-cuda-runtime-13'
    New-Item -ItemType Directory -Path $dependencyPath -Force | Out-Null
    $manifestPath = Join-Path $dependencyPath 'redistrib_13.0.2.json'
    if (-not (Test-Path -LiteralPath $manifestPath)) {
        Invoke-WebRequest -Uri 'https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.0.2.json' -OutFile $manifestPath
    }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $packages = @{}
    foreach ($packageName in @('cuda_cudart', 'cuda_crt')) {
        $package = $manifest.$packageName.'windows-x86_64'
        $archiveName = [IO.Path]::GetFileName($package.relative_path)
        $archivePath = Join-Path $dependencyPath $archiveName
        if (-not (Test-Path -LiteralPath $archivePath)) {
            Invoke-WebRequest -Uri ('https://developer.download.nvidia.com/compute/cuda/redist/' + $package.relative_path) -OutFile $archivePath
        }
        if ((Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $package.sha256) { throw "CUDA 官方包校验失败：$archivePath" }
        $packagePath = Join-Path $dependencyPath ([IO.Path]::GetFileNameWithoutExtension($archiveName))
        if (-not (Test-Path -LiteralPath $packagePath)) { Expand-Archive -LiteralPath $archivePath -DestinationPath $dependencyPath }
        $packages[$packageName] = $packagePath
    }
    $CudaRuntime = $packages.cuda_cudart
    $CudaCompilerHeaders = $packages.cuda_crt
}
if ($CudaRuntime) { $runtimeOption = @("-DCUDA_RUNTIME_ROOT=$CudaRuntime") }
if ($CudaCompilerHeaders) { $runtimeOption += "-DCUDA_CRT_ROOT=$CudaCompilerHeaders" }
& cmake -S $PSScriptRoot -B $buildPath "-DLIBTORCH=$LibTorch" @runtimeOption
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& cmake --build $buildPath --config Release -j 3
if ($LASTEXITCODE) { exit $LASTEXITCODE }
[IO.File]::WriteAllText((Join-Path $buildPath 'libtorch-path.txt'), (Resolve-Path $LibTorch).Path, [Text.UTF8Encoding]::new($false))
