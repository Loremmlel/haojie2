#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cache="/content/haojie-dependencies"
mkdir -p "$cache"
# 先检查实际分配的 GPU，避免在 T4/P100 上下载不适用的依赖。
python3 - <<'PY'
import subprocess
capability = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], text=True
).splitlines()[0]
if float(capability) < 8:
    raise SystemExit("需要原生 BF16 GPU（例如 L4/A100）；当前 T4/P100 不受支持。")
PY
# 复用 Colab 的 pip 向独立环境安装，不依赖系统额外提供 ensurepip。
python3 -m venv --without-pip "$cache/venv"
python3 -m pip --python "$cache/venv/bin/python" install --no-cache-dir 'torch==2.14.0' --index-url https://download.pytorch.org/whl/cu126
python3 -m pip --python "$cache/venv/bin/python" install --no-cache-dir 'nvidia-cuda-nvcc-cu12==12.6.85' 'cmake>=3.24'
export PATH="$cache/venv/bin:$HOME/.cargo/bin:$PATH"
if ! command -v rustup >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o "$cache/rustup-init.sh"
  sh "$cache/rustup-init.sh" -y --profile minimal --default-toolchain 1.98.1 --no-modify-path
  rm "$cache/rustup-init.sh"
else
  rustup toolchain install 1.98.1 --profile minimal
fi
export RUSTUP_TOOLCHAIN=1.98.1
export LIBTORCH="$(python -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')"
export TORCH_CXX11_ABI="$(python -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
export CUDA_RUNTIME_ROOT="$(dirname "$LIBTORCH")/nvidia/cuda_runtime"
export CUDA_CRT_ROOT="$(dirname "$LIBTORCH")/nvidia/cuda_nvcc"
# 限制编译并行度，给 Colab 的运行内存留出空间。
export BUILD_JOBS="${BUILD_JOBS:-2}"
bash "$repo_root/training/native/build.sh"
bash "$repo_root/training/native/run.sh" configure --device cuda
