#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
: "${LIBTORCH:?Set LIBTORCH to an extracted LibTorch directory}"
cargo build --locked --release --lib --manifest-path "$repo_root/native/engine/Cargo.toml" --target-dir "$repo_root/artifacts/native-target"
cmake -S "$repo_root/training/native" -B "$repo_root/artifacts/cpp-training-build" -DLIBTORCH="$LIBTORCH" -DCMAKE_BUILD_TYPE=Release -DCUDA_RUNTIME_ROOT="${CUDA_RUNTIME_ROOT:-}" -DCUDA_CRT_ROOT="${CUDA_CRT_ROOT:-}" -DTORCH_CXX11_ABI="${TORCH_CXX11_ABI:-1}"
cmake --build "$repo_root/artifacts/cpp-training-build" -j "${BUILD_JOBS:-2}"
printf '%s\n' "$LIBTORCH" > "$repo_root/artifacts/cpp-training-build/libtorch-path.txt"
