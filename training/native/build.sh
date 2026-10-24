#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
: "${LIBTORCH:?Set LIBTORCH to an extracted LibTorch directory}"
cargo build --locked --release --lib --manifest-path "$repo_root/native/engine/Cargo.toml" --target-dir "$repo_root/artifacts/native-target"
cmake -S "$repo_root/training/native" -B "$repo_root/artifacts/cpp-training-build" -DLIBTORCH="$LIBTORCH" -DCMAKE_BUILD_TYPE=Release
cmake --build "$repo_root/artifacts/cpp-training-build" -j "${BUILD_JOBS:-2}"
