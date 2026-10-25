#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
build_path="$repo_root/artifacts/cpp-training-build"
LIBTORCH="${LIBTORCH:-$(cat "$build_path/libtorch-path.txt")}"
export LD_LIBRARY_PATH="$LIBTORCH/lib:$repo_root/artifacts/native-target/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# 安装阶段可复用 Python 分发的库，运行阶段只加载原生共享库。
shopt -s nullglob
for library in "$LIBTORCH"/../nvidia/*/lib; do
  export LD_LIBRARY_PATH="$library:$LD_LIBRARY_PATH"
done
exec "$build_path/haojie-train" "$@"
