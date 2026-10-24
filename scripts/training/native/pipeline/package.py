"""打包独立 C++/Rust 宿主源码、验证入口和规则数据，不收集本地产物。"""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    files = [
        *root.glob("training/native/src/*"),
        *root.glob("training/native/*.ps1"),
        *root.glob("training/native/*.sh"),
        root / "training/native/CMakeLists.txt",
        root / "training/native/.clang-format",
        *root.glob("training/native/**/*.md"),
        *root.glob("training/native/results/**/*.json"),
        *root.glob("training/native/results/**/*.csv"),
        root / "training/README.md",
        *root.glob("native/engine/src/**/*.rs"),
        *root.glob("native/engine/data/*"),
        root / "native/engine/Cargo.toml",
        root / "native/engine/Cargo.lock",
        root / "native/engine/build.rs",
        root / "native/engine/README.md",
        *root.glob("scripts/training/native/ci/*.py"),
        *root.glob("scripts/training/native/performance/*.py"),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.output.open("xb") as destination,
        tarfile.open(fileobj=destination, mode="w:gz") as archive,
    ):
        for path in sorted(files):
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
    manifest = {
        "format": "haojie-native-source-v1",
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "files": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
    }
    with args.output.with_suffix(".manifest.json").open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


if __name__ == "__main__":
    main()
