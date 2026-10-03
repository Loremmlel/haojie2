"""只打包训练所需 Python/Rust 源码与已提交规则数据；不收集 node_modules、TS 或本地产物。"""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--starts", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    files = [
        *root.glob("training/haojie_training/**/*.py"),
        root / "training/pyproject.toml",
        *root.glob("native/engine-prototype/src/**/*.rs"),
        *root.glob("native/engine-prototype/data/*.json"),
        root / "native/engine-prototype/Cargo.toml",
        root / "native/engine-prototype/Cargo.lock",
        root / "training/tests/test_native.py",
        Path(__file__).with_name("accept.py"),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.output.open("xb") as destination,
        tarfile.open(fileobj=destination, mode="w:gz") as archive,
    ):
        for path in sorted(files):
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
        archive.add(args.starts, arcname="fixtures/late-starts.json", recursive=False)
    manifest = {
        "format": "haojie-native-source-v1",
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "files": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
        "starts_sha256": hashlib.sha256(args.starts.read_bytes()).hexdigest(),
    }
    with args.output.with_suffix(".manifest.json").open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


if __name__ == "__main__":
    main()
