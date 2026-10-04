"""只打包训练所需 Python/Rust 源码与已提交规则数据；不收集 node_modules、TS 或本地产物。"""

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--starts", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    starts = args.starts or root / "tests/fixtures/native/late-starts.json.gz"
    starts_bytes = (
        gzip.decompress(starts.read_bytes())
        if starts.suffix == ".gz"
        else starts.read_bytes()
    )
    json.loads(starts_bytes)
    files = [
        *root.glob("training/haojie_training/**/*.py"),
        *root.glob("training/haojie_training/console/assets/*"),
        root / "training/pyproject.toml",
        root / "training/NATIVE.md",
        root / "training/CONSOLE.md",
        *root.glob("native/engine/src/**/*.rs"),
        *root.glob("native/engine/data/*"),
        root / "native/engine/Cargo.toml",
        root / "native/engine/Cargo.lock",
        root / "native/engine/build.rs",
        root / "native/engine/README.md",
        *root.glob("training/tests/test_native*.py"),
        root / "training/tests/test_resident.py",
        *root.glob("training/tests/console/*.py"),
        Path(__file__).with_name("accept.py"),
        Path(__file__).with_name("measure.py"),
        *root.glob("scripts/training/native/ci/*.py"),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.output.open("xb") as destination,
        tarfile.open(fileobj=destination, mode="w:gz") as archive,
    ):
        for path in sorted(files):
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
        entry = tarfile.TarInfo("fixtures/late-starts.json")
        entry.size = len(starts_bytes)
        archive.addfile(entry, io.BytesIO(starts_bytes))
    manifest = {
        "format": "haojie-native-source-v1",
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "files": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
        "starts_sha256": hashlib.sha256(starts_bytes).hexdigest(),
    }
    with args.output.with_suffix(".manifest.json").open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


if __name__ == "__main__":
    main()
