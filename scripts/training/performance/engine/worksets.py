"""合并预先固定的新旧局面，记录每份输入和输出的内容身份，不重写来源。"""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows, sources = [], []
    for path in args.inputs:
        raw = path.read_bytes()
        chunk = json.loads(raw)
        rows.extend(chunk)
        sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "cases": len(chunk)})
    data = json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as file:
        file.write(data)
    manifest = {"sources": sources, "cases": len(rows), "sha256": hashlib.sha256(data).hexdigest()}
    with args.output.with_suffix(".manifest.json").open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
