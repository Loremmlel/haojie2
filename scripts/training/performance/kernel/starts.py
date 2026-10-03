"""从本轮已冻结、双端核验的自然完整对局提取前缀，不恢复历史产物或修改胜负。"""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tail", type=int, default=20)
    parser.add_argument("--slices", action="store_true")
    args = parser.parse_args()
    if args.tail < 1:
        raise ValueError("尾部长度须为正")
    starts, sources = [], []
    for rules, seed in (("classic", 731270001), ("classic", 731270031),
                        ("shrine", 741270001), ("shrine", 741270037)):
        path = args.workset / f"{rules}-{seed}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        commands = payload["decisions"]
        positions = [0, len(commands) // 2] if args.slices else []
        positions.append(max(0, len(commands) - args.tail))
        for position in positions:
            starts.append({"seed": seed, "rules": rules, "prelude": [
                {"actor": p["actor"], "command": p["command"]} for p in commands[:position]
            ]})
        sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "commands": len(commands), "positions": positions})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(starts, file, separators=(",", ":"))
    with args.output.with_suffix(".sources.json").open("x", encoding="utf-8") as file:
        json.dump(sources, file, indent=2)
    print(json.dumps({"starts": len(starts), "source_commands": [s["commands"] for s in sources]}))


if __name__ == "__main__":
    main()
