"""将已冻结完整轨迹的成功命令前缀另存为验收起点；不修改原文件或补造局面。"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tail", type=int, default=20)
    args = parser.parse_args()
    starts = []
    for rules, seed in (
        ("classic", 731270001),
        ("classic", 731270017),
        ("shrine", 741270001),
        ("shrine", 741270019),
    ):
        path = args.workset / f"{rules}-{seed}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        commands = payload["decisions"]
        starts.append(
            {
                "seed": seed,
                "rules": rules,
                "prelude": [
                    {"actor": p["actor"], "command": p["command"]}
                    for p in commands[: -args.tail]
                ],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(starts, file, separators=(",", ":"))


if __name__ == "__main__":
    main()
