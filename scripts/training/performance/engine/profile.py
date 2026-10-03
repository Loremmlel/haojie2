"""汇总独立内核探针的调用、独占时间和累计分配；不把探针时间当正式倍率。"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    totals = {}
    for case in data["rows"]:
        for row in case["profile"]["rows"]:
            total = totals.setdefault(row["name"], {"calls": 0, "selfMs": 0.0, "allocations": 0, "bytes": 0})
            for key, value in zip(total, (row["calls"], row["selfMs"], *row["selfAllocations"][:2])):
                total[key] += value
    print(json.dumps(dict(sorted(totals.items(), key=lambda row: -row[1]["selfMs"])), indent=2))


if __name__ == "__main__":
    main()
