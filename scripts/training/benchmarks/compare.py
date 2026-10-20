"""完整周期对照：先核对任务、样本、预算和更新量，再报告墙钟及吞吐。"""

import argparse
import json
import statistics
from pathlib import Path


def compare(before, after):
    if not before["complete"] or not after["complete"]:
        raise ValueError("未完成的测量不能进入吞吐对照")
    for key in ("torch", "python", "parameters", "config", "starts_sha256"):
        if before[key] != after[key]:
            raise ValueError(f"对照条件不同：{key}")
    for key in ("start_indices", "rounds", "ordered_updates", "threads", "device"):
        if before["options"][key] != after["options"][key]:
            raise ValueError(f"对照选项不同：{key}")
    if len(before["rounds"]) != len(after["rounds"]):
        raise ValueError("完成的训练轮数不同")
    results = []
    for a, b in zip(before["rounds"], after["rounds"]):
        for key in ("samples_hash", "counts", "updates", "samples_consumed", "tasks"):
            if a[key] != b[key]:
                raise ValueError(f"第 {a['round']} 轮负载不同：{key}")
        results.append(
            {
                "before_seconds": a["cycle_seconds"],
                "after_seconds": b["cycle_seconds"],
                "time_reduction_percent": 100 * (1 - b["cycle_seconds"] / a["cycle_seconds"]),
                "throughput_gain_percent": 100 * (a["cycle_seconds"] / b["cycle_seconds"] - 1),
                "sample_seconds": [a["sample_seconds"], b["sample_seconds"]],
                "update_seconds": [a["update_seconds"], b["update_seconds"]],
                "save_seconds": [a["save_seconds"], b["save_seconds"]],
                "games": a["counts"]["games"],
                "truncated": a["counts"]["truncated"],
                "retained_samples": a["counts"]["retained_samples"],
                "updates": a["updates"],
                "samples_consumed": a["samples_consumed"],
                "samples_hash": a["samples_hash"],
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", nargs="+", type=Path, required=True)
    parser.add_argument("--after", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.before) != len(args.after):
        parser.error("优化前后测量次数必须相同")
    pairs = []
    for a, b in zip(args.before, args.after):
        pairs.extend(
            compare(
                json.loads(a.read_text(encoding="utf-8")), json.loads(b.read_text(encoding="utf-8"))
            )
        )
    before = statistics.median(r["before_seconds"] for r in pairs)
    after = statistics.median(r["after_seconds"] for r in pairs)
    result = {
        "format": "haojie-cycle-comparison-v1",
        "matched_work": True,
        "pairs": pairs,
        "median_before_seconds": before,
        "median_after_seconds": after,
        "time_reduction_percent": 100 * (1 - after / before),
        "throughput_gain_percent": 100 * (before / after - 1),
        "before_range": [
            min(r["before_seconds"] for r in pairs),
            max(r["before_seconds"] for r in pairs),
        ],
        "after_range": [
            min(r["after_seconds"] for r in pairs),
            max(r["after_seconds"] for r in pairs),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
