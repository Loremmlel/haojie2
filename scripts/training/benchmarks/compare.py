"""完整周期对照：先核对任务、样本、预算和更新量，再报告墙钟及吞吐。"""

import argparse
import json
import statistics
from pathlib import Path


def validate_round(row, planned, updates):
    """所有计划任务都要有结案记录；真实预算截断可保留，取消和遗漏不能混入计时。"""
    counts, tasks = row["counts"], row["tasks"]
    if (
        counts["tasks"] != planned
        or len(tasks) != planned
        or len({task["task"] for task in tasks}) != planned
        or counts["games"] + counts["truncated"] != planned
        or counts["discarded"]
        or sum(bool(task["outcome"]["terminated"]) for task in tasks) != counts["games"]
        or any(task["outcome"].get("reason") in {"cancelled", "error"} for task in tasks)
        or any(not task["final_hash"] for task in tasks)
    ):
        raise ValueError("计划任务未全部正常结案，或终局/截断/丢弃计数不一致")
    if row["updates"] != updates or not counts["games"]:
        raise ValueError("完整周期没有得到预期的有效更新或真实终局")


def compare(before, after, *, allow_thread_change=False):
    if any(r.get("format") != "haojie-cycle-benchmark-v2" for r in (before, after)):
        raise ValueError("需要 v2 基准报告；旧报告缺少稳定硬件和完整任务验收，请重新测量")
    if not before["measurement_eligible"] or not after["measurement_eligible"]:
        raise ValueError("请求捕获运行包含额外开销，不能进入吞吐对照")
    if not before["complete"] or not after["complete"]:
        raise ValueError("未完成的测量不能进入吞吐对照")
    if not before["options"]["ordered_updates"] or not after["options"]["ordered_updates"]:
        raise ValueError("配对测量须使用 --ordered-updates，确保实际更新消费相同的样本序列")
    for key in (
        "torch",
        "python",
        "parameters",
        "model_config",
        "initial_model_sha256",
        "config",
        "starts_sha256",
        "hardware_identity",
        "execution",
    ):
        if before[key] != after[key]:
            raise ValueError(f"对照条件不同：{key}")
    for key in ("start_indices", "rounds", "ordered_updates", "device"):
        if before["options"][key] != after["options"][key]:
            raise ValueError(f"对照选项不同：{key}")
    execution_changes = {}
    if before["options"]["threads"] != after["options"]["threads"]:
        if not allow_thread_change or before["execution"]["device"] != "cpu":
            raise ValueError("对照选项不同：threads")
        execution_changes["threads"] = {
            "before": before["options"]["threads"],
            "after": after["options"]["threads"],
        }
    if any(len(r["rounds"]) != r["options"]["rounds"] or not r["rounds"] for r in (before, after)):
        raise ValueError("未完成全部计划训练轮次")
    results = []
    for a, b in zip(before["rounds"], after["rounds"]):
        for report, row in ((before, a), (after, b)):
            validate_round(
                row, report["config"]["games_per_round"], report["config"]["updates_per_round"]
            )
        for key in ("round", "samples_hash", "counts", "updates", "samples_consumed", "tasks"):
            if a[key] != b[key]:
                raise ValueError(f"第 {a['round']} 轮负载不同：{key}")
        results.append(
            {
                "before_seconds": a["cycle_seconds"],
                "after_seconds": b["cycle_seconds"],
                "execution_changes": execution_changes,
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
    parser.add_argument(
        "--allow-thread-change",
        action="store_true",
        help="显式比较 CPU 算子线程调整并列出差异；精度、硬件、任务及其他条件仍须相同",
    )
    args = parser.parse_args()
    if len(args.before) != len(args.after):
        parser.error("优化前后测量次数必须相同")
    pairs = []
    for a, b in zip(args.before, args.after):
        pairs.extend(
            compare(
                json.loads(a.read_text(encoding="utf-8")),
                json.loads(b.read_text(encoding="utf-8")),
                allow_thread_change=args.allow_thread_change,
            )
        )
    before = statistics.median(r["before_seconds"] for r in pairs)
    after = statistics.median(r["after_seconds"] for r in pairs)
    result = {
        "format": "haojie-cycle-comparison-v1",
        "matched_work": True,
        "same_execution": all(not pair["execution_changes"] for pair in pairs),
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
