"""从完整账本、单遍审核和资源趋势生成产能账；错误中止只能生成条件外推。"""

import argparse
import json
import sqlite3
import statistics
from pathlib import Path


def distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": values[0],
        "median": statistics.median(values),
        "p90_observed": values[int((len(values) - 1) * 0.9)]
        if len(values) >= 20
        else None,
        "max": values[-1],
        "mean": statistics.mean(values),
        "sum": sum(values),
    }


def lines(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path, help="保留失败验收的时间成本；同编号终局不重复计数")
    args = parser.parse_args()
    root = args.root
    db = sqlite3.connect(root / "formal/tasks.sqlite")
    options = json.loads(db.execute("SELECT config FROM runs WHERE id=1").fetchone()[0])
    failure = root / "formal/failure-1.json"
    report = json.loads(
        (failure if failure.exists() else root / "formal/report-0001.json").read_text()
    )
    pool = report.get("pool", report)
    offset = 3600 - options["seconds"]
    sampling = report["seconds"] + offset
    total_file = root / "formal.total.json"
    if total_file.exists():
        sampling = json.loads(total_file.read_text())["wrapper_seconds"]
    audit = json.loads((root / "audit.total.json").read_text())
    audits = {r["task"]: r for r in lines(root / "audit/audits.jsonl")}
    games = []
    for task, start, status, value in db.execute(
        "SELECT id,start,status,result FROM tasks ORDER BY id"
    ):
        start = json.loads(start)
        row = json.loads(value) if value else {}
        attempt = db.execute(
            "SELECT result FROM attempts WHERE task=? ORDER BY attempt DESC LIMIT 1",
            (task,),
        ).fetchone()
        progress = json.loads(attempt[0]) if attempt and attempt[0] else {}
        outcome = audits[task]["audit"]["outcome"]
        games.append(
            {
                "task": task,
                "rules": start["rules"],
                "seed": start["seed"],
                "status": status,
                "commands": outcome["commands"],
                "ply": audits[task]["audit"]["ply"],
                "requests": row.get("requests", progress.get("requests", 0)),
                "seconds": row.get("task_seconds", progress.get("seconds")),
                "started_wall": row.get("started_wall"),
                "finished_wall": row.get("finished_wall"),
                "terminal": outcome["terminated"],
                "reason": outcome["reason"],
                "rejected": audits[task]["audit"].get("rejected", 0),
                "winner": outcome["winner"],
                "samples": audits[task].get("samples", 0),
                "value_labels": audits[task].get("value_labels", 0),
                "split": audits[task].get("split"),
                "record_bytes": audits[task]["record_bytes"],
                "feature_bytes": audits[task].get("feature_bytes", 0),
                "shard_bytes": audits[task].get("shard_bytes", 0),
            }
        )
    db.close()
    modes = {}
    for mode in ("classic", "shrine"):
        selected = [g for g in games if g["rules"] == mode]
        modes[mode] = {"assigned": len(selected)}
        for label, group in (
            ("terminal", [g for g in selected if g["terminal"]]),
            ("unfinished_or_error", [g for g in selected if not g["terminal"]]),
        ):
            modes[mode][label] = {
                key: distribution([g[key] for g in group if g[key] is not None])
                for key in ("commands", "ply", "requests", "seconds", "samples")
            }
    terminal = [g for g in games if g["terminal"]]
    cost = {
        "sampling_seconds": sampling,
        "import_offset_seconds": offset,
        "sampling_including_shutdown_seconds": report["seconds"],
        "audit_prepare_seconds": audit["wrapper_seconds"],
        "serial_seconds": sampling + audit["wrapper_seconds"],
        "commands": sum(g["commands"] for g in games),
        "commands_per_second": sum(g["commands"] for g in games) / sampling,
        "requests_per_second": pool["requests"] / sampling,
        "terminal_samples": sum(g["samples"] for g in terminal),
        "samples_per_terminal_game": sum(g["samples"] for g in terminal)
        / max(1, len(terminal)),
        "all_samples_per_valid_game": audit["samples"] / max(1, len(terminal)),
        "train_samples": sum(g["samples"] for g in games if g["split"] == "train"),
        "validation_samples": sum(g["samples"] for g in games if g["split"] == "validation"),
    }
    valid = audit["valid_terminal"]
    telemetry = lines(root / "formal/telemetry-0001.jsonl")
    slots = {}
    for snapshot in telemetry:
        for active in snapshot["active"]:
            slots[active["task"]] = active["slot"]
    gaps = []
    for slot in range(pool["environments"]):
        ordered = sorted(
            [g for g in games if slots.get(g["task"]) == slot and g["started_wall"]],
            key=lambda g: g["started_wall"],
        )
        for old, new in zip(ordered, ordered[1:]):
            if old["finished_wall"]:
                gaps.append(new["started_wall"] - old["finished_wall"])
    end = min(3300, pool.get("stop_admission_seconds") or pool["seconds"])
    window = [r for r in telemetry if 300 <= r["seconds"] <= end]
    steady = {"predeclared_start": 300, "predeclared_end": end, "available": False}
    if len(window) >= 2:
        first, last = window[0], window[-1]
        seconds = last["seconds"] - first["seconds"]
        terminals = last["terminal_total"] - first["terminal_total"]
        steady.update(
            available=True,
            observed_start=first["seconds"],
            observed_end=last["seconds"],
            seconds=seconds,
            terminals=terminals,
            games_per_hour=terminals / seconds * 3600,
            requests_per_second=(last["requests"] - first["requests"]) / seconds,
            average_active_slots=(last["active_slot_seconds"] - first["active_slot_seconds"]) / seconds,
        )
    cost.update(
        startup_and_import_seconds=pool["startup_seconds"] + offset,
        first_fill_including_import_seconds=telemetry[0]["seconds"] + offset,
        final_drain_seconds=pool["seconds"] - pool["stop_admission_seconds"]
        if pool.get("stop_admission_seconds") is not None else None,
        rejected=sum(g["rejected"] for g in games),
    )
    scenarios = []
    for label, seconds, availability in (
        ("raw_record", sampling, 1),
        ("serial_audit_prepare", cost["serial_seconds"], 1),
        ("serial_80_percent_available", cost["serial_seconds"], 0.8),
        ("serial_50_percent_available", cost["serial_seconds"], 0.5),
    ):
        hourly = valid / seconds * 3600 * availability
        g30 = hourly * 720
        scenarios.append(
            {
                "name": label,
                "availability_assumption": availability,
                "games_per_hour": hourly,
                "g30_conditional": g30,
                "million_gap_games": 1e6 - g30,
                "million_gap_factor": 1e6 / g30 if g30 else None,
                "raw_bytes_30_days": audit["record_bytes"] / max(1, valid) * g30,
                "shard_bytes_30_days": audit["shard_bytes"] / max(1, valid) * g30,
            }
        )
    hardware = lines(root / "formal.hardware.jsonl")
    utilization = {}
    for key in (
        "python_rss_bytes",
        "engine_rss_bytes",
        "system_memory_load",
        "system_available_bytes",
    ):
        utilization[key] = distribution([r[key] for r in hardware])
    gpu_fields = (
        "util_percent",
        "memory_util_percent",
        "temperature_c",
        "memory_mib",
        "power_w",
        "clock_mhz",
    )
    for i, key in enumerate(gpu_fields):
        utilization["gpu_" + key] = distribution(
            [float(r["gpu"].split(",")[i]) for r in hardware if r.get("gpu")]
        )
    cpu_total = (
        hardware[-1]["system_cpu_total_seconds"]
        - hardware[0]["system_cpu_total_seconds"]
    )
    busy = (
        hardware[-1]["system_cpu_busy_seconds"] - hardware[0]["system_cpu_busy_seconds"]
    )
    own = (
        hardware[-1]["python_cpu_seconds"]
        - hardware[0]["python_cpu_seconds"]
        + hardware[-1]["engine_cpu_seconds"]
        - hardware[0]["engine_cpu_seconds"]
    )
    utilization.update(
        cpu_busy_percent=100 * busy / cpu_total,
        cpu_own_percent=100 * own / cpu_total,
        cpu_background_percent=100 * (busy - own) / cpu_total,
        cpu_temperature="unavailable",
    )
    utilization["trend_quarters"] = [
        {
            "from_seconds": hardware[-1]["seconds"] * i / 4,
            "to_seconds": hardware[-1]["seconds"] * (i + 1) / 4,
            "gpu_util": distribution([
                float(r["gpu"].split(",")[0]) for r in hardware
                if r.get("gpu") and hardware[-1]["seconds"] * i / 4 <= r["seconds"]
                <= hardware[-1]["seconds"] * (i + 1) / 4
            ]),
            "rss_bytes": distribution([
                r["python_rss_bytes"] + r["engine_rss_bytes"] for r in hardware
                if hardware[-1]["seconds"] * i / 4 <= r["seconds"]
                <= hardware[-1]["seconds"] * (i + 1) / 4
            ]),
        }
        for i in range(4)
    ]
    result = {
        "status": "aborted; no validated sustainable rate"
        if failure.exists()
        else "completed",
        "options": options,
        "games": games,
        "modes": modes,
        "cost": cost,
        "pool": pool,
        "audit": audit,
        "hardware": utilization,
        "conditional_arithmetic_only": scenarios,
        "training_reference": {
            "bf16_samples_per_second_previous_workload": 208.37,
            "fp32_samples_per_second_previous_workload": 145.03,
            "one_use_terminal_samples_seconds_bf16": cost["terminal_samples"] / 208.37,
            "one_use_all_samples_seconds_bf16": audit["samples"] / 208.37,
            "one_use_all_samples_seconds_fp32": audit["samples"] / 145.03,
            "one_use_training_split_seconds_bf16": cost["train_samples"] / 208.37,
            "one_use_training_split_seconds_fp32": cost["train_samples"] / 145.03,
            "uses": 1,
            "one_use_serial_g30_bf16": valid / (cost["serial_seconds"] + audit["samples"] / 208.37) * 2592000,
            "one_use_serial_g30_fp32": valid / (cost["serial_seconds"] + audit["samples"] / 145.03) * 2592000,
            "one_use_train_split_serial_g30_bf16": valid / (cost["serial_seconds"] + cost["train_samples"] / 208.37) * 2592000,
            "comparability": "same model/device/batch4/threads1; previous input lengths differ; conditional estimate only",
        },
        "storage": {
            "all_attempts": {
                key: sum(g[key] for g in games)
                for key in ("record_bytes", "feature_bytes", "shard_bytes", "samples", "value_labels")
            },
            "terminal_only": {
                key: sum(g[key] for g in terminal)
                for key in ("record_bytes", "feature_bytes", "shard_bytes", "samples", "value_labels")
            },
            "bytes_per_sample": {
                key: audit[key] / max(1, audit["samples"])
                for key in ("record_bytes", "feature_bytes", "shard_bytes")
            },
            "bytes_per_valid_game_including_tails": {
                key: audit[key] / max(1, valid)
                for key in ("record_bytes", "feature_bytes", "shard_bytes")
            },
        },
        "steady_window": steady,
        "occupancy": {
            "overall_average_active_slots": pool["active_slot_seconds"] / pool["seconds"],
            "snapshots_at_full_capacity": sum(len(r["active"]) == pool["environments"] for r in telemetry),
            "snapshots": len(telemetry),
            "refill_transaction_seconds": pool["refill_seconds"],
            "refill_acknowledgement_to_next_start_seconds": distribution(gaps),
            "tasks_with_snapshot_slot": len(slots),
            "note": "补位事务时间不等于端到端空闲；满载比例为五秒快照，慢局及排空计入总墙钟",
        },
        "sampling_usage": json.loads((root / "formal.hardware.usage.json").read_text()),
        "audit_usage": json.loads((root / "audit.hardware.usage.json").read_text()),
    }
    if args.previous:
        previous = json.loads(args.previous.read_text())
        all_seconds = cost["serial_seconds"] + previous["cost"]["serial_seconds"]
        result["including_failed_acceptance"] = {
            "previous_serial_seconds": previous["cost"]["serial_seconds"],
            "serial_seconds": all_seconds,
            "valid_unique_games": valid,
            "games_per_hour": valid / all_seconds * 3600,
            "g30_conditional": valid / all_seconds * 2592000,
            "duplicate_previous_terminals_not_added": previous["audit"]["valid_terminal"],
        }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "cost": cost,
                "conditional": scenarios,
                "modes": modes,
                "hardware": utilization,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
