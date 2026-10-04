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
                "terminal": outcome["terminated"],
                "winner": outcome["winner"],
                "samples": audits[task].get("samples", 0),
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
    }
    valid = audit["valid_terminal"]
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
        },
        "steady_window": "unavailable: aborted before predeclared 300 second start",
        "sampling_usage": json.loads((root / "formal.hardware.usage.json").read_text()),
        "audit_usage": json.loads((root / "audit.hardware.usage.json").read_text()),
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
