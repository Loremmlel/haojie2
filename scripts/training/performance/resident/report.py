"""从完整账本、单遍审核和资源趋势生成产能账；错误中止只能生成条件外推。"""

import argparse
import json
import shutil
import sqlite3
import statistics
from collections import Counter
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


def steady_window(run):
    pool = run["pool"]
    end = min(3300, pool.get("stop_admission_seconds") or pool["seconds"])
    rows = [r for r in run["telemetry"] if 300 <= r["seconds"] <= end]
    result = {
        "run": run["run"],
        "predeclared_start": 300,
        "predeclared_end": end,
        "available": False,
    }
    if len(rows) >= 2:
        first, last = rows[0], rows[-1]
        seconds = last["seconds"] - first["seconds"]
        terminals = last["terminal_total"] - first["terminal_total"]
        result.update(
            available=True,
            observed_start=first["seconds"],
            observed_end=last["seconds"],
            seconds=seconds,
            terminals=terminals,
            games_per_hour=terminals / seconds * 3600,
            requests_per_second=(last["requests"] - first["requests"]) / seconds,
            mean_batch=(last["requests"] - first["requests"])
            / max(1, last["forwards"] - first["forwards"]),
            average_active_slots=(
                last["active_slot_seconds"] - first["active_slot_seconds"]
            )
            / seconds,
            rolling_response_p95_seconds=distribution(
                [r["response_p95_seconds"] for r in rows]
            ),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--previous", type=Path, help="保留失败验收的时间成本；同编号终局不重复计数"
    )
    args = parser.parse_args()
    root = args.root
    db = sqlite3.connect(root / "formal/tasks.sqlite")
    runs = []
    for number, raw in db.execute("SELECT id,config FROM runs ORDER BY id"):
        options = json.loads(raw)
        prefix = "formal" if number == 1 else "formal-resume"
        failure = root / f"formal/failure-{number}.json"
        report = json.loads(
            (
                failure
                if failure.exists()
                else root / f"formal/report-{number:04d}.json"
            ).read_text(encoding="utf-8")
        )
        launch = root / f"{prefix}.launch.json"
        offset = (
            json.loads(launch.read_text(encoding="utf-8"))["launch_offset_seconds"]
            if launch.exists()
            else 3600 - options["seconds"]
        )
        total_file = root / f"{prefix}.total.json"
        seconds = (
            json.loads(total_file.read_text(encoding="utf-8"))["wrapper_seconds"]
            if total_file.exists()
            else report["seconds"] + offset
        )
        runs.append(
            {
                "run": number,
                "options": options,
                "pool": report.get("pool", report),
                "offset": offset,
                "seconds": seconds,
                "aborted": failure.exists(),
                "telemetry": lines(root / f"formal/telemetry-{number:04d}.jsonl"),
                "hardware": lines(root / f"{prefix}.hardware.jsonl"),
                "usage": json.loads(
                    (root / f"{prefix}.hardware.usage.json").read_text(encoding="utf-8")
                ),
            }
        )
    pool = report.get("pool", report)
    sampling = sum(r["seconds"] for r in runs)
    previous = (
        json.loads(args.previous.read_text(encoding="utf-8")) if args.previous else None
    )
    if previous:
        sampling += previous["cost"]["sampling_seconds"]
    resource_runs = runs
    if previous:
        resource_runs = [
            {
                "seconds": previous["cost"]["sampling_seconds"],
                "pool": previous["pool"],
                "usage": previous["sampling_usage"],
                "hardware": lines(args.previous.parent / "formal.hardware.jsonl"),
            },
            *runs,
        ]
    audit = json.loads((root / "audit.total.json").read_text(encoding="utf-8"))
    all_audits = lines(root / "audit/audits.jsonl")
    audits = {r["task"]: r for r in all_audits}
    games = []
    for task, start, status, value in db.execute(
        "SELECT id,start,status,result FROM tasks ORDER BY id"
    ):
        start = json.loads(start)
        attempt = db.execute(
            "SELECT result FROM attempts WHERE task=? ORDER BY attempt DESC LIMIT 1",
            (task,),
        ).fetchone()
        progress = json.loads(attempt[0]) if attempt and attempt[0] else {}
        row = progress
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
                "input_sha256": audits[task]["audit"]["inputSha256"],
                "final_hash": audits[task]["audit"]["finalHash"],
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
    attempt_lookup = {(r["task"], r["attempt"]): r for r in all_audits}
    before_repair = {}
    backup = root / "formal/tasks.before-budget-repair.sqlite"
    if backup.exists():
        with sqlite3.connect(backup) as old:
            before_repair = {
                (t, a): json.loads(v)
                for t, a, v in old.execute(
                    "SELECT task,attempt,result FROM attempts WHERE result IS NOT NULL"
                )
            }
    attempts = []
    for task, attempt, run, status, raw in db.execute(
        "SELECT task,attempt,run,status,result FROM attempts ORDER BY task,attempt"
    ):
        row = {
            **before_repair.get((task, attempt), {}),
            **(json.loads(raw) if raw else {}),
        }
        evidence = attempt_lookup[(task, attempt)]
        attempts.append(
            {
                "task": task,
                "attempt": attempt,
                "run": run,
                "status": status,
                "commands": evidence["audit"]["commands"],
                "ply": evidence["audit"]["ply"],
                "requests": row.get("requests", 0),
                "seconds": row.get("task_seconds", row.get("seconds")),
                "started_wall": row.get("started_wall"),
                "finished_wall": row.get("finished_wall"),
                "eligible": evidence["eligible"],
                "rejected": evidence["audit"].get("rejected", 0),
                "reason": evidence["audit"]["outcome"]["reason"],
                "rules": next(g["rules"] for g in games if g["task"] == task),
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
    command_count = sum(r["commands"] for r in attempts) + (
        previous["cost"]["commands"] if previous else 0
    )
    request_count = sum(r["pool"]["requests"] for r in runs) + (
        previous["pool"]["requests"] if previous else 0
    )
    audit_seconds = audit["wrapper_seconds"] + (
        previous["cost"]["audit_prepare_seconds"] if previous else 0
    )
    cost = {
        "sampling_seconds": sampling,
        "import_offset_seconds": sum(r["offset"] for r in runs)
        + (previous["cost"]["import_offset_seconds"] if previous else 0),
        "sampling_including_shutdown_seconds": sampling,
        "audit_prepare_seconds": audit_seconds,
        "serial_seconds": sampling + audit_seconds,
        "commands": command_count,
        "commands_per_second": command_count / sampling,
        "requests": request_count,
        "requests_per_second": request_count / sampling,
        "terminal_samples": sum(g["samples"] for g in terminal),
        "samples_per_terminal_game": sum(g["samples"] for g in terminal)
        / max(1, len(terminal)),
        "all_samples_per_valid_game": audit["samples"] / max(1, len(terminal)),
        "train_samples": sum(g["samples"] for g in games if g["split"] == "train"),
        "validation_samples": sum(
            g["samples"] for g in games if g["split"] == "validation"
        ),
    }
    valid = audit["valid_terminal"]
    telemetry = runs[-1]["telemetry"]
    slots = {}
    for run in runs:
        for snapshot in run["telemetry"]:
            for active in snapshot["active"]:
                slots[(run["run"], active["task"], active["attempt"])] = active["slot"]
    gaps = []
    for run_number, slot in (
        (r["run"], i) for r in runs for i in range(r["pool"]["environments"])
    ):
        ordered = sorted(
            [
                g
                for g in attempts
                if g["run"] == run_number
                and slots.get((run_number, g["task"], g["attempt"])) == slot
                and g["started_wall"]
            ],
            key=lambda g: g["started_wall"],
        )
        for old, new in zip(ordered, ordered[1:]):
            if old["finished_wall"]:
                gaps.append(new["started_wall"] - old["finished_wall"])
    steady = steady_window(runs[-1])
    cost.update(
        startup_and_import_seconds=sum(
            r["pool"]["startup_seconds"] + r["offset"] for r in runs
        )
        + (
            previous["pool"]["startup_seconds"]
            + previous["cost"]["import_offset_seconds"]
            if previous
            else 0
        ),
        first_fill_including_import_seconds=telemetry[0]["seconds"] + offset,
        final_drain_seconds=pool["seconds"] - pool["stop_admission_seconds"]
        if pool.get("stop_admission_seconds") is not None
        else None,
        rejected=sum(g["rejected"] for g in attempts),
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
                "raw_bytes_30_days": (
                    audit["record_bytes"]
                    + (previous["audit"]["record_bytes"] if previous else 0)
                )
                / max(1, valid)
                * g30,
                "shard_bytes_30_days": (
                    audit["shard_bytes"]
                    + (previous["audit"]["shard_bytes"] if previous else 0)
                )
                / max(1, valid)
                * g30,
            }
        )
    hardware = []
    elapsed = 0
    cpu_total = busy = own = 0
    for run in resource_runs:
        segment = run["hardware"]
        first, last = segment[0], segment[-1]
        cpu_total += (
            last["system_cpu_total_seconds"] - first["system_cpu_total_seconds"]
        )
        busy += last["system_cpu_busy_seconds"] - first["system_cpu_busy_seconds"]
        own += (
            last["python_cpu_seconds"]
            - first["python_cpu_seconds"]
            + last["engine_cpu_seconds"]
            - first["engine_cpu_seconds"]
        )
        for index, row in enumerate(segment):
            item = {**row, "seconds": row["seconds"] + elapsed}
            if index:
                prior = segment[index - 1]
                interval = (
                    row["system_cpu_total_seconds"] - prior["system_cpu_total_seconds"]
                )
                used = row["system_cpu_busy_seconds"] - prior["system_cpu_busy_seconds"]
                owned = (
                    row["python_cpu_seconds"]
                    - prior["python_cpu_seconds"]
                    + row["engine_cpu_seconds"]
                    - prior["engine_cpu_seconds"]
                )
                if interval > 0:
                    item["cpu_busy_percent_interval"] = 100 * used / interval
                    item["cpu_background_percent_interval"] = (
                        100 * (used - owned) / interval
                    )
            hardware.append(item)
        elapsed += run["seconds"]
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
            "gpu_util": distribution(
                [
                    float(r["gpu"].split(",")[0])
                    for r in hardware
                    if r.get("gpu")
                    and hardware[-1]["seconds"] * i / 4
                    <= r["seconds"]
                    <= hardware[-1]["seconds"] * (i + 1) / 4
                ]
            ),
            "rss_bytes": distribution(
                [
                    r["python_rss_bytes"] + r["engine_rss_bytes"]
                    for r in hardware
                    if hardware[-1]["seconds"] * i / 4
                    <= r["seconds"]
                    <= hardware[-1]["seconds"] * (i + 1) / 4
                ]
            ),
            "gpu_temperature_c": distribution(
                [
                    float(r["gpu"].split(",")[2])
                    for r in hardware
                    if r.get("gpu")
                    and hardware[-1]["seconds"] * i / 4
                    <= r["seconds"]
                    <= hardware[-1]["seconds"] * (i + 1) / 4
                ]
            ),
            **{
                field: distribution(
                    [
                        r[field]
                        for r in hardware
                        if field in r
                        and hardware[-1]["seconds"] * i / 4
                        <= r["seconds"]
                        <= hardware[-1]["seconds"] * (i + 1) / 4
                    ]
                )
                for field in (
                    "cpu_busy_percent_interval",
                    "cpu_background_percent_interval",
                )
            },
        }
        for i in range(4)
    ]
    result = {
        "status": "aborted; no validated sustainable rate"
        if failure.exists()
        else "completed",
        "options": options,
        "games": games,
        "counts": {
            "unique_tasks": len(games),
            "attempts_in_ledger": len(attempts),
            "all_formal_attempts": len(attempts)
            + (previous["audit"]["attempts"] if previous else 0),
            "task_states": dict(Counter(g["status"] for g in games)),
            "attempt_reasons": dict(Counter(a["reason"] for a in attempts)),
            "valid_terminal": valid,
            "raw_terminal_attempts_including_duplicates": sum(
                r["audit"]["complete"] and r["audit"]["outcome"]["terminated"]
                for r in all_audits
            )
            + (previous["audit"]["valid_terminal"] if previous else 0),
            "draws": sum(g["winner"] == "draw" for g in terminal),
            "cancelled_current": sum(g["reason"] == "cancelled" for g in games),
            "unfinished_current": sum(g["reason"] == "interrupted" for g in games),
            "budget_truncated_tasks": sum(
                g["status"] == "decode-budget" for g in games
            ),
        },
        "attempts": attempts,
        "runs": [
            {k: v for k, v in r.items() if k not in {"telemetry", "hardware"}}
            for r in runs
        ],
        "modes": modes,
        "cost": cost,
        "pool": pool,
        "cumulative_pool_metrics": {
            key: sum(r["pool"][key] for r in resource_runs)
            for key in (
                "requests",
                "forwards",
                "inference_seconds",
                "collation_seconds",
                "send_seconds",
                "queue_seconds",
                "tensor_read_seconds",
                "decode_validate_seconds",
                "active_slot_seconds",
                "tensor_bytes",
                "refills",
                "refill_seconds",
                "restarts",
            )
        },
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
            "one_use_serial_g30_bf16": valid
            / (cost["serial_seconds"] + audit["samples"] / 208.37)
            * 2592000,
            "one_use_serial_g30_fp32": valid
            / (cost["serial_seconds"] + audit["samples"] / 145.03)
            * 2592000,
            "one_use_train_split_serial_g30_bf16": valid
            / (cost["serial_seconds"] + cost["train_samples"] / 208.37)
            * 2592000,
            "comparability": "same model/device/batch4/threads1; previous input lengths differ; conditional estimate only",
        },
        "storage": {
            "all_attempts": {
                key: audit[key] + (previous["audit"][key] if previous else 0)
                for key in (
                    "record_bytes",
                    "feature_bytes",
                    "shard_bytes",
                    "samples",
                    "value_labels",
                )
            },
            "terminal_only": {
                key: sum(g[key] for g in terminal)
                for key in (
                    "record_bytes",
                    "feature_bytes",
                    "shard_bytes",
                    "samples",
                    "value_labels",
                )
            },
            "bytes_per_sample": {
                key: audit[key] / max(1, audit["samples"])
                for key in ("record_bytes", "feature_bytes", "shard_bytes")
            },
            "bytes_per_valid_game_including_tails": {
                key: (audit[key] + (previous["audit"][key] if previous else 0))
                / max(1, valid)
                for key in ("record_bytes", "feature_bytes", "shard_bytes")
            },
        },
        "steady_window": steady,
        "steady_windows": [steady_window(r) for r in runs],
        "latest_run_new_terminals": sum(
            a["status"] == "terminal" for a in attempts if a["run"] == runs[-1]["run"]
        ),
        "superseded_attempts": {
            mode: {
                key: distribution(
                    [
                        a[key]
                        for a in attempts
                        if a["rules"] == mode
                        and not a["eligible"]
                        and a[key] is not None
                    ]
                )
                for key in ("commands", "ply", "requests", "seconds")
            }
            for mode in ("classic", "shrine")
        },
        "occupancy": {
            "overall_average_active_slots": sum(
                r["pool"]["active_slot_seconds"] for r in runs
            )
            / sum(r["pool"]["seconds"] for r in runs),
            "snapshots_at_full_capacity": sum(
                len(r["active"]) == pool["environments"] for r in telemetry
            ),
            "snapshots": len(telemetry),
            "refill_transaction_seconds": sum(
                r["pool"]["refill_seconds"] for r in runs
            ),
            "refill_acknowledgement_to_next_start_seconds": distribution(gaps),
            "tasks_with_snapshot_slot": len(slots),
            "note": "补位事务时间不等于端到端空闲；满载比例为五秒快照，慢局及排空计入总墙钟",
        },
        "sampling_usage": {
            key: (max if "peak" in key else sum)(r["usage"][key] for r in resource_runs)
            for key in (
                "cpu_seconds",
                "python_cpu_seconds",
                "engine_cpu_seconds",
                "sampled_peak_rss_bytes",
                "peak_single_process_bytes",
            )
        },
        "audit_usage": json.loads(
            (root / "audit.hardware.usage.json").read_text(encoding="utf-8")
        ),
    }
    if previous:
        result["initial_failed_acceptance_already_included"] = {
            "previous_serial_seconds": previous["cost"]["serial_seconds"],
            "serial_seconds": cost["serial_seconds"],
            "valid_unique_games": valid,
            "duplicate_previous_terminals_not_added": previous["audit"][
                "valid_terminal"
            ],
        }
    disk = shutil.disk_usage(root)
    raw_per_game = result["storage"]["bytes_per_valid_game_including_tails"][
        "record_bytes"
    ]
    shard_per_game = result["storage"]["bytes_per_valid_game_including_tails"][
        "shard_bytes"
    ]
    result["disk_capacity_if_all_retained"] = {
        "free_bytes_after_acceptance": disk.free,
        "volume_bytes": disk.total,
        "additional_games_raw_only": disk.free / max(1, raw_per_game),
        "additional_games_raw_and_shards": disk.free
        / max(1, raw_per_game + shard_per_game),
        "assumption": "实测字节/有效局保持；全量保留、不删除已有数据、无外部出口；非30天实测",
    }
    update_file = root / "update-cost.json"
    if update_file.exists():
        update = json.loads(update_file.read_text(encoding="utf-8"))
        rate = update["samples_per_second_including_load"]
        learning_seconds = cost["train_samples"] / rate
        result["measured_update_cost"] = {
            **{k: v for k, v in update.items() if k != "steps"},
            "uses": 1,
            "training_samples_per_valid_game": cost["train_samples"] / valid,
            "one_use_training_split_seconds": learning_seconds,
            "one_use_serial_games_per_hour": valid
            / (cost["serial_seconds"] + learning_seconds)
            * 3600,
            "one_use_serial_g30": valid
            / (cost["serial_seconds"] + learning_seconds)
            * 2592000,
            "assumption": "32批短测成本按训练划分样本数线性外推一次使用；另不重复扣80%学习预留",
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
