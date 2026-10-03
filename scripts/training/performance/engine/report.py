"""读取已经完成的交错计时，保留全部轮次及最近秩 p95；不混合不同边界的倍率。"""

import argparse
import json
import math
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def distribution(values):
    ordered = sorted(values)
    return {"rounds": values, "median": statistics.median(values),
            "p95": ordered[math.ceil(len(ordered) * .95) - 1]}


def metrics(rows):
    keys = set().union(*(row.keys() for row in rows))
    return {key: (max if key.startswith("max") else sum)(row.get(key, 0) for row in rows)
            for key in sorted(keys)}


def summarize(kind, path):
    data = read(path / "summary.json")
    pairs = data["measurements"] if kind == "runtime" else data["pairs"]
    result = {"path": str(path.resolve()), "kind": kind}
    totals = {}
    for label in ("baseline", "candidate"):
        runs = [pair[label] for pair in pairs]
        if kind == "runtime":
            seconds = [r["elapsedMs"] / 1000 for r in runs]
            phase_rows = [metrics([row["metrics"] for row in r["rows"]]) for r in runs]
            phases = [{k: v / 1000 for k, v in m.items() if k.endswith("Ms")} for m in phase_rows]
            tails = [distribution([row["elapsedMs"] / 1000 for row in r["rows"]])["p95"] for r in runs]
            commands = [sum(row["commands"] for row in r["rows"]) for r in runs]
            if "before" in runs[0]:
                cpu = [r["after"]["CPU"] - r["before"]["CPU"] for r in runs]
                rss = [r["after"]["PeakWorkingSet64"] for r in runs]
            else:
                cpu = [(r["cpuUserUs"] + r["cpuSystemUs"]) / 1e6 for r in runs]
                rss = [max(row["rss"] for row in r["rows"]) for r in runs]
            entry = {"case_p95_seconds": distribution(tails), "cpu_seconds": distribution(cpu),
                     "rss_bytes": distribution(rss),
                     "work": {k: v for k, v in phase_rows[0].items() if not k.endswith("Ms")}}
            entry["engine_work_seconds"] = distribution([
                sum(p[k] for k in ("treeMs", "encodingMs", "samplingMs", "observationMs", "stepMs"))
                for p in phases])
            entry["unclassified_import_context_seconds"] = distribution([
                seconds[i] - sum(p.values()) for i, p in enumerate(phases)])
            if "requestMs" in runs[0]:
                entry["request_seconds"] = distribution([r["requestMs"] / 1000 for r in runs])
        elif kind == "application":
            seconds = [r["seconds"] for r in runs]
            phases = [r["phases"] for r in runs]
            games = [[g for report in r["reports"] for g in report["games"]] for r in runs]
            commands = [sum(g["outcome"]["commands"] for g in row) for row in games]
            entry = {"game_p95_seconds": distribution([
                distribution([g["elapsedMs"] / 1000 for g in row])["p95"] for row in games]),
                "model_forward_seconds": distribution([sum(p["inference_seconds"] for p in r["reports"]) for r in runs]),
                "model_forwards": [sum(p["forwards"] for p in r["reports"]) for r in runs],
                "metrics": {key: distribution([r["metrics"][key] for r in runs]) for key in runs[0]["metrics"]},
                "parameters": runs[0]["parameters"], "validation_examples": runs[0]["validation_examples"]}
        else:
            seconds = [r["strict_seconds"] for r in runs]
            phases = [{k: v["seconds"] for k, v in r["stages"].items()} for r in runs]
            commands = [sum(row["report"]["commands"] for row in r["stages"]["audit"]["rows"]) for r in runs]
            entry = {"stages": {key: {
                "record_p95_seconds": distribution([r["stages"][key]["record_p95_seconds"] for r in runs]),
                "startup_seconds": distribution([r["stages"][key]["startup_seconds"] for r in runs]),
                "work": {field: sum(row[field] for row in runs[0]["stages"][key]["rows"])
                         for field in ("examples", "entities", "candidates", "tensor_bytes")}}
                for key in runs[0]["stages"]}}
        if kind != "runtime":
            entry["resources"] = {key: distribution([r["resources"][key] for r in runs])
                                  for key in ("cpu_seconds", "engine_cpu_seconds", "python_cpu_seconds", "sampled_peak_rss_bytes")}
        entry.update(seconds=distribution(seconds), commands=commands,
                     commands_per_second=distribution([n / t for n, t in zip(commands, seconds)]),
                     phases={k: distribution([p[k] for p in phases]) for k in phases[0]})
        result[label] = entry
        totals[label] = seconds
    result["speedup"] = distribution([a / b for a, b in zip(totals["baseline"], totals["candidate"])])
    if (path / "manifest.json").exists():
        result["manifest"] = read(path / "manifest.json")
    result["note"] = "runtime 的 CPU 包含未计入内核墙钟的边界工作；TS 的 RSS 共用进程。应用 inferenceMs 是进程累计等待，不能与墙钟相加。p95 为最近秩，三轮总时长的 p95 等于最大轮。"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("runtime", "data", "application"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(summarize(args.kind, args.path), ensure_ascii=False, indent=2)
    if args.output:
        with args.output.open("x", encoding="utf-8") as file:
            file.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
