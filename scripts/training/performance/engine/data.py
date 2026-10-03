"""固定原始记录上的独立审核/审核编码出口；完整验证次数相等，含真实二进制消费。"""
import argparse
import hashlib
import json
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path

import torch
from haojie_training.native.client import Client

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernel"))
from usage import Usage


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "records", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--rounds", type=int, default=3)
    args = p.parse_args()
    if not 1 <= args.rounds <= 10:
        raise ValueError("轮数必须为1至10")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    records = sorted(args.records.glob("sample-*/game-*.jsonl"))
    if not records:
        records = sorted(args.records.glob("game-*.jsonl"))
    if not records:
        raise ValueError("没有固定记录")
    binaries = {}
    for label in ("baseline", "candidate"):
        binaries[label] = args.output / (label + ".exe")
        shutil.copyfile(getattr(args, label), binaries[label])
    shutil.copyfile(__file__, args.output / "data.py")
    manifest = {"python": platform.python_version(), "torch": str(torch.__version__),
                "records": [{"path": str(f.resolve()), "sha256": digest(f), "bytes": f.stat().st_size} for f in records],
                "engines": {k: digest(v) for k, v in binaries.items()}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    references = {}

    def run(label, name):
        stages = {}
        usage = Usage().start()
        try:
            for encode in (False, True):
                rows = []
                started = time.perf_counter()
                with Client(binaries[label]) as client:
                    cold = time.perf_counter() - started
                    for i, path in enumerate(records):
                        start = time.perf_counter()
                        counts = {"examples": 0, "entities": 0, "candidates": 0, "tensor_bytes": 0}
                        client.send({"op": "audit", "record": str(path.resolve()), "encode": encode})
                        while True:
                            row = client.receive()
                            if row["type"] == "example":
                                counts["examples"] += 1
                                for key in ("entities", "candidates"):
                                    counts[key] += row[key]
                                counts["tensor_bytes"] += row["bytes"]
                            if row["type"] == "done":
                                break
                        seconds = time.perf_counter() - start
                        report = row["report"]
                        identity = {k: report[k] for k in ("outcome", "complete", "incompleteTail", "finalHash", "commands")}
                        expected = references.setdefault((encode, i), (identity, counts))
                        if (identity, counts) != expected:
                            raise AssertionError("审核结果/输出工作量不等价")
                        rows.append({"seconds": seconds, "report": report, **counts})
                elapsed = time.perf_counter() - started
                timings = sorted(r["seconds"] for r in rows)
                stages["encode" if encode else "audit"] = {"seconds": elapsed, "startup_seconds": cold,
                    "record_p95_seconds": timings[(len(timings) * 95 + 99) // 100 - 1], "rows": rows,
                    "commands_per_second": sum(r["report"]["commands"] for r in rows) / elapsed}
        finally:
            resources = usage.stop()
        result = {"label": label, "stages": stages, "resources": resources,
                  "strict_seconds": sum(s["seconds"] for s in stages.values())}
        (args.output / (name + ".json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"run": name, "strict_seconds": result["strict_seconds"],
                          "stages": {k: v["seconds"] for k, v in stages.items()}}), flush=True)
        return result

    run("baseline", "warmup-baseline")
    run("candidate", "warmup-candidate")
    pairs = []
    for n in range(args.rounds):
        order = ("candidate", "baseline") if n % 2 else ("baseline", "candidate")
        pairs.append({label: run(label, f"round-{n}-{label}") for label in order})
    summary = {"pairs": pairs, "speedup_median": statistics.median(
        pair["baseline"]["strict_seconds"] / pair["candidate"]["strict_seconds"] for pair in pairs)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
