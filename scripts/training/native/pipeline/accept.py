"""有限 CPU 闭环验收：另起进程恢复，并核对真实权重变化与优化器连续性。"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--starts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    entries = []

    def run(name, module, *argv):
        before = time.perf_counter()
        with (args.output / f"{name}.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, "-m", module, *map(str, argv)],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=900,
            )
        entries.append({"stage": name, "seconds": time.perf_counter() - before})

    def native(name, *argv):
        run(name, "haojie_training.native", "--engine", args.engine, *argv)

    initial = args.output / "initial.pt"
    first = args.output / "updated.pt"
    restored = args.output / "restored.pt"
    straight = args.output / "straight.pt"
    data = args.output / "data"
    native("init", "init", "--tiny", "--checkpoint", initial)
    native(
        "sample",
        "sample",
        "--checkpoint",
        initial,
        "--starts",
        args.starts,
        "--output",
        args.output / "sample",
        "--commands",
        500,
        "--plies",
        60,
    )
    records = sorted((args.output / "sample").glob("*.jsonl"))
    native("audit", "audit", *records)
    native("prepare", "prepare", *records, "--output", data)
    manifest = json.loads((data / "manifest.json").read_text())
    assert all(s["value_labels"] > 0 for s in manifest["splits"].values())
    common = [
        "--data",
        data / "train.pt",
        "--validation",
        data / "validation.pt",
        "--batch-size",
        4,
        "--threads",
        1,
        "--device",
        "cpu",
    ]
    run(
        "update",
        "haojie_training.train",
        *common,
        "--steps",
        2,
        "--initialize-from",
        initial,
        "--checkpoint",
        first,
        "--report",
        args.output / "update.json",
    )
    run(
        "restore",
        "haojie_training.train",
        *common,
        "--steps",
        2,
        "--resume",
        first,
        "--checkpoint",
        restored,
        "--report",
        args.output / "restore.json",
    )
    run(
        "straight",
        "haojie_training.train",
        *common,
        "--steps",
        4,
        "--initialize-from",
        initial,
        "--checkpoint",
        straight,
    )
    checkpoints = [
        torch.load(p, map_location="cpu", weights_only=True)
        for p in (initial, first, restored, straight)
    ]
    a, b, c, d = checkpoints
    assert (a["updates"], b["updates"], c["updates"], d["updates"]) == (0, 2, 4, 4)
    changed = sum(not torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert changed > 0
    assert all(torch.equal(c["model"][k], d["model"][k]) for k in c["model"])
    for state in c["optimizer"]["state"].values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all()
    native(
        "sample-updated",
        "sample",
        "--checkpoint",
        restored,
        "--starts",
        args.starts,
        "--output",
        args.output / "sample-updated",
        "--commands",
        500,
        "--plies",
        60,
    )
    sampled = json.loads((args.output / "sample-updated/report.json").read_text())
    assert sampled["forwards"] > 0
    initial_report = json.loads((args.output / "sample/report.json").read_text())
    assert sampled["model"] != initial_report["model"]
    native(
        "audit-updated",
        "audit",
        *sorted((args.output / "sample-updated").glob("*.jsonl")),
    )
    report = {
        "stages": entries,
        "seconds": time.perf_counter() - started,
        "splits": manifest["splits"],
        "updates": [p["updates"] for p in checkpoints],
        "changed_parameter_tensors": changed,
        "resume_equals_uninterrupted": True,
        "model_hash": sampled["model"],
        "python": sys.version,
        "torch": str(torch.__version__),
        "pid": os.getpid(),
        "scope": "真实可重放晚盘功能验收，不是开局吞吐或棋力证据",
    }
    (args.output / "acceptance.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
