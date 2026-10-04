"""同CUDA BF16执行方式的另进程0→2→4与连续四步验收，含全量指标和新权重采样。"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


def same(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b, (a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "data", "engine", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    common = [
        sys.executable,
        "-m",
        "haojie_training.train",
        "--data",
        str(args.data / "train.pt"),
        "--validation",
        str(args.data / "validation.pt"),
        "--device",
        "cuda",
        "--precision",
        "bf16",
        "--threads",
        "1",
        "--batch-size",
        "4",
        "--length-bucket-size",
        "64",
        "--deterministic",
    ]
    reports = {}
    for name, steps, parent, option in [
        ("two", 2, args.checkpoint, "--initialize-from"),
        ("resumed", 2, args.output / "two.pt", "--resume"),
        ("continuous", 4, args.checkpoint, "--initialize-from"),
    ]:
        command = common + [
            "--steps",
            str(steps),
            option,
            str(parent),
            "--checkpoint",
            str(args.output / (name + ".pt")),
            "--report",
            str(args.output / (name + ".json")),
        ]
        with (args.output / (name + ".log")).open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        reports[name] = json.loads(
            (args.output / (name + ".json")).read_text(encoding="utf-8")
        )
    a = torch.load(args.output / "resumed.pt", weights_only=True)
    b = torch.load(args.output / "continuous.pt", weights_only=True)
    for key in (
        "model",
        "optimizer",
        "scaler",
        "rng_cpu",
        "rng_device",
        "steps",
        "updates",
        "metadata",
    ):
        same(a[key], b[key])
    assert a["updates"] == 4
    # 两个自然开局只做新权重管道验收，截断保持unknown，不重复生成训练库。
    starts = args.output / "starts.json"
    starts.write_text(
        json.dumps([{"seed": 71, "rules": "classic"}, {"seed": 72, "rules": "shrine"}]),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "haojie_training.native",
        "--engine",
        str(args.engine),
        "sample",
        "--checkpoint",
        str(args.output / "resumed.pt"),
        "--starts",
        str(starts),
        "--output",
        str(args.output / "sample"),
        "--device",
        "cuda",
        "--precision",
        "bf16",
        "--commands",
        "8",
    ]
    with (args.output / "sample.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "haojie_training.native",
                "--engine",
                str(args.engine),
                "audit",
                *map(str, sorted((args.output / "sample").glob("*.jsonl"))),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    result = {
        "equal": True,
        "keys": [
            "model",
            "optimizer",
            "scaler",
            "rng_cpu",
            "rng_device",
            "steps",
            "updates",
            "metadata",
        ],
        "updates": a["updates"],
        "reports": reports,
    }
    (args.output / "report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps({"equal": True, "updates": a["updates"], "sample_audit": True}))


if __name__ == "__main__":
    main()
