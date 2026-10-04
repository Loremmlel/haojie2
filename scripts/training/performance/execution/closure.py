"""少量实际Rust闭环；统一起点/命令预算，前缀重放和推理等待按原生字段单列。"""

import argparse
import json
import time
from pathlib import Path

import torch

from haojie_training.native import pipeline
from haojie_training.native.client import Client
from measure import load_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("engine", "checkpoint", "reference", "starts", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--commands", type=int, default=24)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    load_reference(args.reference)
    from reference_training.native.pipeline import sample as old_sample

    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.starts.read_text(encoding="utf-8"))
    starts = [source[i] for i in (0, 1, 2, 6, 7, 8, 12, 18)]
    (args.output / "starts.json").write_text(json.dumps(starts), encoding="utf-8")
    reports = []
    if args.startup_only:
        # 不执行模型：首请求到达后取消。此上界含前缀重放、记录头及首次查询/编码。
        for i, start in enumerate(starts):
            begin = time.perf_counter()
            with Client(args.engine) as client:
                startup = time.perf_counter() - begin
                begin = time.perf_counter()
                client.send(
                    {
                        "op": "sample",
                        "record": str((args.output / f"prefix-{i}.jsonl").resolve()),
                        "start": start,
                        "model": "a" * 64,
                        "samplerSeed": 20261003 + i,
                        "maxCommands": args.commands,
                        "maxPlies": 60,
                    }
                )
                message = client.receive()
                ready_seconds = time.perf_counter() - begin
                assert message["type"] == "infer"
                client.send(
                    {"id": message["id"], "model": message["model"], "cancel": True}
                )
                done = client.receive()
                assert done["outcome"]["reason"] == "cancelled"
                reports.append(
                    {
                        "start": i,
                        "rules": start["rules"],
                        "prelude_commands": len(start.get("prelude", [])),
                        "startup_seconds": startup,
                        "first_request_seconds": ready_seconds,
                    }
                )
        (args.output / "report.json").write_text(
            json.dumps(reports, indent=2), encoding="utf-8"
        )
        print(json.dumps(reports))
        return
    cases = [
        ("reference", 4, "fp32", 0),
        ("optimized", 4, "fp32", 0),
        ("microbatch", 4, "fp32", 0.5),
        ("bf16", 4, "bf16", 0.5),
        ("bf16-eight", 8, "bf16", 0.5),
    ]
    if args.final:
        cases = [
            ("reference-four", 4, "fp32", 0),
            ("optimized-four", 4, "fp32", 0),
            ("bf16-four", 4, "bf16", 0),
            ("reference-eight", 8, "fp32", 0),
            ("optimized-eight", 8, "fp32", 0),
            ("bf16-eight", 8, "bf16", 0),
        ]
    reference_hashes = None
    if args.cases:
        cases = [case for case in cases if case[0] in args.cases]
    for name, concurrency, precision, wait_ms in cases:
        begin = time.perf_counter()
        samples, audits = [], []
        for offset in range(0, len(starts), concurrency):
            output = args.output / f"{name}-{offset}"
            options = (
                {}
                if name.startswith("reference")
                else {"precision": precision, "batch_wait_ms": wait_ms}
            )
            sample = (old_sample if name.startswith("reference") else pipeline.sample)(
                args.engine,
                args.checkpoint,
                starts[offset : offset + concurrency],
                output,
                commands=args.commands,
                plies=60,
                device="cuda",
                seed=20261003 + offset,
                **options,
            )
            samples.append(sample)
        sampling_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        with Client(args.engine) as client:
            for path in sorted(args.output.glob(f"{name}-*/*.jsonl")):
                client.send(
                    {"op": "audit", "record": str(path.resolve()), "encode": False}
                )
                audits.append(client.receive())
        commands = sum(g["outcome"]["commands"] for s in samples for g in s["games"])
        hashes = [g["finalHash"] for s in samples for g in s["games"]]
        if reference_hashes is None:
            reference_hashes = hashes
        row = {
            "case": name,
            "concurrency": concurrency,
            "precision": precision,
            "batch_wait_ms": wait_ms,
            "sampling_seconds": sampling_seconds,
            "audit_seconds": time.perf_counter() - begin,
            "commands": commands,
            "commands_per_second": commands / sampling_seconds,
            "same_final_hashes": hashes == reference_hashes,
            "samples": samples,
            "audits": audits,
        }
        reports.append(row)
        (args.output / "report.json").write_text(
            json.dumps(reports, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {k: v for k, v in row.items() if k not in ("samples", "audits")}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
