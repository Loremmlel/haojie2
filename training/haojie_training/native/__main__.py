"""独立原生训练入口；学习/续训仍使用 haojie_training.train。"""

import argparse
import json
from pathlib import Path

import torch

from .client import Client
from .pipeline import audit as audit_record
from .pipeline import initialize, prepare, sample
from .resident import resident


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=1)
    sub = parser.add_subparsers(dest="op", required=True)
    init = sub.add_parser("init")
    init.add_argument("--checkpoint", type=Path, required=True)
    init.add_argument("--tiny", action="store_true")
    init.add_argument("--seed", type=int, default=20261003)
    sampling = sub.add_parser("sample")
    sampling.add_argument("--checkpoint", type=Path, required=True)
    sampling.add_argument("--starts", type=Path, required=True)
    sampling.add_argument("--output", type=Path, required=True)
    sampling.add_argument("--commands", type=int, default=2000)
    sampling.add_argument("--plies", type=int, default=200)
    sampling.add_argument("--device", choices=["auto", "cpu", "cuda", "xpu"], default="auto")
    sampling.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    sampling.add_argument("--batch-wait-ms", type=float, default=0.0)
    sampling.add_argument("--seed", type=int, default=20261003)
    continuous = sub.add_parser("resident", help="常驻补位采样；STOP文件或Ctrl+C停止接新任务")
    continuous.add_argument("--checkpoint", type=Path, required=True)
    continuous.add_argument("--output", type=Path, required=True)
    continuous.add_argument("--environments", type=int, default=8)
    continuous.add_argument("--tasks", type=int)
    continuous.add_argument("--target", type=int, default=64)
    continuous.add_argument("--seconds", type=float, default=3600)
    continuous.add_argument("--drain-seconds", type=float, default=120)
    continuous.add_argument("--seed", type=int, default=20261004)
    continuous.add_argument("--rules", choices=["mixed", "classic", "shrine"], default="mixed")
    continuous.add_argument("--commands", type=int, default=20000)
    continuous.add_argument("--plies", type=int, default=1000)
    continuous.add_argument("--device", choices=["cuda", "cpu", "auto", "xpu"], default="cuda")
    continuous.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    continuous.add_argument("--batch-wait-ms", type=float, default=0)
    continuous.add_argument("--resume", action="store_true")
    pool_audit = sub.add_parser("audit-pool", help="有界单遍审核及逐片准备工作池产物")
    pool_audit.add_argument("--source", type=Path, required=True)
    pool_audit.add_argument("--output", type=Path, required=True)
    pool_audit.add_argument("--shard-size", type=int, default=64)
    audit = sub.add_parser("audit")
    audit.add_argument("records", nargs="+", type=Path)
    audit.add_argument("--prepare", type=Path, help="在同一次审核中编码并准备数据，不重复重放")
    audit.add_argument("--shard-size", type=int, default=64)
    preparation = sub.add_parser("prepare")
    preparation.add_argument("records", nargs="+", type=Path)
    preparation.add_argument("--output", type=Path, required=True)
    preparation.add_argument("--shard-size", type=int, default=64)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("线程数必须为正")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.op == "init":
        result = initialize(args.engine, args.checkpoint, args.seed, args.tiny)
    elif args.op == "audit-pool":
        from .stream import prepare_pool

        result = prepare_pool(args.engine, args.source, args.output, args.shard_size)
    elif args.op == "resident":
        result = resident(
            args.engine,
            args.checkpoint,
            args.output,
            **{
                key: value
                for key, value in vars(args).items()
                if key not in {"engine", "checkpoint", "output", "op", "threads"}
            },
        )
    elif args.op == "sample":
        starts = json.loads(args.starts.read_text(encoding="utf-8"))
        if not isinstance(starts, list) or not 1 <= len(starts) <= 8:
            parser.error("每批支持1至8个环境")
        result = sample(
            args.engine,
            args.checkpoint,
            starts,
            args.output,
            args.commands,
            args.plies,
            args.device,
            args.seed,
            precision=args.precision,
            batch_wait_ms=args.batch_wait_ms,
        )
    elif args.op == "prepare":
        if args.shard_size < 1:
            parser.error("分片大小必须为正")
        result = prepare(args.engine, args.records, args.output, args.shard_size)
    elif args.prepare is not None:
        if args.shard_size < 1:
            parser.error("分片大小必须为正")
        audited = [audit_record(args.engine, path) for path in args.records]
        result = {
            "audits": [r.report for r in audited],
            "prepared": prepare(
                args.engine, args.records, args.prepare, args.shard_size, audited=audited
            ),
        }
    else:
        result = []
        with Client(args.engine) as client:
            for path in args.records:
                client.send({"op": "audit", "record": str(path.resolve()), "encode": False})
                result.append(client.receive())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
