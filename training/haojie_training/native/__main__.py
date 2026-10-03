"""独立原生训练入口；学习/续训仍使用 haojie_training.train。"""

import argparse
import json
from pathlib import Path

import torch

from .client import Client
from .pipeline import initialize, prepare, sample


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
    sampling.add_argument("--device", choices=["cpu", "cuda", "xpu"], default="cpu")
    sampling.add_argument("--seed", type=int, default=20261003)
    audit = sub.add_parser("audit")
    audit.add_argument("records", nargs="+", type=Path)
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
        )
    elif args.op == "prepare":
        if args.shard_size < 1:
            parser.error("分片大小必须为正")
        result = prepare(args.engine, args.records, args.output, args.shard_size)
    else:
        result = []
        with Client(args.engine) as client:
            for path in args.records:
                client.send({"op": "audit", "record": str(path.resolve()), "encode": False})
                result.append(client.receive())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
