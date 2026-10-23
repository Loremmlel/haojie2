"""冻结权重导出和双座正式对局；窗口结束保存未完成工作。"""

import argparse
import json
import time
from pathlib import Path

from ..session import Session
from . import export, frozen_session, start_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    exporting = commands.add_parser("export")
    exporting.add_argument("checkpoint", type=Path)
    exporting.add_argument("output", type=Path)
    evaluating = commands.add_parser("evaluate")
    evaluating.add_argument("--policy", type=Path)
    evaluating.add_argument("--opponent", type=Path)
    evaluating.add_argument("--resume", type=Path)
    evaluating.add_argument("--directory", type=Path, required=True)
    evaluating.add_argument("--seed", type=int, default=70001)
    evaluating.add_argument("--pairs", type=int, default=8)
    evaluating.add_argument("--seconds", type=float, default=600)
    evaluating.add_argument("--workers", type=int, default=2)
    evaluating.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.action == "export":
        export(args.checkpoint, args.output)
        return
    if args.pairs < 1 or args.pairs > 64 or args.seconds <= 0:
        raise ValueError("评测要求 1–64 个种子对及正窗口时间")
    args.directory.mkdir(parents=True, exist_ok=False)
    session = None
    outcomes = None
    try:
        if args.resume:
            session = Session.restore(args.resume, args.device)
            if session.training:
                raise ValueError("评测只能恢复冻结会话")
        else:
            if not args.policy or not args.opponent:
                raise ValueError("评测需要双方冻结权重")
            session = frozen_session(
                args.policy, args.opponent, device=args.device, workers=args.workers
            )
            start_pairs(session, range(args.seed, args.seed + args.pairs))
        outcomes = (args.directory / "completed.jsonl").open("x", encoding="utf-8", buffering=1)
        session.on_result = lambda result: outcomes.write(json.dumps(result) + "\n")
        began = time.monotonic()
        while session.games and time.monotonic() - began < args.seconds:
            session.tick(admit=False)
        result = session.status()
        result["policy"] = str(args.policy)
        result["opponent"] = str(args.opponent)
        result["complete"] = not session.games
        # 统计先冻结；未完成对局保存到磁盘，继续执行不混入本窗口。
        (args.directory / "window.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        session.save(args.directory / "checkpoint.pt")
        print(json.dumps(result), flush=True)
    finally:
        if session:
            session.close()
        if outcomes:
            outcomes.close()


if __name__ == "__main__":
    main()
