"""同机自然新局吞吐：旧代码只从外部冻结目录加载，不保存测试权重。"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def baseline(args):
    if args.source is None or args.engine is None:
        raise ValueError("旧基线需要冻结源码目录和旧引擎")
    sys.path.insert(0, str(args.source.resolve() / "training"))
    from haojie_training import diagnostics

    snapshots = []
    command_counts = {}

    class Measured(diagnostics.MeasuredController):
        def _save(self):
            # 仅吞吐实验省略权重落盘；生产控制器源码保持冻结。
            pass

        def _progress(self, metrics):
            elapsed = time.perf_counter() - self.measurement_start
            if elapsed <= args.seconds:
                for row in metrics.get("active", []):
                    key = row["task"]
                    command_counts[key] = max(command_counts.get(key, 0), row["commands"] or 0)
                snapshots[:] = [
                    dict(
                        seconds=elapsed,
                        commands=sum(command_counts.values()),
                        completed=self.counts["games"],
                        consumed_terminal_samples=self.samples.consumed,
                        optimizer_steps=self.trainer.updates,
                        runtime=metrics,
                        telemetry=self.telemetry,
                    )
                ]
            self.last_sampling_snapshot = metrics
            return diagnostics.Controller._progress(self, metrics)

    diagnostics.MeasuredController = Measured
    starts = args.output / "starts.json"
    starts.write_text(
        json.dumps(
            [
                dict(rules="classic", seed=(args.seed + i * 2654435761) & 0xFFFFFFFF)
                for i in range(1, args.concurrency + 1)
            ]
        ),
        encoding="utf-8",
    )
    sys.argv = [
        sys.argv[0],
        "--engine",
        str(args.engine.resolve()),
        "--output",
        str(args.output.resolve() / "details"),
        "--seconds",
        str(args.seconds),
        "--environments",
        str(args.concurrency),
        "--inference-batch-size",
        "64",
        "--host-workers",
        "1",
        "--sampling-backend",
        "pipeline",
        "--history-percent",
        "0",
        "--seed",
        str(args.seed),
        "--starts",
        str(starts),
    ]
    diagnostics.main()
    if not snapshots:
        raise RuntimeError("基线窗口没有进度快照")
    result = snapshots[0]
    result["command_accounting"] = "窗口末前最后一次状态；每局已观察命令下界，排除关闭排空"
    return result


def current(args):
    from haojie_training.online.session import Session, Settings

    session = Session(Settings(concurrency=args.concurrency, workers=2, seed=args.seed), "cuda")
    try:
        started, last = time.monotonic(), 0.0
        readings = []
        while time.monotonic() - started < args.seconds:
            session.tick()
            now = time.monotonic()
            if now - last >= 5:
                readings.append(session.status())
                last = now
        result = session.status()
        result["readings"] = readings
        result["command_accounting"] = "窗口结束已收取原生帧的命令计数，排除关闭排空"
        return result
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["old", "new"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = baseline(args) if args.mode == "old" else current(args)
    result["mode"] = args.mode
    package = Path(sys.modules["haojie_training"].__file__).parent
    result["source_sha256"] = hashlib.sha256(
        b"".join(
            str(p.relative_to(package)).encode() + p.read_bytes().replace(b"\r\n", b"\n")
            for p in sorted(package.rglob("*.py"))
        )
    ).hexdigest()
    (args.output / "window.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("mode", "seconds", "commands", "completed")}))


if __name__ == "__main__":
    main()
