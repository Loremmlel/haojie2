"""预定并发容量扫描；各档使用独立进程，窗口统计在保存之前冻结。"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from haojie_training.online.session import Session, Settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--concurrency", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.concurrency is None:
        for concurrency in (1, 16, 64, 128):
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--output",
                    str(args.output / str(concurrency)),
                    "--concurrency",
                    str(concurrency),
                    "--seconds",
                    str(args.seconds),
                ],
                check=True,
            )
        return
    session = Session(Settings(concurrency=args.concurrency, workers=2), "cuda")
    try:
        started, last = time.monotonic(), 0.0
        samples = []
        while time.monotonic() - started < args.seconds:
            session.tick()
            now = time.monotonic()
            if now - last >= 5:
                samples.append(session.status())
                last = now
        window = session.status()
        (args.output / "window.json").write_text(json.dumps(window, indent=2), encoding="utf-8")
        (args.output / "samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
        print(json.dumps({"concurrency": args.concurrency, **window}), flush=True)
    finally:
        session.close()


if __name__ == "__main__":
    main()
