"""Linux原生成本账；阶段二顺序计时，不与编译/验收并行。完整命令预算由调用方预先冻结。"""

import argparse
import hashlib
import json
import os
import platform
import resource
import signal
import threading
import time
from pathlib import Path

import torch

from haojie_training.data import load_dataset, sample_indices, select_batch
from haojie_training.evaluate import evaluate
from haojie_training.native.client import Client
from haojie_training.native.pipeline import model_from, prepare, sample
from haojie_training.runtime import Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--starts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commands", type=int, default=12000)
    parser.add_argument("--plies", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    starts = json.loads(args.starts.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    signal.signal(
        signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("1800秒总预算"))
    )
    signal.alarm(1800)

    def memory_limit(*_):
        raise MemoryError("本组RSS超过6GiB保护线")

    signal.signal(signal.SIGUSR1, memory_limit)
    before = [
        resource.getrusage(who)
        for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
    ]
    peak = [0]
    stopped = threading.Event()

    def memory():
        # 此脚本只在独立PID命名空间运行；采样整个命名空间RSS，不混入其他用户任务。
        while not stopped.is_set():
            total = 0
            for path in Path("/proc").glob("[0-9]*/status"):
                try:
                    for line in path.read_text().splitlines():
                        if line.startswith("VmRSS:"):
                            total += int(line.split()[1])
                except OSError:
                    pass
            peak[0] = max(peak[0], total)
            if total > 6 * 1024 * 1024:
                os.kill(os.getpid(), signal.SIGUSR1)
                return
            stopped.wait(0.1)

    threading.Thread(target=memory, daemon=True).start()
    started = time.perf_counter()
    reports = []
    records = []
    batches = [starts] if args.parallel else [[s] for s in starts]
    for i, batch in enumerate(batches):
        folder = args.output / f"sample-{i}"
        reports.append(
            sample(
                args.engine,
                args.checkpoint,
                batch,
                folder,
                args.commands,
                args.plies,
                seed=args.seed + (0 if args.parallel else i),
            )
        )
        records.extend(sorted(folder.glob("*.jsonl")))
    sampling_seconds = time.perf_counter() - started
    audit_start = time.perf_counter()
    with Client(args.engine) as client:
        for path in records:
            client.send({"op": "audit", "record": str(path.resolve()), "encode": False})
            client.receive()
    audit_seconds = time.perf_counter() - audit_start
    phases = {"sampling": sampling_seconds, "audit": audit_seconds}
    if args.prepare:
        t = time.perf_counter()
        prepare(args.engine, records, args.output / "data")
        phases["prepare"] = time.perf_counter() - t
        with Client(args.engine) as client:
            model, _ = model_from(args.checkpoint, client.ready)
        trainer = Trainer(model, torch.device("cpu"))
        t = time.perf_counter()
        dataset, _ = load_dataset(args.output / "data/train.pt", model.config)
        validation, _ = load_dataset(args.output / "data/validation.pt", model.config)
        phases["load"] = time.perf_counter() - t
        t = time.perf_counter()
        for step in range(2):
            indices = sample_indices(
                dataset, 4, torch.Generator().manual_seed(81 + step)
            )
            trainer.step(select_batch(dataset, indices))
        phases["optimizer"] = time.perf_counter() - t
        t = time.perf_counter()
        evaluation = evaluate(model, validation, torch.device("cpu"), "fp32", 4)
        phases["evaluate"] = time.perf_counter() - t
        assert trainer.health()["finite_optimizer"]
    else:
        evaluation = None
    stopped.set()
    after = [
        resource.getrusage(who)
        for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
    ]
    result = {
        "seconds": time.perf_counter() - started,
        "phases": phases,
        "reports": reports,
        "cpu_seconds": sum(
            b.ru_utime + b.ru_stime - a.ru_utime - a.ru_stime
            for a, b in zip(before, after)
        ),
        "namespace_peak_rss_kib_100ms": peak[0],
        "max_process_rss_kib": max(r.ru_maxrss for r in after),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "engine_sha256": hashlib.sha256(args.engine.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "starts_sha256": hashlib.sha256(args.starts.read_bytes()).hexdigest(),
        "records": [hashlib.sha256(p.read_bytes()).hexdigest() for p in records],
        "validation_examples": evaluation["examples"] if evaluation else None,
    }
    (args.output / "measurement.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "reports"}, indent=2))


if __name__ == "__main__":
    main()
