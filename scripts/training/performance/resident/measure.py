"""固定模型筛选与自然局验收；所有原始趋势写盘，采样与审核串行记账。"""

# 在导入PyTorch前开始总账；这些延迟导入不改变模型执行。
# ruff: noqa: E402

import argparse
import ctypes
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

PROCESS_STARTED = time.perf_counter()

import torch

from haojie_training.native.execution import PolicyInference
from haojie_training.native.pipeline import BatchJobs
from haojie_training.native.pool import Pool
from haojie_training.native.resident import resident, task_identity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernel"))
from usage import Usage, read  # noqa: E402


class Monitor:
    """两秒趋势；CPU累积时间按差分展示，不与并发墙钟相加。GPU是整卡值。"""

    def __init__(self, path):
        self.path = path
        self.stop = threading.Event()

    def __enter__(self):
        self.started = time.perf_counter()
        self.usage = Usage().start()

        def worker():
            with self.path.open("x", encoding="utf-8") as stream:
                while not self.stop.is_set():
                    row = {
                        "seconds": time.perf_counter() - self.started,
                        "python_cpu_seconds": time.process_time(),
                        "python_rss_bytes": read()[1],
                    }
                    with self.usage.lock:
                        children = [read(p) for p in self.usage.processes.values()]
                        row.update(
                            engine_cpu_seconds=self.usage.child_cpu
                            + sum(r[0] for r in children),
                            engine_rss_bytes=sum(r[1] for r in children),
                            engine_processes=len(children),
                        )
                    if os.name == "nt":
                        from ctypes import wintypes

                        class MemoryStatus(ctypes.Structure):
                            _fields_ = [
                                ("length", wintypes.DWORD),
                                ("load", wintypes.DWORD),
                            ] + [
                                (name, ctypes.c_ulonglong)
                                for name in (
                                    "total",
                                    "available",
                                    "page_total",
                                    "page_available",
                                    "virtual_total",
                                    "virtual_available",
                                    "extended",
                                )
                            ]

                        memory = MemoryStatus()
                        memory.length = ctypes.sizeof(memory)
                        ctypes.windll.kernel32.GlobalMemoryStatusEx(
                            ctypes.byref(memory)
                        )
                        row.update(
                            system_memory_load=memory.load,
                            system_available_bytes=memory.available,
                        )

                        times = [wintypes.FILETIME() for _ in range(3)]
                        ctypes.windll.kernel32.GetSystemTimes(
                            *(ctypes.byref(t) for t in times)
                        )
                        idle, kernel, user = [
                            (t.dwHighDateTime << 32 | t.dwLowDateTime) / 1e7
                            for t in times
                        ]
                        row.update(
                            system_cpu_busy_seconds=kernel + user - idle,
                            system_cpu_total_seconds=kernel + user,
                        )
                    try:
                        result = subprocess.run(
                            [
                                "nvidia-smi",
                                "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,memory.used,power.draw,clocks.sm",
                                "--format=csv,noheader,nounits",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=5,
                            creationflags=subprocess.CREATE_NO_WINDOW
                            if os.name == "nt"
                            else 0,
                        )
                        row["gpu"] = result.stdout.strip()
                        if result.returncode:
                            row["gpu_error"] = result.stderr.strip()
                    except Exception as error:
                        row["gpu_error"] = repr(error)
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    self.stop.wait(2)

        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.path.with_suffix(".usage.json").write_text(
            json.dumps(self.usage.stop(), indent=2), encoding="utf-8"
        )


def screen(args):
    args.output.mkdir(parents=True, exist_ok=False)
    # 单批参照为903de3c完整Python包，不与上一轮几秒测量拼接。
    spec = importlib.util.spec_from_file_location(
        "resident_reference",
        args.reference / "__init__.py",
        submodule_search_locations=[str(args.reference)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from resident_reference.native.pipeline import sample as old_sample

    starts = [task_identity(202610041, i)["start"] for i in range(64)]
    (args.output / "starts.json").write_text(json.dumps(starts), encoding="utf-8")
    reports = []
    with Monitor(args.output / "hardware.jsonl"):
        begin = time.perf_counter()
        old = []
        for offset in (0, 8):
            old.append(
                old_sample(
                    args.engine,
                    args.checkpoint,
                    starts[offset : offset + 8],
                    args.output / f"old-{offset}",
                    48,
                    1000,
                    "cuda",
                    202610041 + offset,
                )
            )
        reports.append(
            {
                "case": "old-two-batches-eight",
                "seconds": time.perf_counter() - begin,
                "commands": sum(
                    g["outcome"]["commands"] for r in old for g in r["games"]
                ),
                "hashes": [g["finalHash"] for r in old for g in r["games"]],
            }
        )
        model = model_hash = None
        for name, environments, count in (
            ("resident-eight-control", 8, 16),
            ("screen-eight", 8, 64),
            ("screen-sixteen", 16, 64),
            ("screen-thirtytwo", 32, 64),
        ):
            folder = args.output / name
            folder.mkdir()
            runner = (
                None
                if model is None
                else (PolicyInference(model, "cuda", "fp32"), model_hash)
            )
            pool = Pool(
                args.engine, args.checkpoint, environments, "cuda", inference=runner
            )
            model, model_hash = pool.inference.model, pool.model_hash
            jobs = BatchJobs(starts[:count], folder, 202610041, 48, 1000)
            row = pool.run(jobs)
            row.update(
                case=name,
                commands=sum(g["outcome"]["commands"] for g in jobs.results),
                hashes=[g["finalHash"] for g in jobs.results],
            )
            if name == "resident-eight-control":
                assert row["hashes"] == reports[0]["hashes"]
            elif name != "screen-eight":
                assert row["hashes"] == reports[2]["hashes"]
            row["commands_per_second"] = row["commands"] / row["seconds"]
            reports.append(row)
            (args.output / "report.json").write_text(
                json.dumps(reports, indent=2), encoding="utf-8"
            )
            print(
                json.dumps(
                    {k: v for k, v in row.items() if k not in ("hashes", "batch_sizes")}
                ),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("op", choices=["screen", "formal", "audit"])
    for key in ("engine", "checkpoint", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--environments", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.op == "screen":
        screen(args)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        begin = PROCESS_STARTED
        with Monitor(args.output.with_suffix(".hardware.jsonl")):
            if args.op == "formal":
                launch_offset = time.perf_counter() - PROCESS_STARTED
                result = resident(
                    args.engine,
                    args.checkpoint,
                    args.output,
                    environments=args.environments,
                    seconds=3600 - launch_offset,
                    drain_seconds=120,
                    target=64,
                    seed=2026100407,
                    commands=20000,
                    plies=1000,
                    device="cuda",
                )
                result["launch_offset_seconds"] = launch_offset
            else:
                from haojie_training.native.stream import prepare_pool

                result = prepare_pool(args.engine, args.source, args.output, 32)
        result["wrapper_seconds"] = time.perf_counter() - begin
        args.output.with_suffix(".total.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
