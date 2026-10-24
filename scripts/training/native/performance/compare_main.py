"""固定 main 与原生 Transformer 的独立进程短测；只做诊断记账，不修改训练算法。"""

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from unittest.mock import patch


def write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main_window(args):
    import torch
    import haojie_training.console.controller as module
    from haojie_training.console.sampling import MemoryJobs
    from haojie_training.console.storage import Store
    from haojie_training.native.aten import operator

    if operator() is None:
        raise RuntimeError("main 必须启用其正式 C++ ATen 路径")

    class Jobs(MemoryJobs):
        def finish(self, task, message):
            self.c.closed_commands += message["outcome"]["commands"]
            return super().finish(task, message)

    class Controller(module.Controller):
        def _loop(self):
            pass

        def _sample(self):
            with patch.object(module, "MemoryJobs", Jobs):
                return super()._sample()

    starts = [
        {"rules": args.rules, "seed": (1 + i * 2654435761) % 2**32}
        for i in range(1, 16385)
    ]
    c = Controller(
        args.engine,
        Store(args.directory / "training"),
        device="cuda",
        threads=1,
        starts=starts,
    )
    c.thread.join()
    c.closed_commands = 0
    c.config.update(
        environments=args.concurrency,
        inference_batch_size=args.inference_batch,
        history_percent=0,
        classic_percent=100 if args.rules == "classic" else 0,
        seed=1,
        sampling_precision="bf16",
        updates_per_round=0,
        pool_mib=2048,
        eval_games=100000,
        save_games=0,
    )
    c.want_run = True
    c._initialize()
    c.pool = c._new_pool(args.concurrency)
    runtime = c.pool.metrics()
    if (
        runtime["dispatch_backend"] != "aten-cpp"
        or runtime["policy_layout"] != "compact-candidates"
    ):
        raise RuntimeError("main 没有使用正式原帧/C++/紧凑候选路径")
    ledger = (0, 0)
    original = c.trainer.step_batches

    def step(batches):
        nonlocal ledger
        before = c.trainer.updates
        result = original(batches)
        if c.trainer.updates > before:
            ledger = (
                c.trainer.updates,
                ledger[1] + sum(b["value"].shape[0] for b in batches),
            )
        return result

    c.trainer.step_batches = step
    began = time.perf_counter()
    report = {
        "model_parameters": sum(p.numel() for p in c.trainer.model.parameters()),
        "parameter_dtype": str(next(c.trainer.model.parameters()).dtype),
        "config": copy.deepcopy(c.config),
        "torch": torch.__version__,
    }

    def freeze():
        metrics = copy.deepcopy(c.pool.metrics())
        active = metrics["active"]
        # 消费只计已返回成功优化步；蓄水池提前抽出的在途样本单列。
        updates, consumed = ledger
        window = {
            "seconds": time.perf_counter() - began,
            "commands": c.closed_commands + sum(x["commands"] or 0 for x in active),
            "completed_games": c.counts["games"],
            "updates": updates,
            "consumed": consumed,
            "drawn_samples": c.samples.consumed,
            "produced": c.counts["retained_samples"],
            "pool": copy.deepcopy(c.samples.snapshot()),
            "incomplete_seen": sum(t["trajectory"].seen for t in c.pool.slots.values()),
            "active": len(active),
            "admitted": c.counts["tasks"],
            "truncated": c.counts["truncated"],
            "runtime": metrics,
        }
        write(args.directory / "window.json", window)
        report["window"] = window
        c.cancel.set()

    timer = threading.Timer(args.seconds, freeze)
    timer.start()
    try:
        while not c.cancel.is_set():
            c._sample()
            if not c.cancel.is_set():
                c._update()
        c._save()
        report["after_window"] = {
            "counts": copy.deepcopy(c.counts),
            "updates": c.trainer.updates,
            "drawn_samples": c.samples.consumed,
            "committed_consumed": ledger[1],
        }
    finally:
        timer.cancel()
        timer.join()
        c._close_pool()
        c.close()
        report["counts_after_close"] = copy.deepcopy(c.counts)
        write(args.directory / "main.json", report)
    if "window" not in report or report["window"]["admitted"] >= len(starts):
        raise RuntimeError("测量窗口未完成或起始种子表耗尽")


def run(args):
    import psutil

    root = Path(__file__).resolve().parents[4]
    output = args.output.resolve()
    setup = (args.setup or output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = output / "runs"
    runs.mkdir(exist_ok=False)
    source = setup / "main-source"
    engine = setup / "native-target/release/haojie-engine.exe"
    native = output / "native-bin"
    native.mkdir(exist_ok=False)
    for path in (
        root / "artifacts/cpp-training-build/Release/haojie-train.exe",
        root / "artifacts/native-target/release/haojie_native.dll",
    ):
        shutil.copy2(path, native / path.name)
    manifest = {
        "main_commit": subprocess.check_output(
            ["git", "rev-parse", "main"], text=True
        ).strip(),
        "native_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "main_archive_sha256": sha(setup / "main-source.tar"),
        "main_engine_sha256": sha(engine),
        "native_exe_sha256": sha(native / "haojie-train.exe"),
        "main_aten_sha256": {
            str(p.relative_to(setup)): sha(p)
            for p in (setup / "torch-cache").rglob("*.pyd")
        },
        "harness_sha256": sha(Path(__file__).resolve()),
        "native_rules_sha256": sha(native / "haojie_native.dll"),
        "seconds": args.seconds,
        "concurrency": args.counts,
        "repeats": args.repeats,
        "rules": args.rules,
        "seed": 1,
        "inference_batch": args.inference_batch,
        "native_learning_batch": args.native_batch,
        "game_seed": "(1 + game_id * 2654435761) mod 2^32",
        "main_source_hashes": {
            str(p.relative_to(source)): sha(p)
            for p in sorted((source / "training/haojie_training").rglob("*"))
            if p.suffix in {".py", ".cpp"}
        },
        "native_source_hashes": {
            str(p.relative_to(root)): sha(p)
            for p in sorted((root / "training/native/src").rglob("*"))
            if p.is_file()
        },
    }
    write(output / "manifest.json", manifest)
    hidden = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    results = []
    for repeat in range(1, args.repeats + 1):
        order = [
            name
            for name in (("main", "native") if repeat % 2 else ("native", "main"))
            if name in args.implementations
        ]
        for count in args.counts:
            for implementation in order:
                name = f"{implementation}-{count}-{repeat}"
                directory = runs / name
                directory.mkdir()
                env = dict(
                    os.environ,
                    HAOJIE_HOST_WORKERS="0",
                    HAOJIE_ATEN="1",
                    PYTHONPATH=str(source / "training"),
                    TORCH_EXTENSIONS_DIR=str(setup / "torch-cache"),
                )
                if implementation == "main":
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--child",
                        "--engine",
                        str(engine),
                        "--directory",
                        str(directory),
                        "--concurrency",
                        str(count),
                        "--seconds",
                        str(args.seconds),
                        "--rules",
                        args.rules,
                        "--inference-batch",
                        str(args.inference_batch),
                    ]
                else:
                    command = [
                        str(native / "haojie-train.exe"),
                        "run",
                        "--directory",
                        str(directory / "session"),
                        "--device",
                        "cuda",
                        "--concurrency",
                        str(count),
                        "--workers",
                        "8",
                        "--seconds",
                        str(args.seconds),
                        "--inference-batch",
                        str(args.inference_batch),
                        "--batch",
                        str(args.native_batch),
                        "--save-seconds",
                        "0",
                    ]
                    if args.rules == "shrine":
                        command.append("--shrine")
                    env["PATH"] = (
                        str(native)
                        + os.pathsep
                        + str(root / "training/.venv/Lib/site-packages/torch/lib")
                        + os.pathsep
                        + env["PATH"]
                    )
                gpu_output = (directory / "gpu.csv").open("w", encoding="utf-8")
                gpu = subprocess.Popen(
                    [
                        "nvidia-smi",
                        "--query-gpu=timestamp,utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                        "-lms",
                        "200",
                    ],
                    stdout=gpu_output,
                    stderr=subprocess.STDOUT,
                    **hidden,
                )
                observed, peak = {}, 0
                began = time.perf_counter()
                with (
                    (directory / "stdout.log").open("w", encoding="utf-8") as stdout,
                    (directory / "stderr.log").open("w", encoding="utf-8") as stderr,
                ):
                    process = subprocess.Popen(
                        command,
                        env=env,
                        cwd=root,
                        stdout=stdout,
                        stderr=stderr,
                        **hidden,
                    )
                    try:
                        while process.poll() is None:
                            try:
                                owner = psutil.Process(process.pid)
                                members = [owner, *owner.children(recursive=True)]
                                rss = 0
                                for member in members:
                                    try:
                                        cpu = member.cpu_times()
                                        observed[(member.pid, member.create_time())] = (
                                            cpu.user + cpu.system
                                        )
                                        rss += member.memory_info().rss
                                    except psutil.Error:
                                        pass
                                peak = max(peak, rss)
                            except psutil.Error:
                                pass
                            time.sleep(0.2)
                        wall = time.perf_counter() - began
                    finally:
                        gpu.terminate()
                        gpu.wait(timeout=10)
                        gpu_output.close()
                if process.returncode:
                    raise RuntimeError(f"{name} 失败，见 {directory / 'stderr.log'}")
                window_path = directory / (
                    "session/window.json"
                    if implementation == "native"
                    else "window.json"
                )
                window = json.loads(window_path.read_text(encoding="utf-8"))
                with (directory / "gpu.csv").open(encoding="utf-8") as file:
                    readings = list(csv.reader(file))
                utilization = [
                    float(row[1])
                    for row in readings
                    if len(row) == 3 and row[1].strip().isdigit()
                ]
                result = {
                    "name": name,
                    "implementation": implementation,
                    "concurrency": count,
                    "repeat": repeat,
                    "process_wall_seconds": wall,
                    "peak_process_tree_rss": peak,
                    "sampled_process_tree_cpu_seconds": sum(observed.values()),
                    "gpu_process_mean": sum(utilization) / len(utilization)
                    if utilization
                    else None,
                    "gpu_process_peak": max(utilization) if utilization else None,
                    "window": window,
                }
                write(directory / "measurement.json", result)
                results.append(result)
                write(output / "results.json", results)
                print(
                    json.dumps(
                        {
                            k: result[k]
                            for k in (
                                "name",
                                "process_wall_seconds",
                                "gpu_process_mean",
                            )
                        }
                        | {
                            k: window[k]
                            for k in (
                                "seconds",
                                "commands",
                                "consumed",
                                "updates",
                                "completed_games",
                            )
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/main-comparison-20261024")
    )
    parser.add_argument("--setup", type=Path)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--native-batch", type=int, default=32)
    parser.add_argument("--inference-batch", type=int, default=12)
    parser.add_argument("--counts", type=int, nargs="+", default=[12])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=["main", "native"],
        default=["main", "native"],
    )
    parser.add_argument("--rules", choices=["classic", "shrine"], default="shrine")
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--concurrency", type=int)
    parsed = parser.parse_args()
    os.environ.setdefault("HAOJIE_HOST_WORKERS", "0")
    if parsed.child:
        main_window(parsed)
    else:
        run(parsed)
