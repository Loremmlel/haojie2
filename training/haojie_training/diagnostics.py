"""复用正式控制器测量自然终局、更新与保存；可选采集主线程 CPU 和短 CUDA 时间线。"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

from .console.config.resources import _processes, hardware, monitor
from .console.controller import Controller
from .console.sampling import MemoryJobs
from .console.storage import Store
from .data import synthetic_batch
from .native import client as client_module
from .native.client import Client
from .native.execution import PolicyInference
from .native.packed import PackedPolicy
from .native.pool import Pool
from .runtime import autocast


class MeasuredController(Controller):
    def _loop(self):
        # 诊断线程直接调用正式阶段；HTTP 后台循环不重复拥有同一模型。
        pass

    def _progress(self, metrics):
        self.last_sampling_snapshot = metrics
        if hasattr(self, "measurement_start"):
            reading = {
                "seconds": time.perf_counter() - self.measurement_start,
                "games": self.counts["games"],
                "retained_samples": self.counts["retained_samples"],
                "requests": metrics.get("inference_requests", 0),
                "mean_batch": metrics.get("mean_batch", 0),
                "cpu_cores": self.telemetry.get("cpu_cores"),
                "gpu_utilization": self.telemetry.get("gpu_utilization"),
            }
            self.measurements.append(reading)
            print(json.dumps(reading), flush=True)
        return super()._progress(metrics)


@contextmanager
def profiling(mode, output, report):
    if mode == "none":
        yield
        return
    owner = threading.get_ident()
    stages = {}
    trace = None
    if mode == "cuda":
        trace = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=32, repeat=1),
            on_trace_ready=lambda p: p.export_chrome_trace(str(output / "trace.json")),
        )
    with ExitStack() as stack:
        if trace:
            stack.enter_context(trace)

        def instrument(cls, name):
            original = getattr(cls, name)
            label = f"{cls.__name__}.{name}"

            def measured(*args, **kwargs):
                label_thread = ("main:" if threading.get_ident() == owner else "readers:") + label
                began, process = time.perf_counter(), time.thread_time()
                try:
                    if trace:
                        with torch.profiler.record_function(label):
                            return original(*args, **kwargs)
                    return original(*args, **kwargs)
                finally:
                    row = stages.setdefault(label_thread, {"calls": 0, "wall": 0, "cpu": 0})
                    row["calls"] += 1
                    row["wall"] += time.perf_counter() - began
                    row["cpu"] += time.thread_time() - process
                    if trace and cls is PolicyInference and name == "finish":
                        trace.step()

            stack.enter_context(patch.object(cls, name, measured))

        for cls, names in (
            (Pool, ("run",)),
            (PackedPolicy, ("batch", "forward")),
            (PolicyInference, ("submit", "finish")),
            (Client, ("send",)),
            (MemoryJobs, ("example", "infer")),
        ):
            for name in names:
                instrument(cls, name)
        instrument(client_module, "decode")
        instrument(client_module, "send_many")
        try:
            yield
        finally:
            report["profile_stages_inclusive"] = stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--environments", type=int, default=30)
    parser.add_argument("--host-workers", type=int, default=0)
    parser.add_argument("--inference-batch-size", type=int, default=24)
    parser.add_argument("--batch-wait-ms", type=float, default=0)
    parser.add_argument("--sampling-precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--deadline", type=float, default=600)
    parser.add_argument("--seconds", type=float, help="固定时长测量，替代终局数量和超时时限")
    parser.add_argument("--switch-ms", type=float, help="当前进程的 Python 线程切换间隔")
    parser.add_argument("--profile", choices=("none", "cpu", "cuda"), default="none")
    args = parser.parse_args()
    if min(args.games, args.deadline) <= 0 or args.updates < 0 or not 0 <= args.host_workers <= 16:
        parser.error("终局和时限须为正数，更新数须非负，输入进程数须为 0 至 16")
    if args.seconds is not None and args.seconds <= 0:
        parser.error("固定测量时长须为正数")
    if args.switch_ms is not None:
        if not 0 < args.switch_ms <= 100:
            parser.error("线程切换间隔须在 0 至 100 毫秒之间")
        sys.setswitchinterval(args.switch_ms / 1000)
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["HAOJIE_HOST_WORKERS"] = str(args.host_workers)
    controller = MeasuredController(
        args.engine, Store(args.output / "training"), device=args.device, threads=1
    )
    controller.thread.join()
    controller.config.update(
        environments=args.environments,
        inference_batch_size=args.inference_batch_size,
        batch_wait_ms=args.batch_wait_ms,
        sampling_precision=args.sampling_precision,
        games_per_round=48,
        updates_per_round=args.updates,
        seed=args.seed,
        history_percent=0,
        eval_games=100000,
        pool_mib=2048,
    )
    controller.want_run = True
    report = {
        "format": "haojie-throughput-diagnostics-v1",
        "profile": args.profile,
        "complete": False,
        "hardware": hardware(),
        "torch": torch.__version__,
        "python_switch_ms": sys.getswitchinterval() * 1000,
        "engine_sha256": hashlib.sha256(args.engine.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(
            b"".join(
                str(path.relative_to(Path(__file__).parent)).encode()
                + path.read_bytes().replace(b"\r\n", b"\n")
                for path in sorted(Path(__file__).parent.rglob("*"))
                if path.suffix in {".py", ".cpp"}
            )
        ).hexdigest(),
        "options": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    timer = None
    limit_reached = threading.Event()
    monitoring = None
    try:
        controller._initialize()
        report["initial_model"] = controller.version
        report["model_config"] = asdict(controller.trainer.model.config)
        report["config"] = dict(controller.config)
        controller.trainer.model.eval()
        warmup = synthetic_batch(controller.trainer.model.config, 8, 64, 24)
        warmup = {key: value.to(controller.device) for key, value in warmup.items()}
        with torch.no_grad(), autocast(controller.device, args.sampling_precision):
            for _ in range(3):
                controller.trainer.model.policy_logits(warmup)
        if controller.device.type == "cuda":
            torch.cuda.synchronize()
        controller.pool = controller._new_pool(args.environments)
        report["startup_seconds"] = controller.pool.startup_seconds
        import psutil

        process = psutil.Process()
        before = _processes(process)
        before_times = process.cpu_times()
        waited_before = sum(
            getattr(before_times, key, 0) for key in ("children_user", "children_system")
        )
        began, cpu = time.perf_counter(), time.process_time()
        controller.measurement_start, controller.measurements = began, []
        monitoring = threading.Thread(target=monitor, args=(controller,), daemon=True)
        monitoring.start()

        def end_measurement():
            limit_reached.set()
            controller.cancel.set()

        timer = threading.Timer(args.seconds or args.deadline, end_measurement)
        timer.start()
        windows = []
        with profiling(args.profile, args.output, report):
            while (
                args.seconds is not None or controller.counts["games"] < args.games
            ) and not controller.cancel.is_set():
                window = time.perf_counter()
                controller._sample()
                if controller.cancel.is_set():
                    break
                sampled = time.perf_counter()
                controller._update()
                windows.append(
                    {
                        "sampling_seconds": sampled - window,
                        "update_seconds": time.perf_counter() - sampled,
                        "games": controller.counts["games"],
                        "retained_samples": controller.counts["retained_samples"],
                        "updates": controller.trainer.updates,
                    }
                )
                print(json.dumps(windows[-1]), flush=True)
            saving = time.perf_counter()
            controller._save()
        elapsed = time.perf_counter() - began
        saved_seconds = time.perf_counter() - saving
        after = _processes(process)
        after_times = process.cpu_times()
        waited_after = sum(
            getattr(after_times, key, 0) for key in ("children_user", "children_system")
        )
        report.update(
            complete=bool(
                (args.seconds is not None and limit_reached.is_set())
                or (controller.counts["games"] >= args.games and not controller.cancel.is_set())
            ),
            stop_reason="duration"
            if args.seconds is not None
            else "deadline"
            if limit_reached.is_set()
            else "terminal_target",
            seconds=elapsed,
            parent_cpu_seconds=time.process_time() - cpu,
            python_switch_ms_after=sys.getswitchinterval() * 1000,
            cpu_tree_seconds=max(0, waited_after - waited_before)
            + sum(
                max(0, value["cpu"] - before.get(key, {"cpu": 0})["cpu"])
                for key, value in after.items()
            ),
            cpu_tree_includes_reaped=hasattr(after_times, "children_user"),
            windows=windows,
            measurements=controller.measurements,
            save_seconds=saved_seconds,
            counts=dict(controller.counts),
            updates=controller.trainer.updates,
            consumed_samples=controller.samples.consumed,
            terminal_samples_per_second=controller.counts["retained_samples"] / elapsed,
            terminal_games_per_second=controller.counts["games"] / elapsed,
            runtime=controller.pool.metrics(),
            last_sampling_snapshot=getattr(controller, "last_sampling_snapshot", None),
        )
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if timer:
            timer.cancel()
        controller._close_pool()
        report["counts_after_close"] = dict(controller.counts)
        controller.close()
        if monitoring:
            monitoring.join(timeout=3)
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("complete", "seconds", "terminal_samples_per_second", "updates")
            }
        )
    )
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
