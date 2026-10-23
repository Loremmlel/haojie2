"""复用正式控制器测量自然终局、更新与保存；可选采集主线程 CPU 和短 CUDA 时间线。"""

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import threading
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

from .console.config.resources import _cpu_total, _processes, hardware, monitor
from .console.controller import Controller
from .console.sampling import MemoryJobs
from .console.storage import Store
from .data import synthetic_batch
from .native import client as client_module
from .native.client import Client
from .native.execution import ForwardWorkers, PolicyInference
from .native.packed import PackedPolicy
from .native.pool import Pool
from .native.transport import ReadLoop


class MeasuredJobs(MemoryJobs):
    """仅在诊断入口记录结案分账，原采样、收益和蓄水处理仍由父类完成。"""

    def record(self, task, status, retained=0, returns=None):
        self.c.closed_tasks.append(
            {
                "task": task["id"],
                "slot": task.get("slot"),
                "pid": task.get("pid"),
                "rules": task["context"]["rules"],
                "status": status,
                "requests": task["requests"],
                "elapsed_seconds": time.perf_counter() - task["began"],
                "retained_samples": retained,
                "returns": returns,
            }
        )

    def finish(self, task, message):
        before = self.c.counts["retained_samples"]
        result = super().finish(task, message)
        terminal = message["outcome"]["terminated"]
        self.record(
            task,
            "terminal" if terminal else "truncated",
            self.c.counts["retained_samples"] - before,
            message["outcome"]["returns"] if terminal else None,
        )
        return result

    def unfinished(self, task):
        result = super().unfinished(task)
        self.record(task, "unknown")
        return result


class MeasuredController(Controller):
    def _loop(self):
        # 诊断线程直接调用正式阶段；HTTP 后台循环不重复拥有同一模型。
        pass

    def _sample(self):
        with patch.object(sys.modules[Controller.__module__], "MemoryJobs", MeasuredJobs):
            return super()._sample()

    def _update(self):
        if getattr(self, "fixed_work", False):
            self.completed_rows.extend(row for row, _, _ in self.samples.rows)
        return super()._update()

    def _progress(self, metrics):
        self.last_sampling_snapshot = metrics
        if hasattr(self, "measurement_start"):
            reading = {
                **(self.sampling_jobs.publication_metrics() if self.sampling_jobs else {}),
                "seconds": time.perf_counter() - self.measurement_start,
                "games": self.counts["games"],
                "retained_samples": self.counts["retained_samples"],
                "requests": metrics.get("inference_requests", 0),
                "mean_batch": metrics.get("mean_batch", 0),
                "mean_entities": metrics.get("mean_entities", 0),
                "cpu_cores": self.telemetry.get("cpu_cores"),
                "gpu_utilization": self.telemetry.get("gpu_utilization"),
                "environments": metrics.get("environments"),
                "target_environments": metrics.get("target_environments"),
                "retiring_environments": metrics.get("retiring_environments", 0),
                "starting_environments": metrics.get("starting_environments", 0),
                "discarded": self.counts["discarded"],
            }
            if getattr(self, "trace_resize", False):
                reading["active"] = metrics.get("active", [])
            self.measurements.append(reading)
            print(json.dumps(reading), flush=True)
        return super()._progress(metrics)


def sample_fingerprint(rows):
    """固定任务集按样本内容排序；计时结束后核对真实编码、选择、收益与来源。"""
    records = []
    for row in rows:
        metadata = {key: value for key, value in row.get("_meta", {}).items() if key != "entered"}
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
        for key, tensor in sorted(row.items()):
            if isinstance(tensor, torch.Tensor):
                value = tensor.detach().cpu().contiguous()
                digest.update(json.dumps((key, str(value.dtype), list(value.shape))).encode())
                digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        records.append(digest.digest())
    return hashlib.sha256(b"".join(sorted(records))).hexdigest()


@contextmanager
def profiling(mode, output, report, wait_batches=1):
    if mode == "none":
        yield
        return
    owner = threading.get_ident()
    stages = {}
    trace = None
    if mode == "cuda":
        trace = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            schedule=torch.profiler.schedule(wait=wait_batches, warmup=1, active=32, repeat=1),
            on_trace_ready=lambda p: p.export_chrome_trace(str(output / "trace.json")),
        )
    with ExitStack() as stack:
        if trace:
            stack.enter_context(trace)

        def instrument(cls, name):
            original = getattr(cls, name)
            label = f"{cls.__name__}.{name}"

            def measured(*args, **kwargs):
                label_thread = ("main:" if threading.get_ident() == owner else "workers:") + label
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
            (ForwardWorkers, ("run",)),
            (Client, ("send", "_read", "_publish")),
            (ReadLoop, ("run",)),
            (MemoryJobs, ("example", "infer")),
        ):
            for name in names:
                instrument(cls, name)
        instrument(client_module, "decode")
        instrument(client_module, "send_many")
        instrument(json, "loads")
        try:
            yield
        finally:
            report["profile_stages_inclusive"] = stages


def resize_point(value):
    """诊断计划采用经过秒数:目标并发，生产并发校验仍由控制器执行。"""
    try:
        seconds, count = value.split(":")
        seconds, count = float(seconds), int(count)
    except ValueError:
        raise argparse.ArgumentTypeError("并发计划须为秒数:环境数") from None
    if not math.isfinite(seconds) or seconds <= 0 or not 1 <= count <= 128:
        raise argparse.ArgumentTypeError("计划时间须为有限正数，环境数须为 1 至 128")
    return seconds, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--initialize-from", type=Path, help="只读加载固定兼容权重，重建优化器以比较执行路径"
    )
    parser.add_argument("--environments", type=int, default=30)
    parser.add_argument("--host-workers", type=int, default=0)
    parser.add_argument("--launch-workers", type=int, choices=(0, 1, 2))
    parser.add_argument("--inference-batch-size", type=int, default=24)
    parser.add_argument("--batch-wait-ms", type=float, default=0)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--finite", action="store_true", help="固定任务批次完成后再更新一次")
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--history-percent", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument(
        "--starts", type=Path, help="可重放起点 JSON 或 JSON.gz；省略时使用自然新局"
    )
    parser.add_argument("--start-indices", type=int, nargs="+", help="起点文件中的零基索引")
    parser.add_argument("--deadline", type=float, default=600)
    parser.add_argument("--seconds", type=float, help="固定时长测量，替代终局数量和超时时限")
    parser.add_argument(
        "--resize-at", type=resize_point, action="append", default=[], metavar="秒数:环境数"
    )
    parser.add_argument("--switch-ms", type=float, help="当前进程的 Python 线程切换间隔")
    parser.add_argument("--profile", choices=("none", "cpu", "cuda"), default="none")
    parser.add_argument(
        "--profile-wait-batches", type=int, default=1, help="CUDA 剖析前跳过的真实批数"
    )
    args = parser.parse_args()
    if min(args.games, args.deadline) <= 0 or args.updates < 0 or not 0 <= args.host_workers <= 16:
        parser.error("终局和时限须为正数，更新数须非负，输入进程数须为 0 至 16")
    if args.seconds is not None and args.seconds <= 0:
        parser.error("固定测量时长须为正数")
    if args.finite and args.seconds is not None:
        parser.error("固定任务批次与固定时长不能同时选择")
    if args.resize_at and (
        args.seconds is None
        or any(at >= args.seconds for at, _ in args.resize_at)
        or any(a[0] >= b[0] for a, b in zip(args.resize_at, args.resize_at[1:]))
    ):
        parser.error("并发计划须用于固定时长，按时间严格递增且早于测量终点")
    if args.profile_wait_batches < 0:
        parser.error("剖析前跳过的批数须非负")
    if not 0 <= args.history_percent <= 100:
        parser.error("历史对手比例须为 0 至 100")
    if args.start_indices is not None and args.starts is None:
        parser.error("选择起点索引需要 --starts")
    if args.switch_ms is not None:
        if not 0 < args.switch_ms <= 100:
            parser.error("线程切换间隔须在 0 至 100 毫秒之间")
        sys.setswitchinterval(args.switch_ms / 1000)
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["HAOJIE_HOST_WORKERS"] = str(args.host_workers)
    if args.launch_workers is not None:
        os.environ["HAOJIE_LAUNCH_WORKERS"] = str(args.launch_workers)
    start_bytes = args.starts.read_bytes() if args.starts else None
    starts = (
        json.loads(gzip.decompress(start_bytes) if args.starts.suffix == ".gz" else start_bytes)
        if args.starts
        else None
    )
    if args.start_indices is not None:
        if any(index < 0 or index >= len(starts) for index in args.start_indices):
            parser.error("起点索引超出文件范围")
        starts = [starts[index] for index in args.start_indices]
    controller = MeasuredController(
        args.engine,
        Store(args.output / "training"),
        device=args.device,
        threads=1,
        starts=starts,
        imported=args.initialize_from,
    )
    controller.thread.join()
    controller.continuous_selfplay = not args.finite
    controller.fixed_work, controller.completed_rows = args.finite, []
    controller.closed_tasks = []
    controller.trace_resize = bool(args.resize_at)
    controller.config.update(
        environments=args.environments,
        inference_batch_size=args.inference_batch_size,
        batch_wait_ms=args.batch_wait_ms,
        games_per_round=args.games if args.finite else 48,
        updates_per_round=args.updates,
        seed=args.seed,
        history_percent=args.history_percent,
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
        "initial_checkpoint_sha256": hashlib.sha256(args.initialize_from.read_bytes()).hexdigest()
        if args.initialize_from
        else None,
        "starts_sha256": hashlib.sha256(start_bytes).hexdigest() if start_bytes else None,
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
    resize_timers, resize_events = [], []
    report["resize_events"] = resize_events
    limit_reached = threading.Event()
    monitoring = None
    profile_scope = ExitStack()
    try:
        controller._initialize()
        report["initial_model"] = controller.version
        report["model_config"] = asdict(controller.trainer.model.config)
        report["config"] = dict(controller.config)
        controller.trainer.model.eval()
        warmup = synthetic_batch(controller.trainer.model.config, 8, 64, 24)
        warmup = {key: value.to(controller.device) for key, value in warmup.items()}
        with torch.no_grad():
            for _ in range(3):
                controller.trainer.model.policy_logits(warmup)
        if controller.device.type == "cuda":
            torch.cuda.synchronize()
        # Reader 在线程创建时绑定入口，必须先挂剖析，再建立进程池。
        profile_scope.enter_context(
            profiling(args.profile, args.output, report, args.profile_wait_batches)
        )
        controller.pool = controller._new_pool(args.environments)
        report["startup_seconds"] = controller.pool.startup_seconds
        import psutil

        process = psutil.Process()
        before = _processes(process)
        began, cpu = time.perf_counter(), time.process_time()
        controller.measurement_start, controller.measurements = began, []
        monitoring = threading.Thread(target=monitor, args=(controller,), daemon=True)
        monitoring.start()

        def end_measurement():
            limit_reached.set()
            controller.cancel.set()

        timer = threading.Timer(args.seconds or args.deadline, end_measurement)
        timer.start()

        def apply_resize(at, target):
            event = {
                "planned_seconds": at,
                "seconds": time.perf_counter() - began,
                "target": target,
                "discarded_before": controller.counts["discarded"],
                "active_before": [
                    {
                        "slot": slot,
                        **{
                            key: task.get(key)
                            for key in ("id", "pid", "requests", "commands", "ply")
                        },
                    }
                    for slot, task in tuple(controller.pool.slots.items())
                ],
            }
            try:
                controller.command("concurrency", {"environments": target})
                event["actual_after_request"] = controller.pool.environments
            except Exception as error:
                event["error"] = f"{type(error).__name__}: {error}"
                controller.cancel.set()
            resize_events.append(event)

        for at, target in args.resize_at:
            scheduled = threading.Timer(
                max(0, began + at - time.perf_counter()), apply_resize, (at, target)
            )
            resize_timers.append(scheduled)
            scheduled.start()
        windows = []
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
            if args.finite:
                break
        saving = time.perf_counter()
        controller._save()
        elapsed = time.perf_counter() - began
        saved_seconds = time.perf_counter() - saving
        after = _processes(process)
        after_times = process.cpu_times()
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
            cpu_tree_seconds=max(0, _cpu_total(after) - _cpu_total(before)),
            cpu_tree_includes_reaped=hasattr(after_times, "children_user"),
            windows=windows,
            measurements=controller.measurements,
            save_seconds=saved_seconds,
            counts=dict(controller.counts),
            updates=controller.trainer.updates,
            consumed_samples=controller.samples.consumed,
            terminal_samples_per_second=controller.counts["retained_samples"] / elapsed,
            terminal_games_per_second=controller.counts["games"] / elapsed,
            runtime={
                **controller.pool.metrics(),
                **(
                    controller.sampling_jobs.publication_metrics()
                    if controller.sampling_jobs
                    else {}
                ),
            },
            last_sampling_snapshot=getattr(controller, "last_sampling_snapshot", None),
        )
        if args.finite:
            report["sample_sha256"] = sample_fingerprint(controller.completed_rows)
        if len(resize_events) != len(args.resize_at) or any(
            "error" in event for event in resize_events
        ):
            report["complete"] = False
            raise RuntimeError("并发计划未完整执行，详见 resize_events")
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        jobs = controller.sampling_jobs
        if jobs is not None and jobs.result and jobs.result.get("error"):
            report["failed_game"] = jobs.result
            report["updates"] = controller.trainer.updates
            report["last_sampling_snapshot"] = getattr(controller, "last_sampling_snapshot", None)
            try:
                controller._save()
            except Exception as save_error:
                report["failure_recovery_error"] = str(save_error)
        raise
    finally:
        if timer:
            timer.cancel()
        for scheduled in resize_timers:
            scheduled.cancel()
            scheduled.join(timeout=2)
        controller._close_pool()
        report["counts_after_close"] = dict(controller.counts)
        controller.close()
        report["closed_tasks"] = controller.closed_tasks
        report["work_by_outcome"] = {
            status: {
                "tasks": sum(row["status"] == status for row in controller.closed_tasks),
                "requests": sum(
                    row["requests"] for row in controller.closed_tasks if row["status"] == status
                ),
                "retained_samples": sum(
                    row["retained_samples"]
                    for row in controller.closed_tasks
                    if row["status"] == status
                ),
            }
            for status in ("terminal", "truncated", "unknown")
        }
        if monitoring:
            monitoring.join(timeout=3)
        profile_scope.close()
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
