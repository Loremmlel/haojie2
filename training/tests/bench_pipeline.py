"""最终统一验收：正式运行时、同量连续作业、冻结反馈、外部 CPU/GPU 采样。"""

import argparse
import copy
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import pickle
import queue
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import types

import numpy as np


def starting_positions(directory):
    positions = []
    paths = sorted(
        (directory / "games").glob("*/game.json"), key=lambda p: int(p.parent.name)
    )
    for path in paths[:4]:
        meta = json.loads(path.read_text(encoding="utf-8"))
        boundaries = {
            int(meta["steps"] * fraction) for fraction in (0.25, 0.5, 0.75, 0.9)
        }
        with Environment(meta["game"]["seed"]) as env:
            step = 0
            for name in meta["command_segments"]:
                for line in (
                    (path.parent / name).read_text(encoding="utf-8").splitlines()
                ):
                    result = env.call("apply", command=json.loads(line))
                    if result.get("rejected"):
                        raise RuntimeError(f"基准源轨迹重放失败: {path} {step}")
                    step += 1
                    if step in boundaries and result["winner"] is None:
                        positions.append(
                            {
                                "source_game": meta["game"]["id"],
                                "source_step": step,
                                "ply": result["ply"],
                                "environment": env.call("snapshot"),
                            }
                        )
    if len(positions) != 16:
        raise ValueError("基准需要四局已有轨迹，每局可提取四个非终局起点")
    return positions, [str(p) for p in paths[:4]]


def legacy_modules():
    root = Path(os.environ["HAOJIE_BENCH_BASELINE"])
    name = "haojie_benchmark_baseline"
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(root / "source" / "training")]
        sys.modules[name] = module
    return {
        key: importlib.import_module(name + ".haojie.runtime." + key)
        for key in ("trainer", "pipeline", "gpu", "worker", "storage")
    }


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            # Windows 读句柄短暂阻止替换；采样器读完即释放。
            if attempt == 19:
                raise
            time.sleep(0.001)


def read_json(path):
    for attempt in range(20):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.001)


def role(name, tid=None):
    directory = Path(os.environ["HAOJIE_BENCH_OUTPUT"]) / "roles"
    directory.mkdir(exist_ok=True)
    write_json(
        directory / (name + ".json"),
        {
            "name": name,
            "pid": os.getpid(),
            "tid": threading.get_native_id() if tid is None else tid,
        },
    )


def starting_lane(game, starts):
    case = game["id"] % len(starts)
    return {
        "game": game,
        "environment": starts[case]["environment"],
        "rng": np.random.default_rng(game["seed"] ^ 0xA53113).bit_generator.state,
        "root_version": None,
        "teacher": None,
        "steps": 0,
    }


def legacy_worker(*args):
    modules = legacy_modules()
    from_cache = pickle.loads(Path(os.environ["HAOJIE_BENCH_STARTS"]).read_bytes())
    lane = modules["worker"].Lane
    initialize = lane.__init__

    def restored(self, directory, config, game, previous=None):
        snapshot = starting_lane(game, from_cache)
        snapshot["metadata"] = modules["storage"].GameWriter(directory, game).metadata
        initialize(self, directory, config, game, snapshot)

    lane.__init__ = restored
    role(f"actor-{args[0]}")
    modules["worker"].worker(*args)


class Feedback:
    def __init__(self, executor_class, record):
        self.record = record
        self.path = Path(os.environ["HAOJIE_BENCH_FEEDBACK"])
        self.values = {} if record else pickle.loads(self.path.read_bytes())
        self.positions = {}
        original = executor_class.infer

        def controlled(executor, jobs):
            outputs = original(executor, jobs)
            for i, (_, request, owner) in enumerate(jobs):
                root = executor.bindings[owner][0]
                game, step = (
                    map(int, root.split(":")) if isinstance(root, str) else root
                )
                cursor = self.positions.get((game, step), 0)
                self.positions[game, step] = cursor + 1
                key = game % 16, step, cursor
                descriptor = (
                    request.op,
                    getattr(request, "key", request.context),
                    tuple((v.shape, v.dtype.str) for v in request.arrays),
                )
                if record and key not in self.values:
                    self.values[key] = (*descriptor, outputs[i].copy())
                else:
                    if descriptor != self.values[key][:3]:
                        raise AssertionError(
                            f"冻结请求分岔: {key}: {descriptor[:2]} != {self.values[key][:2]}"
                        )
                    outputs[i] = self.values[key][3]
            return outputs

        executor_class.infer = controlled

    def finish(self):
        if self.record:
            self.path.write_bytes(pickle.dumps(self.values))


def legacy_gpu(*args):
    modules = legacy_modules()
    feedback = Feedback(
        modules["gpu"].Executor, os.environ["HAOJIE_BENCH_CAPTURE"] == "1"
    )
    role("gpu")
    try:
        modules["gpu"].serve(*args)
    finally:
        feedback.finish()


def equal_tree(actual, expected, path="root"):
    if isinstance(expected, np.ndarray):
        if path.endswith((".targets", ".weights")):
            np.testing.assert_allclose(
                actual, expected, rtol=1e-6, atol=1e-7, err_msg=path
            )
        else:
            np.testing.assert_array_equal(actual, expected, err_msg=path)
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"{path}: 字段不同")
        for key in expected:
            equal_tree(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            raise AssertionError(f"{path}: 长度不同")
        for index, (left, right) in enumerate(zip(actual, expected)):
            equal_tree(left, right, f"{path}.{index}")
    elif actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def records(directory, count):
    cases, traces = {}, {}
    for number in range(count):
        path = Path(directory) / "games" / str(number)
        meta = json.loads((path / "game.json").read_text(encoding="utf-8"))
        trace = [
            json.loads(line)
            for name in meta["command_segments"]
            for line in (path / name).read_text(encoding="utf-8").splitlines()
        ]
        chunks = []
        for name in meta["chunks"]:
            with np.load(path / name, allow_pickle=False) as data:
                chunks.append({k: data[k] for k in data.files})
        case = {
            "commands": trace,
            "winner": meta["winner"],
            "status": meta["status"],
            "timeline": meta["timeline"],
            "chunks": chunks,
        }
        if number % 16 in cases:
            equal_tree(case, cases[number % 16], f"repeated-case-{number}")
        else:
            cases[number % 16] = case
        traces[number] = trace
    return {"cases": cases, "traces": traces}


def execute(args):
    import torch
    from training.haojie.runtime.config import Config

    global Environment
    from training.haojie.native import Environment

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline = Path(args.baseline).resolve()
    os.environ.update(
        HAOJIE_BENCH_BASELINE=str(baseline),
        HAOJIE_BENCH_OUTPUT=str(output),
        HAOJIE_BENCH_FEEDBACK=str(Path(args.feedback).resolve()),
        HAOJIE_BENCH_CAPTURE=str(int(args.capture)),
    )
    if args.execute == "baseline":
        os.environ["HAOJIE_ENGINE"] = str(baseline / "haojie_native.dll")
    starts, replay = starting_positions(Path(args.checkpoint).resolve().parent)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    updates = args.tasks * args.commands // args.batch
    config = Config(
        workers=args.workers,
        games_per_worker=64 // args.workers,
        batch=args.batch,
        warmup_games=0,
        teacher_fraction=0,
        history_fraction=0,
        max_commands=args.commands,
        sample_reuse=0,
        save_seconds=0,
        publish_updates=updates + 1,
    )
    state = {"assigned": 0, "completed": 0, "full_at": None, "last_write": 0.0}
    sampler = None
    feedback = None
    with tempfile.TemporaryDirectory(prefix="haojie-steady-workload-") as directory:
        starts_file = Path(directory) / "starts.pkl"
        starts_file.write_bytes(pickle.dumps(starts))
        os.environ["HAOJIE_BENCH_STARTS"] = str(starts_file)
        if args.execute == "baseline":
            modules = legacy_modules()
            modules["pipeline"].worker = legacy_worker
            modules["pipeline"].serve = legacy_gpu
            trainer = modules["trainer"].Trainer(directory, config)
            replay_class = modules["storage"].Replay
            role("coordinator")
        else:
            from training.haojie.runtime.trainer import Trainer
            from training.haojie.runtime.storage import Replay
            from training.haojie.runtime import pipeline, gpu
            from training.haojie.native_pool import NativePool

            assign = NativePool.assign
            status = NativePool.status

            def seeded(self, lane, game, restored=None, **kwargs):
                return assign(self, lane, game, starting_lane(game, starts), **kwargs)

            def tracked_status(self):
                result = status(self)
                for index, tid in enumerate(result.get("threads", [])):
                    if tid:
                        role(f"rust-{index}", tid)
                return result

            run_coordinator = pipeline.Coordinator.run

            def coordinated(self):
                role("coordinator")
                return run_coordinator(self)

            NativePool.assign, NativePool.status = seeded, tracked_status
            pipeline.Coordinator.run = coordinated
            feedback = Feedback(gpu.Executor, args.capture)
            trainer = Trainer(directory, config)
            replay_class = Replay
            role("gpu")
        trainer.model.load_state_dict(saved["model"])
        trainer.restore_optimizer(copy.deepcopy(saved["optimizer"]))
        torch.save(trainer.model.state_dict(), trainer.models / "model-00000000.pt")

        class FixedReplay(replay_class):
            def add(self, path):
                pass

        trainer.replay = FixedReplay(games=replay)
        trainer.rng = np.random.default_rng(818031)
        trainer.credit = updates * config.batch

        def progress(force=False):
            now = time.perf_counter()
            if force or now - state["last_write"] >= 0.5:
                state["last_write"] = now
                write_json(
                    output / "monitor.json",
                    {
                        "time": now,
                        "assigned": state["assigned"],
                        "completed": state["completed"],
                        "live": state["assigned"] - state["completed"],
                        "backlog": args.tasks - state["assigned"],
                        "full_at": state["full_at"],
                    },
                )
            if state["completed"] == args.tasks and trainer.stats["updates"] >= updates:
                trainer.stop_requested = True

        class Stats(dict):
            def __reduce__(self):
                # 旧架构 spawn 只接收普通统计字典，回调留在协调进程。
                return dict, (dict(self),)

            def update(self, *values, **keywords):
                super().update(*values, **keywords)
                progress()

        trainer.stats = Stats(trainer.stats)

        def game():
            if state["assigned"] == args.tasks:
                return None
            number = state["assigned"]
            state["assigned"] += 1
            trainer.next_game = state["assigned"]
            if state["assigned"] == 64:
                state["full_at"] = time.perf_counter()
            # 完成事件可能刚记录短暂空位；补满后立即发布真实供给状态。
            progress(
                state["assigned"] - state["completed"] == 64
                or state["assigned"] == args.tasks
            )
            return {
                "id": number,
                "seed": 910001 + number % 16,
                "learner": 1,
                "opponent": "self",
                "difficulty": "easy",
                "ply_limit": 0,
            }

        old_event = trainer.event

        def event(message):
            credit = trainer.credit
            old_event(message)
            if message["type"] == "done":
                # 基准的学习量固定，真实终局奖励仍通过冻结源回放进入学习。
                trainer.credit = credit
                state["completed"] += 1
                progress(state["completed"] == args.tasks)

        trainer.game, trainer.event = game, event
        progress(True)
        if not args.capture:
            sampler = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "training.tests.bench_pipeline",
                    "--sample",
                    str(output),
                ],
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        started = time.perf_counter()
        try:
            trainer.run()
            elapsed = time.perf_counter() - started
            if trainer.stats["updates"] != updates:
                raise AssertionError("学习更新数量改变")
            signature = records(directory, args.tasks)
            signature["learning_rng"] = trainer.rng.bit_generator.state
            signature["counts"] = {
                k: trainer.stats[k]
                for k in (
                    "commands",
                    "inference_roots",
                    "inference_decodes",
                    "updates",
                    "trained_samples",
                    "value_samples",
                    "soft_value_samples",
                )
            }
            (output / "signature.pkl").write_bytes(pickle.dumps(signature))
            result = {
                "backend": args.execute,
                "profile_enabled": os.environ.get("HAOJIE_PROFILE", "0") == "1",
                "trace_path": os.environ.get("HAOJIE_TRACE"),
                "config": asdict(config),
                "workers": config.workers,
                "games_per_worker": config.games_per_worker,
                "training_batch": config.batch,
                "tasks": args.tasks,
                "seconds": elapsed,
                **trainer.stats,
                "commands_per_second": trainer.stats["commands"] / elapsed,
                "trained_samples_per_second": trainer.stats["trained_samples"]
                / elapsed,
            }
            write_json(output / "result.json", result)
        finally:
            (output / "STOP-MONITOR").touch()
            if sampler is not None:
                sampler.wait(timeout=15)
            if feedback is not None:
                feedback.finish()


def summarize(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": values[int((len(values) - 1) * 0.9)],
        "max": max(values),
    }


def start_windows_gpu_memory():
    if os.name != "nt":
        return None
    command = (
        "Get-Counter -Counter '\\GPU Process Memory(*)\\Shared Usage',"
        "'\\GPU Process Memory(*)\\Dedicated Usage' "
        "-SampleInterval 1 -Continuous -ErrorAction Stop | ForEach-Object { "
        "ConvertTo-Json -InputObject @($_.CounterSamples | "
        "Select-Object InstanceName,Path,CookedValue) -Compress }"
    )
    return subprocess.Popen(
        ["pwsh.exe", "-NoProfile", "-Command", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def process_gpu_memory(counters, pids):
    if counters is None:
        return None
    selected = [
        row
        for row in counters
        if row["InstanceName"].startswith("pid_")
        and int(row["InstanceName"].split("_", 2)[1]) in pids
    ]
    result = {"instances": selected}
    for kind in ("shared", "dedicated"):
        values = [
            row["CookedValue"]
            for row in selected
            if row["Path"].lower().endswith(f"\\{kind} usage")
        ]
        # 无对应计数器保持未知，不能以零冒充已测量。
        result[f"{kind}_bytes"] = sum(values) if values else None
    return result


def sample(directory):
    import psutil

    directory = Path(directory)
    output = queue.Queue()
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu,memory.used,memory.free",
        "--format=csv,noheader,nounits",
        "--loop-ms=500",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    def read_gpu():
        for line in process.stdout:
            try:
                output.put(
                    dict(
                        zip(
                            (
                                "active_percent",
                                "memory_active_percent",
                                "watts",
                                "sm_mhz",
                                "temperature",
                                "memory_used_mib",
                                "memory_free_mib",
                            ),
                            [
                                float(v) if v.strip() != "[N/A]" else None
                                for v in line.split(",")
                            ],
                        )
                    )
                )
            except ValueError:
                pass

    reader = threading.Thread(target=read_gpu, daemon=True)
    reader.start()
    memory_process = None
    memory_output = queue.Queue()

    def read_gpu_memory():
        try:
            for line in memory_process.stdout:
                memory_output.put(json.loads(line))
        finally:
            memory_output.put(None)

    previous, rows, gpu, memory_counters = {}, [], None, None
    psutil.cpu_percent(percpu=True)
    try:
        while not (directory / "STOP-MONITOR").exists():
            at = time.perf_counter()
            while not output.empty():
                gpu = output.get()
            while not memory_output.empty():
                memory_counters = memory_output.get()
            state = read_json(directory / "monitor.json")
            if os.name == "nt" and state["full_at"] and memory_process is None:
                # CUDA 初始化后才枚举计数器，避免遗漏新出现的训练 PID 实例。
                memory_process = start_windows_gpu_memory()
                threading.Thread(target=read_gpu_memory, daemon=True).start()
            roles = [read_json(p) for p in (directory / "roles").glob("*.json")]
            processes = {}
            for role in roles:
                pid = role["pid"]
                if pid not in processes:
                    try:
                        p = psutil.Process(pid)
                        times = p.cpu_times()
                        processes[pid] = (
                            times.user + times.system,
                            {t.id: t.user_time + t.system_time for t in p.threads()},
                        )
                    except psutil.Error:
                        continue
            current = {}
            for pid, (total, threads) in processes.items():
                if pid in previous:
                    old_at, old_total, old_threads = previous[pid]
                    dt = at - old_at
                    current[pid] = {
                        "process": (total - old_total) / dt,
                        "threads": {
                            tid: (value - old_threads[tid]) / dt
                            for tid, value in threads.items()
                            if tid in old_threads
                        },
                    }
                previous[pid] = at, total, threads
            thread_cores = {
                r["name"]: current[r["pid"]]["threads"].get(r["tid"], 0)
                for r in roles
                if r["pid"] in current
            }
            steady = bool(
                state["full_at"]
                and at >= state["full_at"] + 5
                and state["live"] == 64
                and state["backlog"] > 0
            )
            rows.append(
                {
                    "time": at,
                    "steady": steady,
                    "state": state,
                    "gpu": gpu,
                    "windows_gpu_memory": process_gpu_memory(
                        memory_counters, {r["pid"] for r in roles}
                    ),
                    "thread_cores": thread_cores,
                    "process_cores": {
                        str(pid): value["process"] for pid, value in current.items()
                    },
                    "system_cpu_percent": psutil.cpu_percent(percpu=True),
                }
            )
            time.sleep(max(0, 0.5 - (time.perf_counter() - at)))
    finally:
        process.terminate()
        process.wait(timeout=5)
        if memory_process is not None:
            if memory_process.poll() is None:
                memory_process.terminate()
            memory_process.wait(timeout=5)
    active = [r for r in rows if r["steady"] and r["gpu"] is not None]
    names = {name for r in active for name in r["thread_cores"]}
    report = {
        "samples": rows,
        "steady_samples": len(active),
        "steady_seconds": len(active) * 0.5,
        "cpu_threads": {
            name: summarize([r["thread_cores"].get(name) for r in active])
            for name in names
        },
        "gpu": {
            name: summarize([r["gpu"][name] for r in active])
            for name in (
                "active_percent",
                "memory_active_percent",
                "watts",
                "sm_mhz",
                "temperature",
                "memory_used_mib",
                "memory_free_mib",
            )
        },
        "windows_gpu_memory": {
            scope: {
                "sample_count": sum(
                    (r["windows_gpu_memory"] or {}).get("shared_bytes") is not None
                    for r in samples
                ),
                **{
                    name: summarize(
                        [(r["windows_gpu_memory"] or {}).get(name) for r in samples]
                    )
                    for name in ("shared_bytes", "dedicated_bytes")
                },
            }
            for scope, samples in (("steady", active), ("whole_run", rows))
        },
    }
    write_json(directory / "utilization.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--baseline", default="artifacts/classic-native-pool-20261029/baseline"
    )
    parser.add_argument(
        "--output", default="artifacts/classic-native-pool-20261029/benchmark"
    )
    parser.add_argument("--tasks", type=int, default=256)
    parser.add_argument("--commands", type=int, default=8)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument("--compare-workers", type=int, nargs=2, metavar=("A", "B"))
    comparison.add_argument("--compare-profile", action="store_true")
    parser.add_argument("--execute", choices=("baseline", "native"))
    parser.add_argument("--feedback")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--sample")
    args = parser.parse_args()
    if args.sample:
        sample(args.sample)
        return
    if args.compare_profile:
        args.workers, args.batch = 8, 16
        if not args.feedback:
            parser.error("--compare-profile requires frozen --feedback")
    if args.trials is None:
        args.trials = 2 if args.compare_profile else 3
    if args.compare_profile and args.trials != 2:
        parser.error("--compare-profile uses two OFF/ON pairs")
    workers = args.compare_workers or [args.workers]
    if any(value < 1 or 64 % value for value in workers):
        parser.error("workers must be positive divisors of 64")
    if args.compare_workers and workers[0] == workers[1]:
        parser.error("--compare-workers requires two different counts")
    if args.batch < 1 or args.tasks * args.commands % args.batch:
        parser.error("--batch must divide the fixed training sample count")
    if args.feedback and not args.capture and not Path(args.feedback).is_file():
        parser.error("--feedback must name an existing frozen feedback file")
    if args.execute:
        execute(args)
        return
    if not args.checkpoint:
        parser.error("--checkpoint is required")
    if args.tasks < 128 or args.tasks % 16:
        parser.error("--tasks must be a multiple of 16 and at least 128")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    feedback = Path(args.feedback).resolve() if args.feedback else root / "feedback.pkl"
    reference = None

    def invoke(backend, workers, target, profile=False, capture=False):
        command = [
            sys.executable,
            "-B",
            "-m",
            "training.tests.bench_pipeline",
            "--execute",
            backend,
            "--checkpoint",
            str(Path(args.checkpoint).resolve()),
            "--baseline",
            str(Path(args.baseline).resolve()),
            "--output",
            str(target),
            "--feedback",
            str(feedback),
            "--tasks",
            str(64 if capture else args.tasks),
            "--commands",
            str(args.commands),
            "--workers",
            str(workers),
            "--batch",
            str(args.batch),
        ]
        if capture:
            command.append("--capture")
        environment = {**os.environ, "HAOJIE_PROFILE": str(int(profile))}
        environment.pop("HAOJIE_TRACE", None)
        with (root / (target.name + ".log")).open("w", encoding="utf-8") as stream:
            subprocess.run(command, stdout=stream, env=environment, check=True)
        return json.loads((target / "result.json").read_text(encoding="utf-8"))

    if args.compare_profile:
        variants = [
            ("profile-off", "native", 8, False),
            ("profile-on", "native", 8, True),
        ]
    elif args.compare_workers:
        variants = [(f"native-{n}", "native", n, False) for n in args.compare_workers]
    else:
        variants = [
            (name, name, args.workers, False) for name in ("baseline", "native")
        ]
    if not args.feedback:
        _, backend, workers, profile = variants[0]
        captured = invoke(backend, workers, root / "capture", profile, capture=True)
        print(json.dumps({"event": "captured", **captured}), flush=True)
    else:
        print(
            json.dumps({"event": "reused_feedback", "path": str(feedback)}), flush=True
        )
    for trial in range(args.trials):
        ordered = variants if trial % 2 == 0 else list(reversed(variants))
        for label, backend, workers, profile in ordered:
            target = root / f"{trial}-{label}"
            result = invoke(backend, workers, target, profile)
            signature = pickle.loads((target / "signature.pkl").read_bytes())
            if reference is None:
                reference = signature
            else:
                equal_tree(signature, reference)
            utilization = json.loads(
                (target / "utilization.json").read_text(encoding="utf-8")
            )
            utilization.pop("samples")
            report = {"trial": trial, **result, "utilization": utilization}
            with (root / "results.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report) + "\n")
            print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
