"""计量真实控制器的完整采样、更新和保存；同一脚本可加载两个固定源码树作对照。"""

import argparse
import copy
import gzip
import hashlib
import json
import platform
import random
import sys
import tempfile
import time
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path

if __package__:
    from .compare import validate_round
else:
    from compare import validate_round


def hardware_identity(hardware):
    """只记录稳定型号和资源配额；动态可用内存、利用率不参与负载一致性。"""
    cpu_model = platform.processor()
    if sys.platform == "linux":
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    elif sys.platform == "win32":
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as key:
            cpu_model = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        **{key: hardware[key] for key in ("logical_cpus", "cpu_capacity", "gpus")},
    }


class RequestCapture:
    """独立采集运行中按形状分桶，每桶最多四条；只保存公开模型输入，不保存局面。"""

    def __init__(self, keys, limit=128):
        self.keys, self.limit = keys, limit
        self.rows, self.buckets = [], Counter()
        self.bytes = self.skipped = 0

    def add(self, messages):
        for message in messages:
            row = message["input"]
            bucket = tuple((len(row[key]) - 1).bit_length() for key in ("entities", "candidates"))
            size = sum(row[key].numel() * row[key].element_size() for key in self.keys)
            if (
                len(self.rows) >= self.limit
                or self.buckets[bucket] >= 4
                or self.bytes + size > 256 * 2**20
            ):
                self.skipped += 1
                continue
            self.rows.append({key: row[key].clone() for key in sorted(self.keys)})
            self.buckets[bucket] += 1
            self.bytes += size


def digest_rows(rows):
    """在计时段外校验样本集合；完成顺序不影响集合摘要，不修改训练张量。"""
    parts = []
    for row, _, _ in rows:
        meta = row["_meta"]
        h = hashlib.sha256()
        h.update(
            json.dumps(
                {k: v for k, v in meta.items() if k not in {"entered", "generation"}},
                sort_keys=True,
            ).encode()
        )
        for key in sorted(k for k in row if not k.startswith("_")):
            value = row[key].contiguous()
            h.update(f"{key}:{value.dtype}:{list(value.shape)}".encode())
            h.update(value.numpy().tobytes())
        parts.append(h.hexdigest())
    return hashlib.sha256("".join(sorted(parts)).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--starts", type=Path, help="可重放真实起点；省略则从自然新局开始")
    parser.add_argument("--start-indices", type=int, nargs="+")
    parser.add_argument("--environments", type=int, default=4)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--batch-wait-ms", type=float, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sampling-precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--max-commands", type=int)
    parser.add_argument("--max-plies", type=int)
    parser.add_argument("--history-percent", type=int, default=0)
    parser.add_argument("--pool-mib", type=int, default=256)
    parser.add_argument("--deadline", type=float, default=1800)
    parser.add_argument(
        "--ordered-updates",
        action="store_true",
        help="仅基准：同一采样集合按任务/决策排序，排除完成顺序对更新负载的干扰",
    )
    parser.add_argument("--state-output", type=Path, help="额外保存末态模型及优化器，供容差对照")
    parser.add_argument(
        "--requests-output",
        type=Path,
        help="单独捕获最多128条代表形状的公开推理输入；本次运行不进入性能对照",
    )
    args = parser.parse_args()
    if min(args.rounds, args.updates, args.threads) < 1:
        parser.error("轮数、更新量和线程数必须为正整数")
    sys.path.insert(0, str(args.source.resolve() / "training"))
    import torch

    from haojie_training.console.config import validate
    from haojie_training.console.config.resources import hardware
    from haojie_training.console.controller import Controller
    from haojie_training.console.sampling import MemoryJobs
    from haojie_training.console.storage import Store
    from haojie_training.data import INPUT_KEYS
    from haojie_training.runtime import synchronize

    # 不启动自动循环，计时驱动持有控制器；采样器、原生进程、学习器和存储均为生产实现。
    class MeasuredController(Controller):
        def _loop(self):
            pass

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    starts, starts_hash = None, None
    if args.starts:
        raw = args.starts.read_bytes()
        starts_hash = hashlib.sha256(raw).hexdigest()
        starts = json.loads(gzip.decompress(raw) if args.starts.suffix == ".gz" else raw)
        if args.start_indices:
            starts = [starts[i] for i in args.start_indices]
    source_hash = hashlib.sha256()
    for path in sorted((args.source / "training/haojie_training").rglob("*.py")):
        source_hash.update(path.relative_to(args.source).as_posix().encode())
        source_hash.update(path.read_bytes())
    hardware_info = hardware()
    report = {
        "format": "haojie-cycle-benchmark-v2",
        "complete": False,
        "measurement_eligible": args.requests_output is None,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "source_sha256": source_hash.hexdigest(),
        "engine_sha256": hashlib.sha256(args.engine.read_bytes()).hexdigest(),
        "starts_sha256": starts_hash,
        "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "hardware": hardware_info,
        "hardware_identity": hardware_identity(hardware_info),
        "rounds": [],
    }
    capture = RequestCapture(INPUT_KEYS) if args.requests_output else None
    with tempfile.TemporaryDirectory(prefix="haojie-cycle-") as folder:
        c = MeasuredController(
            args.engine, Store(Path(folder)), device=args.device, tiny=args.tiny, starts=starts
        )
        c.thread.join()
        c.config.update(
            environments=args.environments,
            games_per_round=args.games,
            updates_per_round=args.updates,
            batch_size=args.batch_size,
            inference_batch_size=args.inference_batch_size,
            batch_wait_ms=args.batch_wait_ms,
            sampling_precision=args.sampling_precision,
            history_percent=args.history_percent,
            seed=args.seed,
            pool_mib=args.pool_mib,
            eval_games=100000,
        )
        for key in ("max_commands", "max_plies"):
            if getattr(args, key) is not None:
                c.config[key] = getattr(args, key)
        validate(c.config)
        c.rng = random.Random(args.seed)
        c.want_run = True
        began = time.perf_counter()
        original_progress, original_finish, original_infer = (
            c._progress,
            MemoryJobs.finish,
            MemoryJobs.infer,
        )
        finished, last_progress = [], {}

        def progress(metrics):
            nonlocal last_progress
            last_progress = metrics
            original_progress(metrics)
            if time.perf_counter() - began > args.deadline:
                raise TimeoutError("完整周期超出测量保护时间；本次不记作成功吞吐")

        def finish(jobs, task, message):
            finished.append(
                {
                    "task": task["id"],
                    "outcome": message["outcome"],
                    "final_hash": message.get("finalHash"),
                    "requests": task["requests"],
                    "commands": task.get("commands"),
                    "ply": message.get("ply"),
                    "error": message.get("error"),
                    "samples_seen": task["trajectory"].seen,
                }
            )
            return original_finish(jobs, task, message)

        def infer(jobs, messages):
            capture.add(messages)
            return original_infer(jobs, messages)

        c._progress = progress
        MemoryJobs.finish = finish
        if capture:
            MemoryJobs.infer = infer
        try:
            c._initialize()
            synchronize(c.device)
            report.update(
                setup_seconds=time.perf_counter() - began,
                config=copy.deepcopy(c.config),
                parameters=sum(p.numel() for p in c.trainer.model.parameters()),
                model_config=asdict(c.trainer.model.config),
                initial_model_sha256=c.version,
                execution={
                    "device": str(c.device),
                    "cuda_version": torch.version.cuda,
                    "sampling_precision": c.config["sampling_precision"],
                    "update_precision": c.trainer.precision,
                    "optimizer": type(c.trainer.optimizer).__name__,
                    "optimizer_options": c.trainer.optimizer.defaults,
                    "deterministic": torch.are_deterministic_algorithms_enabled(),
                    "matmul_precision": torch.get_float32_matmul_precision(),
                    "interop_threads": torch.get_num_interop_threads(),
                },
            )
            for index in range(args.rounds):
                before_counts = copy.deepcopy(c.counts)
                before_consumed, before_updates = c.samples.consumed, c.trainer.updates
                finished.clear()
                start = time.perf_counter()
                with c._phase("sampling"):
                    c._sample()
                synchronize(c.device)
                sample_seconds = time.perf_counter() - start
                validation = time.perf_counter()
                sample_hash = digest_rows(c.samples.rows)
                sample_pool = c.samples.snapshot()
                if args.ordered_updates:
                    c.samples.rows = deque(
                        sorted(
                            c.samples.rows,
                            key=lambda item: (
                                item[0]["_meta"]["game"]["task"],
                                item[0]["_meta"]["decision"] or 0,
                            ),
                        )
                    )
                validation_seconds = time.perf_counter() - validation
                start = time.perf_counter()
                with c._phase("updating"):
                    c._update()
                synchronize(c.device)
                update_seconds = time.perf_counter() - start
                start = time.perf_counter()
                c._save()
                synchronize(c.device)
                save_seconds = time.perf_counter() - start
                row = {
                    "round": index,
                    "sample_seconds": sample_seconds,
                    "update_seconds": update_seconds,
                    "save_seconds": save_seconds,
                    "cycle_seconds": sample_seconds + update_seconds + save_seconds,
                    "validation_seconds_excluded": validation_seconds,
                    "samples_hash": sample_hash,
                    "sample_pool_before_updates": sample_pool,
                    "counts": {
                        k: c.counts[k] - before_counts[k]
                        for k in (
                            "games",
                            "truncated",
                            "discarded",
                            "seen_samples",
                            "retained_samples",
                            "tasks",
                        )
                    },
                    "updates": c.trainer.updates - before_updates,
                    "samples_consumed": c.samples.consumed - before_consumed,
                    "sampling": dict(c.last_round),
                    "sampling_details_cumulative": c.pool.metrics(),
                    "tasks": sorted(finished, key=lambda task: task["task"]),
                }
                row["retained_samples_per_second"] = (
                    row["counts"]["retained_samples"] / row["cycle_seconds"]
                )
                report["rounds"].append(row)
                print(json.dumps(row, ensure_ascii=True), flush=True)
                validate_round(row, args.games, args.updates)
            report["complete"] = True
            if args.state_output:
                args.state_output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(c.trainer.checkpoint(c.meta), args.state_output)
        except BaseException as error:
            report["complete"] = False
            report["error"] = repr(error)
            report["interrupted"] = {
                "phase": c.state,
                "counts": dict(c.counts),
                "finished_tasks": sorted(finished, key=lambda task: task["task"]),
                "last_sampling_snapshot": last_progress,
            }
            raise
        finally:
            MemoryJobs.finish = original_finish
            MemoryJobs.infer = original_infer
            c._close_pool()
            report["wall_seconds"] = time.perf_counter() - began
            report["performance"] = c.performance()
            if capture:
                args.requests_output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "format": "haojie-inference-requests-v1",
                        "source_sha256": report["source_sha256"],
                        "engine_sha256": report["engine_sha256"],
                        "requests": capture.rows,
                    },
                    args.requests_output,
                )
                report["capture"] = {
                    "requests": len(capture.rows),
                    "tensor_bytes": capture.bytes,
                    "skipped": capture.skipped,
                }
                with args.requests_output.open("rb") as captured:
                    report["capture"]["sha256"] = hashlib.file_digest(
                        captured, "sha256"
                    ).hexdigest()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    main()
