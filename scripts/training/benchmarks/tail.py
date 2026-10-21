"""真实长尾采样窗口：分别记录首次更新延迟、已完成工作和仍在运行的对局。"""

import argparse
import copy
import gzip
import hashlib
import importlib
import json
import platform
import random
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("finite", "continuous"), required=True)
    parser.add_argument("--starts", type=Path)
    parser.add_argument("--start-indices", type=int, nargs="+", default=[0, 1, 0])
    parser.add_argument("--environments", type=int, default=2)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--deadline", type=float, default=600)
    args = parser.parse_args()
    if min(args.windows, args.updates, args.threads) < 1:
        parser.error("窗口数、更新量和线程数必须为正整数")
    sys.path.insert(0, str(args.source.resolve() / "training"))
    importlib.import_module("haojie_training")
    import torch

    from haojie_training.console.config import validate
    from haojie_training.console.config.resources import hardware
    from haojie_training.console.controller import Controller
    from haojie_training.console.sampling import MemoryJobs
    from haojie_training.console.storage import Store
    from haojie_training.model import NETWORK_VERSION
    from haojie_training.runtime import synchronize

    class MeasuredController(Controller):
        def _loop(self):
            pass

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    starts_path = args.starts or args.source / "tests/fixtures/native/late-starts.json.gz"
    raw = starts_path.read_bytes()
    starts = json.loads(gzip.decompress(raw) if starts_path.suffix == ".gz" else raw)
    starts = [starts[i] for i in args.start_indices]
    source_hash = hashlib.sha256()
    for path in sorted((args.source / "training/haojie_training").rglob("*.py")):
        source_hash.update(path.relative_to(args.source).as_posix().encode())
        source_hash.update(path.read_bytes())
    report = {
        "format": "haojie-tail-benchmark-v1",
        "complete": False,
        "scope": "equal-completion-thresholds-different-training-trajectories",
        "matched_work": False,
        "network": NETWORK_VERSION,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "hardware": hardware(),
        "source_sha256": source_hash.hexdigest(),
        "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "engine_sha256": hashlib.sha256(args.engine.read_bytes()).hexdigest(),
        "starts_sha256": hashlib.sha256(raw).hexdigest(),
        "windows": [],
        "finished_tasks": [],
    }
    with tempfile.TemporaryDirectory(prefix="haojie-tail-") as folder:
        c = MeasuredController(
            args.engine,
            Store(folder),
            device=args.device,
            tiny=args.tiny,
            starts=starts,
            threads=args.threads,
        )
        c.thread.join()
        c.continuous_selfplay = args.mode == "continuous"
        c.config.update(
            environments=args.environments,
            games_per_round=args.games,
            updates_per_round=args.updates,
            batch_size=8,
            inference_batch_size=args.environments,
            history_percent=0,
            eval_games=100000,
            pool_mib=256,
            seed=args.seed,
        )
        validate(c.config)
        c.rng = random.Random(args.seed)
        c.want_run = True
        original_finish, original_progress = MemoryJobs.finish, c._progress
        began = None

        def progress(metrics):
            original_progress(metrics)
            if time.perf_counter() - began > args.deadline:
                raise TimeoutError("真实长尾测量超过保护时间，未完成局另列")

        def finish(jobs, task, message):
            report["finished_tasks"].append(
                {
                    "task": task["id"],
                    "seconds": time.perf_counter() - began,
                    "updates_before_finish": c.trainer.updates,
                    "outcome": message["outcome"],
                    "final_hash": message.get("finalHash"),
                    "requests": task["requests"],
                    "samples_seen": task["trajectory"].seen,
                    "behavior_generations": sorted(
                        {
                            row["_meta"].get("behavior_generation", c.generation)
                            for row, *_ in task["trajectory"].rows
                        }
                    ),
                }
            )
            return original_finish(jobs, task, message)

        try:
            c._initialize()
            report["initial_model_sha256"] = c.version
            report["parameters"] = sum(p.numel() for p in c.trainer.model.parameters())
            report["model_config"] = asdict(c.trainer.model.config)
            report["config"] = dict(c.config)
            synchronize(c.device)
            began = time.perf_counter()
            c._progress = progress
            with patch.object(MemoryJobs, "finish", finish):
                for index in range(args.windows):
                    before = len(report["finished_tasks"])
                    started = time.perf_counter()
                    with c._phase("sampling"):
                        c._sample()
                    synchronize(c.device)
                    sample_seconds = time.perf_counter() - started
                    active = [
                        {
                            "task": task["id"],
                            "requests": task["requests"],
                            "commands": task.get("commands"),
                            "samples_seen": task["trajectory"].seen,
                        }
                        for task in c.pool.slots.values()
                    ]
                    started = time.perf_counter()
                    if index == 0:
                        report["first_update_started_seconds"] = started - began
                    updates = c.trainer.updates
                    with c._phase("updating"):
                        c._update()
                    synchronize(c.device)
                    if index == 0:
                        report["first_update_finished_seconds"] = time.perf_counter() - began
                    row = {
                        "window": index + 1,
                        "sample_seconds": sample_seconds,
                        "update_seconds": time.perf_counter() - started,
                        "finished_tasks": [r["task"] for r in report["finished_tasks"][before:]],
                        "active_tasks": active,
                        "updates": c.trainer.updates - updates,
                        "sampling": dict(c.last_round),
                    }
                    report["windows"].append(row)
                    print(json.dumps(row), flush=True)
            report["complete"] = True
        except BaseException as error:
            report["error"] = repr(error)
            raise
        finally:
            report["wall_seconds"] = time.perf_counter() - began if began is not None else None
            report["counts_before_close"] = copy.deepcopy(c.counts)
            report["unfinished_at_window_end"] = (
                c.pool.metrics()["active"] if c.pool is not None else []
            )
            c._close_pool()
            c.close()
            report["counts_after_close"] = dict(c.counts)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
