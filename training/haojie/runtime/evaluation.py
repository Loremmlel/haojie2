"""定时冻结已发布权重并启动独立评测；错误只影响评测，不停止训练。"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


def read_evaluation(directory):
    path = Path(directory) / "evaluations" / "status.json"
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if status["state"] in ("starting", "running"):
        owner = Path(directory) / "evaluations" / status["round"] / "process.json"
        process = (
            json.loads(owner.read_text(encoding="utf-8")) if owner.exists() else status
        )
        try:
            alive = (
                psutil.Process(process["pid"]).create_time()
                == process["process_created"]
            )
        except psutil.Error:
            alive = False
        if not alive:
            status.update(
                state="failed", error="评测进程已退出，查看本轮 process.log。"
            )
    return status


class EvaluationSchedule:
    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.config = config
        previous = read_evaluation(directory)
        self.next_at = (
            previous["started_at"] + config.evaluation_seconds if previous else None
        )
        self.snapshot = self.launch = self.process = None
        self.writer = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="evaluation-launch"
        )

    def tick(self, local):
        if not self.config.evaluation_seconds:
            return
        try:
            if self.launch is not None:
                if not self.launch.done():
                    return
                self.process = self.launch.result()
                self.launch = None
            if self.snapshot is not None:
                if self.snapshot.done():
                    frozen = self.snapshot.result()
                    self.snapshot = None
                    self.launch = self.writer.submit(self.start, frozen)
                return
            now = time.time()
            if self.next_at is None:
                self.next_at = now + self.config.evaluation_seconds
            if now < self.next_at:
                return
            previous = read_evaluation(self.directory)
            if (self.process is not None and self.process.poll() is None) or (
                previous and previous["state"] in ("starting", "running")
            ):
                self.next_at = now + 5
                return
            self.next_at = now + self.config.evaluation_seconds
            self.snapshot = local.submit("evaluation")
        except Exception as error:
            from .storage import atomic_json

            self.snapshot = self.launch = None
            self.next_at = time.time() + self.config.evaluation_seconds
            path = self.directory / "evaluations"
            path.mkdir(exist_ok=True)
            atomic_json(
                path / "status.json",
                {
                    "state": "failed",
                    "started_at": time.time(),
                    "error": str(error),
                },
            )

    def start(self, frozen):
        import torch
        from .storage import atomic_json

        directory = self.directory / "evaluations" / f"round-{time.time_ns()}"
        directory.mkdir(parents=True)
        checkpoint = directory / "model.pt"
        torch.save(frozen["model"], checkpoint)
        manifest = {
            "version": frozen["version"],
            "started_at": time.time(),
            "seed": self.config.seed ^ 0xE7A1,
            "games": 4,
            "simulations": self.config.simulations,
            "max_commands": self.config.max_commands,
            "device": self.config.device,
        }
        atomic_json(directory / "round.json", manifest)
        atomic_json(
            directory.parent / "status.json",
            {
                **manifest,
                "state": "starting",
                "round": directory.name,
                "pid": os.getpid(),
                "process_created": psutil.Process().create_time(),
                "results": [],
                "completed": 0,
                "total": 12,
            },
        )
        # 评测拥有独立的进程、模型和 RNG；训练保存/退出不等待它结束。
        with (directory / "process.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-B",
                    "-m",
                    "training.haojie",
                    "evaluate-round",
                    "--directory",
                    str(directory),
                ],
                cwd=Path(__file__).resolve().parents[3],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                start_new_session=os.name != "nt",
            )
        atomic_json(
            directory / "process.json",
            {
                "pid": process.pid,
                "process_created": psutil.Process(process.pid).create_time(),
            },
        )
        return process

    def close(self):
        # 只等小型权重文件和进程启动完成，不等待对弈。
        self.writer.shutdown(wait=True)
