"""定时冻结已发布权重并启动独立评测；错误只影响评测，不停止训练。"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil


def read_evaluation(directory, folder="evaluations"):
    path = Path(directory) / folder / "status.json"
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        status = None
    if folder == "history-evaluations":
        queued = []
        for job in sorted(path.parent.glob("round-*/round.json")):
            item = json.loads(job.with_name("status.json").read_text(encoding="utf-8"))
            if item["state"] == "queued":
                queued.append(item)
        if status is None and queued:
            status = queued[0]
        if status is not None:
            status["pending"] = len(queued)
    if status is None:
        return None
    if status["state"] in ("starting", "running"):
        owner = Path(directory) / folder / status["round"] / "process.json"
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


def launch_round(directory, manifest):
    from .storage import atomic_json

    status = {
        **manifest,
        "state": "starting",
        "round": directory.name,
        "pid": os.getpid(),
        "process_created": psutil.Process().create_time(),
        "results": [],
        "completed": 0,
        "total": manifest["games"]
        * len(manifest.get("opponents", ["easy", "medium", "hard"])),
    }
    atomic_json(directory / "status.json", status)
    atomic_json(directory.parent / "status.json", status)
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
            "depth": self.config.depth,
            "max_commands": self.config.max_commands,
            "device": self.config.device,
        }
        atomic_json(directory / "round.json", manifest)
        return launch_round(directory, manifest)

    def close(self):
        # 只等小型权重文件和进程启动完成，不等待对弈。
        self.writer.shutdown(wait=True)


class HistoryEvaluationSchedule:
    """归档时冻结双方，逐轮执行；尚未开始的轮次保留在磁盘，续训后接着排队。"""

    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.config = config
        self.root = self.directory / "history-evaluations"
        self.writer = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="history-evaluation"
        )
        self.preparing = []
        self.process = None
        self.next_at = 0

    def enqueue(self, entries):
        trained = [e["version"] for e in entries if e["version"] > 0]
        if len(trained) < 2 or not self.config.history_evaluation_games:
            return
        version = trained[-1]
        opponents = list(dict.fromkeys([trained[0], trained[-2]]))
        # 把配置也冻结在归档时；稍后启动不受运行中参数修改影响。
        manifest = dict(
            kind="history",
            version=version,
            opponents=opponents,
            started_at=time.time(),
            seed=self.config.seed ^ 0xA17E,
            games=self.config.history_evaluation_games,
            simulations=max(1, self.config.simulations),
            depth=self.config.depth,
            max_commands=self.config.max_commands,
            device=self.config.device,
        )
        self.preparing.append(self.writer.submit(self.prepare, manifest))

    def prepare(self, manifest):
        from .storage import atomic_json

        job = self.root / f"round-{time.time_ns()}"
        job.mkdir(parents=True)
        for version, name in [(manifest["version"], "model.pt")] + [
            (v, f"opponent-{v}.pt") for v in manifest["opponents"]
        ]:
            shutil.copyfile(
                self.directory / "models" / f"model-{version:08d}.pt", job / name
            )
        atomic_json(
            job / "status.json", {**manifest, "state": "queued", "round": job.name}
        )
        # 清单最后写入，排队读取者只会看到已冻结完整的轮次。
        atomic_json(job / "round.json", manifest)

    def tick(self):
        from .storage import atomic_json

        try:
            for future in list(self.preparing):
                if future.done():
                    self.preparing.remove(future)
                    future.result()
            if time.time() < self.next_at:
                return
            self.next_at = time.time() + 5
            previous = read_evaluation(self.directory, "history-evaluations")
            if (self.process is not None and self.process.poll() is None) or (
                previous and previous["state"] in ("starting", "running")
            ):
                return
            for path in sorted(self.root.glob("round-*/round.json")):
                status = json.loads(
                    path.with_name("status.json").read_text(encoding="utf-8")
                )
                if status["state"] == "queued":
                    manifest = json.loads(path.read_text(encoding="utf-8"))
                    self.process = launch_round(path.parent, manifest)
                    return
        except Exception as error:
            self.root.mkdir(exist_ok=True)
            atomic_json(
                self.root / "status.json",
                {
                    "kind": "history",
                    "state": "failed",
                    "started_at": time.time(),
                    "error": str(error),
                },
            )

    def close(self):
        self.writer.shutdown(wait=True)
        self.next_at = 0
        self.tick()
