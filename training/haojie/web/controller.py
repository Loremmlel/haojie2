"""浏览器之外持有训练进程、观测历史和保存重启状态。"""

from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

import psutil

from .settings import defaults, validate
from ..runtime.evaluation import read_evaluation


class Controller:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.process = None
        self.reader = None
        self.name = None
        self.config = defaults()
        self.state = "idle"
        self.message = "选择已有训练，或新建一次训练。"
        self.stats = {}
        self.history = deque(maxlen=17280)
        self.logs = deque(maxlen=200)
        self.revision = self.sequence = 0
        self.session = str(time.time_ns())
        self.restart = None
        self.closing = threading.Event()
        self.last_status = self.rate_base = None
        self.rates = {}
        self.loss_update = None
        self.processes = {}
        self.sampler = threading.Thread(
            target=self._sample_loop, daemon=True, name="training-monitor"
        )
        self.sampler.start()

    def directory(self, name):
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[\w\-]{1,64}", name)
            or name in (".", "..")
        ):
            raise ValueError("训练名称使用 1–64 个汉字、字母、数字、下划线或短横线")
        path = self.root / name
        if path.resolve().parent != self.root:
            raise ValueError("训练目录须位于指定根目录内")
        return path

    def runs(self):
        result = []
        for path in self.root.iterdir():
            if (
                not path.is_dir()
                or path.is_symlink()
                or path.resolve().parent != self.root
            ):
                continue
            status = path / "status.json"
            if (path / "checkpoint.pt").is_file() or status.is_file():
                result.append(
                    dict(
                        name=path.name,
                        updated=(
                            status.stat().st_mtime
                            if status.exists()
                            else path.stat().st_mtime
                        ),
                    )
                )
        return sorted(result, key=lambda row: row["updated"], reverse=True)

    def snapshot(self, since=0):
        with self.lock:
            alive = self.process is not None and self.process.poll() is None
            return dict(
                run=self.name,
                state=self.state,
                message=self.message,
                revision=self.revision,
                session=self.session,
                config=self.config,
                stats=dict(self.stats),
                running=alive,
                checkpoint=bool(
                    self.name and (self.directory(self.name) / "checkpoint.pt").exists()
                ),
                history=[p for p in self.history if p["id"] > since],
                sequence=self.sequence,
                logs=list(self.logs),
                runs=self.runs(),
                cpu_count=psutil.cpu_count(),
                memory_gib=round(psutil.virtual_memory().total / 1024**3, 1),
                evaluation=(
                    read_evaluation(self.directory(self.name)) if self.name else None
                ),
            )

    def control(self, action, name=None, config=None, revision=None, session=None):
        with self.lock:
            if (revision, session) != (self.revision, self.session):
                raise ValueError("运行状态已在其他页面改变，请刷新状态后重试")
            alive = self.process is not None and self.process.poll() is None
            if self.state in ("starting", "stopping", "restarting"):
                raise ValueError("正在处理上一操作，请等待完成")
            if action == "select":
                if alive:
                    raise ValueError("请先保存并停止当前训练")
                path = self.directory(name)
                status = json.loads((path / "status.json").read_text(encoding="utf-8"))
                self.config = validate(status["config"])
                self.name, self.stats = name, status
                self.history.clear()
                self.logs.clear()
                self.state, self.message = "saved", "已选择训练，可从最近恢复点继续。"
            elif action in ("start", "resume"):
                if alive:
                    raise ValueError("已有训练进程运行")
                resume = action == "resume"
                name = self.name if resume else name
                if resume and not name:
                    raise ValueError("先选择需要继续的训练")
                values = validate(config or {}, self.config if resume else None)
                path = self.directory(name)
                if resume:
                    if not (path / "checkpoint.pt").is_file():
                        raise ValueError("该训练尚无完整恢复点")
                elif path.exists():
                    raise ValueError("训练名称已存在，请换一个名称或选择继续训练")
                else:
                    path.mkdir()
                    self.history.clear()
                    self.logs.clear()
                    self.stats = {}
                self._launch(name, values, resume)
            elif action in ("stop", "apply"):
                if not alive:
                    raise ValueError("当前没有运行中的训练")
                self.restart = (
                    validate(config or {}, self.config) if action == "apply" else None
                )
                (self.directory(self.name) / "STOP").touch()
                self.state = "restarting" if self.restart else "stopping"
                self.message = (
                    "正在保存，随后应用参数并继续。"
                    if self.restart
                    else "正在保存模型、回放和在途对局…"
                )
                self.logs.append(datetime.now().strftime("%H:%M:%S ") + self.message)
            else:
                raise ValueError("未知操作")
            self.revision += 1
            return self.snapshot()

    def _launch(self, name, config, resume):
        path = self.directory(name)
        (path / "STOP").unlink(missing_ok=True)
        command = [
            sys.executable,
            "-u",
            "-B",
            "-m",
            "training.haojie",
            "run",
            "--directory",
            str(path),
        ]
        if resume:
            command.append("--resume")
        for key, value in config.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[3],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.process, self.name, self.config = process, name, config
        self.state, self.message = "starting", "正在加载训练环境和模型…"
        self.logs.append(
            datetime.now().strftime("%H:%M:%S ")
            + f"{'继续' if resume else '开始'}训练：{config['workers']} 个线程 × {config['games_per_worker']} 局，学习批量 {config['batch']}。"
        )
        self.last_status = self.rate_base = None
        self.rates = {}
        self.loss_update = None
        self.processes.clear()
        self.reader = threading.Thread(
            target=self._read, args=(process,), daemon=True, name="training-output"
        )
        self.reader.start()

    def _read(self, process):
        for line in process.stdout:
            try:
                data = json.loads(line)
            except (ValueError, TypeError):
                with self.lock:
                    self.logs.append(line.rstrip()[:2000])
                continue
            if not isinstance(data, dict):
                continue
            now = time.time()
            with self.lock:
                if data.get("event") == "started":
                    self.config = data["config"]
                    if self.state == "starting":
                        self.state, self.message = "running", "训练已启动。"
                        self.logs.append(
                            datetime.now().strftime("%H:%M:%S ") + self.message
                        )
                if "status" in data:
                    if self.rate_base and data["status"] == "running":
                        previous_time, previous = self.rate_base
                        elapsed = now - previous_time
                        self.rates = {
                            key: max(0, data.get(counter, 0) - previous.get(counter, 0))
                            / max(elapsed, 0.001)
                            for key, counter in (
                                ("cps", "commands"),
                                ("sps", "trained_samples"),
                                ("ups", "updates"),
                            )
                        }
                    self.rate_base = (now, data)
                    self.last_status, self.stats = now, data
                    if data["status"] == "failed":
                        self.message = data.get("error", "训练异常退出")
                        self.logs.append(self.message)
        code = process.wait()
        process.stdout.close()
        with self.lock:
            self._point({})
            config, self.restart = self.restart, None
            if code == 0 and config and not self.closing.is_set():
                self._launch(self.name, config, True)
            else:
                self.state = "saved" if code == 0 else "error"
                self.message = (
                    "已保存，可继续训练。"
                    if code == 0
                    else f"训练退出（代码 {code}）。"
                    + (self.logs[-1] if self.logs else "查看运行记录定位原因。")
                )
                self.logs.append(datetime.now().strftime("%H:%M:%S ") + self.message)
            self.revision += 1

    def _point(self, values):
        self.sequence += 1
        self.history.append(
            dict(id=self.sequence, t=time.time(), gap=not bool(values), **values)
        )

    def _sample_loop(self):
        while not self.closing.wait(5):
            with self.lock:
                process = self.process
                if process is None or process.poll() is not None:
                    continue
            resources = {}
            try:
                parent = psutil.Process(process.pid)
                live = {p.pid: p for p in [parent, *parent.children(recursive=True)]}
                initial = not self.processes
                self.processes = {
                    pid: self.processes.get(pid, p) for pid, p in live.items()
                }
                resources["cpu"] = sum(
                    p.cpu_percent() for p in self.processes.values()
                ) / (psutil.cpu_count() or 1)
                if initial:
                    resources["cpu"] = None
                resources["ram"] = (
                    sum(p.memory_info().rss for p in self.processes.values()) / 1024**3
                )
            except psutil.Error:
                pass
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                        "-i",
                        "0",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                gpu, used, total = [
                    float(v.strip()) for v in result.stdout.strip().split(",")
                ]
                resources.update(gpu=gpu, vram=used / 1024, vram_total=total / 1024)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            with self.lock:
                if process is not self.process or process.poll() is not None:
                    continue
                fresh = (
                    self.last_status
                    and time.time() - self.last_status <= 15
                    and self.state == "running"
                )
                if fresh:
                    resources.update(self.rates)
                    resources["replay"] = self.stats.get("replay_bytes", 0) / 1024**3
                    if self.stats.get("updates") != self.loss_update:
                        loss = self.stats.get("loss", {})
                        resources.update(
                            policy=loss.get("policy"),
                            value=loss.get("value"),
                            aux=loss.get("auxiliary"),
                        )
                        self.loss_update = self.stats.get("updates")
                self._point(resources)

    def close(self):
        self.closing.set()
        with self.lock:
            self.restart = None
            process = self.process
            if process and process.poll() is None:
                (self.directory(self.name) / "STOP").touch()
        if process:
            process.wait(timeout=60)
        if self.reader:
            self.reader.join(timeout=5)
        self.sampler.join(timeout=5)
