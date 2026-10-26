"""笔记本内的后台测评调度；页面关闭后仍执行，单次只保留一个在途任务。"""

import json
import os
import subprocess
import threading
import time
from pathlib import Path


class EvaluationControl:
    def __init__(self, directory, process, native, python, device):
        self.directory = Path(directory)
        self.process, self.native, self.python, self.device = (
            process,
            native,
            python,
            device,
        )
        self.settings = {"minutes": 30, "pairs": 1, "seconds": 120, "temperature": 0.5}
        self.next_due = time.time() + 1800
        self.job = self.child = self.log = None
        self.pending = False
        self.closed = False
        self.message = "每 30 分钟测评一次，可修改间隔或关闭。"
        self.lock = threading.RLock()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def running(self):
        return self.process is not None and self.process.poll() is None

    def configure(self, value):
        for key, low, high in (
            ("minutes", 0, 1440),
            ("pairs", 1, 20),
            ("seconds", 1, 3600),
        ):
            if type(value.get(key)) is not int or not low <= value[key] <= high:
                raise ValueError(f"{key} 须为 {low}–{high} 的整数")
        temperature = value.get("temperature", self.settings["temperature"])
        if type(temperature) not in (int, float) or not 0 <= temperature <= 2:
            raise ValueError("测评温度须为 0–2，0 为贪心选招")
        with self.lock:
            self.settings = {key: value[key] for key in ("minutes", "pairs", "seconds")}
            self.settings["temperature"] = temperature
            self.next_due = time.time() + value["minutes"] * 60
            self.message = (
                "测评计划已更新。"
                if value["minutes"]
                else "已关闭定期测评，仍可手工测评。"
            )

    def request(self):
        with self.lock:
            if self.pending or self.child:
                raise ValueError("已有测评正在等待或执行")
            if not self.running() and not (self.directory / "checkpoint.pt").exists():
                raise ValueError("请先启动训练或提供已有检查点")
            self.pending = True
            self.message = "等待完整检查点，随后冻结模型测评。"

    def cancel(self):
        with self.lock:
            self.pending = False
            self.next_due = time.time() + self.settings["minutes"] * 60
            if self.child and self.job:
                (self.job / "cancel").write_text("1", encoding="utf-8")
            self.message = "测评取消请求已提交；训练不受影响。"

    def status(self):
        with self.lock:
            result = None
            if self.job and (self.job / "status.json").exists():
                result = json.loads(
                    (self.job / "status.json").read_text(encoding="utf-8")
                )
            return {
                "settings": self.settings.copy(),
                "pending": self.pending,
                "active": self.child is not None,
                "message": self.message,
                "next_due": self.next_due if self.settings["minutes"] else None,
                "report": result,
                "directory": str(self.job) if self.job else None,
            }

    def launch(self):
        run = self.directory
        saved = run / "saved.json"
        before = saved.stat().st_mtime_ns if saved.exists() else 0
        if self.running():
            temporary = run / "save.evaluation.tmp"
            temporary.write_text("1", encoding="utf-8")
            os.replace(temporary, run / "save")
            deadline = time.monotonic() + 300
            while self.pending and not self.closed and time.monotonic() < deadline:
                if saved.exists() and saved.stat().st_mtime_ns > before:
                    break
                if not self.running():
                    break
                time.sleep(0.2)
            else:
                if self.pending and not self.closed:
                    raise RuntimeError("等待保存超过 300 秒，请检查训练状态")
                return
        with self.lock:
            if not self.pending or self.closed:
                return
            self.job = run / "evaluations" / time.strftime("%Y%m%d-%H%M%S")
            self.job.mkdir(parents=True, exist_ok=False)
            self.log = (self.job / "runner.log").open("w", encoding="utf-8")
            try:
                self.child = subprocess.Popen(
                    [
                        self.python,
                        "-B",
                        "-m",
                        "training.evaluation.run",
                        "--checkpoint",
                        str(run / "checkpoint.pt"),
                        "--directory",
                        str(self.job),
                        "--device",
                        self.device,
                        "--pairs",
                        str(self.settings["pairs"]),
                        "--seconds",
                        str(self.settings["seconds"]),
                        "--temperature",
                        str(self.settings["temperature"]),
                        "--",
                        *self.native,
                    ],
                    cwd=Path(__file__).resolve().parents[2],
                    stdout=self.log,
                    stderr=subprocess.STDOUT,
                )
            except BaseException:
                self.log.close()
                self.log = None
                raise
            self.pending = False
            self.message = "测评运行中，训练继续。"

    def loop(self):
        while not self.closed or self.child:
            try:
                with self.lock:
                    if self.child and self.child.poll() is not None:
                        code = self.child.returncode
                        self.child = None
                        self.log.close()
                        self.log = None
                        self.message = (
                            "测评已结束。"
                            if code == 0
                            else f"测评异常退出（{code}），请查看 {self.job / 'runner.log'}"
                        )
                        self.next_due = time.time() + self.settings["minutes"] * 60
                    if (
                        not self.closed
                        and self.running()
                        and self.settings["minutes"]
                        and time.time() >= self.next_due
                        and not self.child
                    ):
                        self.pending = True
                    launch = self.pending and self.child is None
                if launch:
                    self.launch()
            except Exception as error:
                with self.lock:
                    self.pending = False
                    self.message = str(error)
                    self.next_due = time.time() + self.settings["minutes"] * 60
            time.sleep(1)

    def close(self):
        self.cancel()
        self.closed = True
        self.thread.join(timeout=2)
