"""原生训练的 Colab 观测面板；只读状态文件并写入已有控制协议。"""

import json
import math
import os
import subprocess
import threading
import time
import uuid
from collections import deque
from pathlib import Path

ASSETS = Path(__file__).with_name("assets")
TOKENS = Path(__file__).resolve().parents[2] / "src/ui/styles/tokens.css"


class Resources:
    def __init__(self, device):
        self.previous = None
        self.device = device

    def sample(self):
        result = {}
        if Path("/proc/stat").exists():
            allowed = os.sched_getaffinity(0)
            total = idle = 0
            for line in Path("/proc/stat").read_text().splitlines():
                fields = line.split()
                if fields[0].startswith("cpu") and fields[0][3:].isdigit():
                    if int(fields[0][3:]) not in allowed:
                        continue
                    values = [int(v) for v in fields[1:9]]
                    total += sum(values)
                    idle += values[3] + values[4]
            if self.previous and total > self.previous[0]:
                result["cpu"] = 100 * (
                    1 - (idle - self.previous[1]) / (total - self.previous[0])
                )
            self.previous = total, idle
            memory = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, value, *_ = line.split()
                memory[key.rstrip(":")] = int(value) * 1024
            result["ram"] = (memory["MemTotal"] - memory["MemAvailable"]) / 2**30
            result["ram_total"] = memory["MemTotal"] / 2**30
        if str(self.device).startswith("cuda"):
            index = int(self.device.partition(":")[2] or 0)
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            selected = (
                visible[index] if visible[0] and index < len(visible) else str(index)
            )
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        selected,
                        "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=2,
                    stderr=subprocess.DEVNULL,
                )
                values = output.splitlines()[0].split(",")
                for key, value, divisor in zip(
                    ("gpu", "vram", "vram_total", "power"), values, (1, 1024, 1024, 1)
                ):
                    try:
                        number = float(value.strip()) / divisor
                        if math.isfinite(number):
                            result[key] = number
                    except ValueError:
                        pass
            except (OSError, subprocess.SubprocessError, IndexError):
                pass
        return result


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def tail(path):
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, stream.seek(0, 2) - 8000))
        return stream.read().decode("utf-8", errors="replace")


class Panel:
    def __init__(self, directory, process=None, configuration=None):
        self.directory = Path(directory)
        self.process = process
        self.configuration = configuration or {}
        self.identity = uuid.uuid4().hex
        self.resources = Resources(self.configuration.get("device", "cpu"))
        self.history = deque(maxlen=720)
        self.previous = None
        self.sampled = None
        self.last_resource = {}
        self.resource_window = []
        self.origin = None
        self.observed_at = None
        self.sample_error = None
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.thread = None
        self.saved_before = None
        self.stopping = False
        self.evaluation = None

    def read_state(self):
        """读取宿主状态；文件时间与资源采集时间分开，不将旧状态当成新观测。"""
        run = self.directory
        source = run / (
            "window.json" if (run / "window.json").exists() else "status.json"
        )
        metrics = read_json(source) or {}
        code = self.process.poll() if self.process is not None else None
        failure = tail(run / "failure.txt")
        if failure or code not in (None, 0):
            phase = "failed"
        elif self.process is None:
            phase = "unknown" if run.exists() else "not_started"
        elif code == 0:
            phase = "stopped"
        elif self.stopping or (run / "window.json").exists():
            phase = "stopping"
        elif not metrics:
            phase = "initializing"
        elif time.time() - source.stat().st_mtime > 15:
            phase = "stale"
        else:
            phase = "running"
        age = max(0, time.time() - source.stat().st_mtime) if metrics else None
        return metrics, phase, age, code, failure

    def sample(self):
        """后台单次观测：资源每秒采集，五秒入图；缺测后重新建立速率基线。"""
        resource = self.resources.sample()
        try:
            metrics, phase, age, _, _ = self.read_state()
            error = None
        except (OSError, ValueError, KeyError) as exc:
            metrics, phase, age = {}, "unknown", None
            error = str(exc)
        now = time.monotonic()
        with self.lock:
            self.last_resource = resource
            self.observed_at = time.time()
            self.sample_error = error
            self.resource_window.append((now, resource))
            if (
                self.sampled is not None
                and now - self.sampled < 5
                and phase not in ("stopped", "failed")
            ):
                return
            seconds = metrics.get("seconds")
            if (
                self.previous
                and seconds is not None
                and seconds < self.previous["seconds"]
            ):
                self.history.clear()
                self.previous = None
                self.origin = None
            if self.origin is None and seconds is not None:
                self.origin = now - seconds
            if self.origin is None:
                self.resource_window.clear()
                return
            point = {"seconds": now - self.origin, **resource}
            for field in ("cpu", "gpu"):
                total = weighted = 0
                previous = self.sampled
                for stamp, reading in self.resource_window:
                    weight = max(0, stamp - previous) if previous is not None else 1
                    if field in reading:
                        total += weight
                        weighted += reading[field] * weight
                    previous = stamp
                if total:
                    point[field] = weighted / total
            self.resource_window.clear()
            fresh = (
                metrics
                and phase in ("running", "stopping", "stopped")
                and (not self.previous or seconds > self.previous["seconds"])
            )
            if fresh:
                point.update(
                    {
                        "updates": metrics.get("loss_update"),
                        "loss": metrics.get("latest_loss"),
                        "ready": metrics.get("ready_steps"),
                    }
                )
                if self.previous:
                    interval = seconds - self.previous["seconds"]
                    if 0 < interval <= 15:
                        point["interval"] = interval
                        for name, key in (
                            ("actor", "inference_nodes"),
                            ("commands", "commands"),
                            ("consume", "consumed_within_lag"),
                        ):
                            if key in metrics and key in self.previous:
                                point[name] = (
                                    max(0, metrics[key] - self.previous[key]) / interval
                                )
                self.previous = metrics
            else:
                point["gap"] = (
                    "状态文件读取失败"
                    if error
                    else "状态文件更新延迟"
                    if age is not None and age > 15
                    else "等待新的训练状态"
                )
                # 保留最后的宿主时间用于识别重复状态，恢复时不跨缺口计算速率。
            if self.history and self.history[-1].get("gap"):
                for field in ("actor", "commands", "consume", "interval"):
                    point.pop(field, None)
            self.history.append(point)
            self.sampled = now

    def start(self):
        """采样线程归面板运行所有；网页关闭或暂停刷新不会停止它。"""
        if self.thread and self.thread.is_alive():
            return
        self.closed.clear()

        def loop():
            while not self.closed.is_set():
                started = time.monotonic()
                try:
                    self.sample()
                except (OSError, ValueError, KeyError) as error:
                    with self.lock:
                        self.sample_error = str(error)
                if self.process is not None and self.process.poll() is not None:
                    break
                self.closed.wait(max(0.05, 1 - (time.monotonic() - started)))

        self.thread = threading.Thread(
            target=loop, name="haojie-panel-sampler", daemon=True
        )
        self.thread.start()

    def close(self):
        self.closed.set()
        if self.thread:
            self.thread.join(timeout=3)
        if self.evaluation:
            self.evaluation.close()

    def snapshot(self):
        run = self.directory
        metrics, phase, age, code, failure = self.read_state()
        configuration = read_json(run / "configuration.json") or self.configuration
        saved = run / "saved.json"
        saved_at = saved.stat().st_mtime if saved.exists() else None
        save_pending = self.saved_before is not None
        if save_pending and saved_at is not None and saved_at > self.saved_before:
            self.saved_before = None
            save_pending = False
        with self.lock:
            history = list(self.history)
            resources = self.last_resource.copy()
            sampled_at, sample_error = self.observed_at, self.sample_error
        return {
            "phase": phase,
            "metrics": metrics,
            "configuration": configuration,
            "evaluation": self.evaluation.status() if self.evaluation else None,
            "history": history,
            "resources": resources,
            "sampled_at": sampled_at,
            "sample_error": sample_error,
            "status_age": age,
            "saved_at": saved_at,
            "save_pending": save_pending,
            "checkpoint": str(run / "checkpoint.pt"),
            "directory": str(run),
            "log": failure or tail(run.with_suffix(".log")),
            "exit_code": code,
            "can_stop": self.process is not None
            and code is None
            and run.exists()
            and phase not in ("stopping", "failed"),
            "can_control": self.process is not None
            and code is None
            and bool(metrics)
            and phase == "running",
        }

    def call(self, identity, action="status", value=None):
        if identity != self.identity:
            return {"closed": True}
        try:
            if action not in (
                "status",
                "save",
                "stop",
                "resize",
                "evaluate",
                "eval_cancel",
                "eval_config",
            ):
                raise ValueError("不支持此操作")
            if action == "status":
                return self.snapshot()
            if action in ("evaluate", "eval_cancel", "eval_config"):
                if self.evaluation is None:
                    raise ValueError("测评执行器尚未配置，请重新执行安装和面板单元。")
                if action == "evaluate":
                    self.evaluation.request()
                elif action == "eval_cancel":
                    self.evaluation.cancel()
                else:
                    self.evaluation.configure(value)
                result = self.snapshot()
                result["message"] = self.evaluation.message
                return result
            status = self.snapshot()
            if not status["can_stop" if action == "stop" else "can_control"]:
                raise ValueError("当前状态无法提交控制，请等待初始化完成或查看日志。")
            if action == "resize" and (type(value) is not int or not 0 <= value <= 128):
                raise ValueError("并发局数须为 0–128 的整数")
            if action == "save" and self.saved_before is not None:
                raise ValueError("保存请求正在处理，请等待保存完成。")
            target = self.directory / action
            temporary = target.with_suffix(".tmp")
            temporary.write_text(
                str(value) if action == "resize" else "1", encoding="utf-8"
            )
            os.replace(temporary, target)
            if action in ("save", "stop"):
                self.saved_before = status["saved_at"] or 0
            if action == "stop":
                self.stopping = True
                if self.evaluation:
                    self.evaluation.configure(
                        {**self.evaluation.settings, "minutes": 0}
                    )
                    self.evaluation.cancel()
            result = self.snapshot()
            result["message"] = {
                "save": "保存请求已送达，等待检查点写入完成。",
                "stop": "停止请求已送达，正在保存并退出。",
                "resize": f"并发请求已送达：{value} 局，等待宿主应用。",
            }[action]
            return result
        except (OSError, ValueError, KeyError) as error:
            return {"error": str(error)}

    def render(self):
        style = TOKENS.read_text(encoding="utf-8") + ASSETS.joinpath(
            "panel.css"
        ).read_text(encoding="utf-8")
        markup = ASSETS.joinpath("panel.html").read_text(encoding="utf-8")
        script = ASSETS.joinpath("panel.js").read_text(encoding="utf-8")
        return (
            f"<style>{style}</style>"
            + markup.replace("__PANEL_ID__", self.identity)
            + "<script>"
            + script.replace("__PANEL_ID__", json.dumps(self.identity))
            + "</script>"
        )


_current = None


def show(directory, process=None, configuration=None, native=None, python=None):
    """重开面板保留同一运行的曲线，旧面板失效；不会启动、停止或等待训练进程。"""
    global _current
    from google.colab import output
    from IPython.display import HTML, JSON, display

    if _current is None or _current.directory != Path(directory):
        if _current:
            _current.close()
        _current = Panel(directory, process, configuration)
    else:
        _current.identity = uuid.uuid4().hex
        _current.process = process
    if native and not _current.evaluation:
        from training.evaluation.control import EvaluationControl

        _current.evaluation = EvaluationControl(
            directory,
            process,
            native,
            python,
            (configuration or {}).get("device", "cuda"),
        )
    elif _current.evaluation:
        _current.evaluation.process = process
    output.register_callback(
        "haojie.native_panel", lambda *args: JSON(_current.call(*args))
    )
    _current.start()
    display(HTML(_current.render()))
    return _current
