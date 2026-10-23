"""常驻采样/推理进程与学习器的有界交接；跨进程只传 CPU 权重和真实结案。"""

import io
import json
import multiprocessing as mp
import os
import pickle
import queue
import time
import traceback
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

from ..data import synthetic_batch
from ..model import ModelConfig, PolicyValueNet
from ..native.execution import PolicyInference
from ..native.pool import Pool
from .config.resources import available_memory
from .sampling import MemoryJobs

SCHEDULE_COUNTS = ("tasks", "matchups", "history_fallback")
PUBLICATION_FIELDS = (
    "published_model",
    "root_snapshots",
    "coalesced_publications",
    "oldest_root_update_lag",
)


def encode_weights(value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def load_model(weights, device):
    model = PolicyValueNet(ModelConfig(**weights["config"]))
    model.load_state_dict(weights["model"])
    return model.to(device).eval().requires_grad_(False)


def latest_status(channel, value):
    """状态可合并；终局使用另一条不可丢弃的有界队列。"""
    try:
        channel.put_nowait(value)
    except queue.Full:
        try:
            channel.get_nowait()
        except queue.Empty:
            return
        try:
            channel.put_nowait(value)
        except queue.Full:
            pass


class ProducerJobs(MemoryJobs):
    def __init__(self, controller, current, historical, output, stop, continuous):
        super().__init__(controller, current, historical, continuous=continuous)
        self.output, self.stop_event = output, stop
        self.backpressure_seconds = 0.0
        self.pressure_since = None
        self.transfer_seconds = 0.0
        self.completed = deque()
        self.blocked_slots = set()
        if continuous:
            self.target = float("inf")

    def release_unused(self):
        # 常驻接纳没有窗口间隙；下一局的已选历史计划也持有路由，切换计划时才释放。
        super().release_unused((self.historical[1],) if self.historical else ())

    def emit(self, kind, task=None, message=None):
        # 不触碰 trace.rows；保留原帧蓄水，学习器在真实终局才物化最终样本。
        task = (
            {
                key: task[key]
                for key in ("id", "context", "trajectory", "requests", "began", "slot", "pid")
            }
            if task is not None
            else None
        )
        self.completed.append((kind, task, message))
        if task is not None:
            self.blocked_slots.add(task["slot"])
        self.flush()

    def admission_open(self, slot):
        return slot not in self.blocked_slots

    def pressure_seconds(self):
        now = time.perf_counter()
        if not self.completed and self.pressure_since is not None:
            self.backpressure_seconds += now - self.pressure_since
            self.pressure_since = None
        elif self.completed and self.pressure_since is None:
            self.pressure_since = now
        return self.backpressure_seconds + (
            now - self.pressure_since if self.pressure_since is not None else 0
        )

    def waiting_for_admission(self):
        return bool(self.completed)

    def flush(self, block=False):
        """背压停止补局；每个已结案槽保留自身轨迹，不阻塞其他在途局的 GPU 服务。"""
        began = time.perf_counter()
        while self.completed:
            if not block and self.output.full():
                break
            payload = pickle.dumps(self.completed[0], protocol=5)
            try:
                self.output.put(payload, timeout=0.1 if block else 0)
            except queue.Full:
                if not block:
                    break
                if not mp.parent_process().is_alive():
                    self.stop_event.set()
                    break
            else:
                _, task, _ = self.completed.popleft()
                if task is not None:
                    self.blocked_slots.remove(task["slot"])
        self.pressure_seconds()
        self.transfer_seconds += time.perf_counter() - began

    def finish(self, task, message):
        self.finished += 1
        self.result = message
        self.emit("done", task, message)

    def unfinished(self, task):
        self.emit("unknown", task)

    def resource_error(self, reason):
        self.emit("resource", message=reason)


def sampling_process(bootstrap, publications, output, status, stop, target, admission):
    """此进程唯一拥有冻结模型和环境池；维护回调不排空 GPU，不运行优化器。"""
    pool = jobs = None
    error = None
    profile_report = {}
    profile_options = bootstrap.get("profile")
    profile_path = None
    try:
        torch.set_num_threads(1)
        initial = torch.load(
            io.BytesIO(bootstrap["weights"]), weights_only=True, map_location="cpu"
        )
        device = torch.device(bootstrap["device"])
        if device.type == "cuda":
            os.environ.setdefault("HAOJIE_HOST_WORKERS", "auto")
        model = load_model(initial, device)
        warmup = {
            key: value.to(device) for key, value in synthetic_batch(model.config, 8, 64, 24).items()
        }
        with torch.no_grad():
            for _ in range(3):
                model.policy_logits(warmup)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        del warmup
        c = SimpleNamespace(
            config=bootstrap["config"],
            device=device,
            counts=bootstrap["counts"],
            starts=bootstrap["starts"],
            generation=initial["generation"],
            trainer=SimpleNamespace(model=model, updates=initial["updates"]),
            cancel=stop,
            eval_cancel=stop,
        )
        pool = c.pool = Pool(
            bootstrap["engine"],
            target.value,
            device,
            batch_wait_ms=c.config["batch_wait_ms"],
            max_batch_size=c.config["inference_batch_size"],
            inference=(PolicyInference(model, device), initial["version"]),
            frame_limit=bootstrap["frame_limit"],
        )

        def historical(weights):
            if weights is None or weights["version"] == initial["version"]:
                return None
            if jobs is not None:
                versions = [v for v in jobs.inferences if v != jobs.current]
                if weights["version"] in versions:
                    return jobs.inferences[weights["version"]].model, weights["version"]
                if len(versions) >= c.config["history_size"]:
                    version = versions[c.generation % len(versions)]
                    return jobs.inferences[version].model, version
            return load_model(weights, device), weights["version"]

        jobs = ProducerJobs(
            c,
            initial["version"],
            historical(initial["history"]),
            output,
            stop,
            bootstrap["continuous"],
        )
        pending = None
        next_status = 0.0

        def publish_status():
            latest_status(
                status,
                {
                    "metrics": {
                        **pool.metrics(),
                        **jobs.publication_metrics(),
                        "sampling_process": mp.current_process().pid,
                        "sampling_backend": "process-pipeline",
                        "inflight_bytes": sum(t["trajectory"].used for t in pool.slots.values()),
                        "inflight_samples": sum(t["trajectory"].seen for t in pool.slots.values()),
                        "historical_models": max(0, len(jobs.inferences) - 1),
                        "terminal_queue_capacity": 4,
                        "terminal_backpressure_seconds": jobs.pressure_seconds(),
                        "terminal_transfer_seconds": jobs.transfer_seconds,
                        "producer_finished": jobs.finished,
                        "pending_terminals": len(jobs.completed),
                        "pending_terminal_bytes": sum(
                            task["trajectory"].used
                            for _, task, _ in jobs.completed
                            if task is not None
                        ),
                        "learner_updates_seen": c.trainer.updates,
                        "gpu_allocated": torch.cuda.memory_allocated(device)
                        if device.type == "cuda"
                        else 0,
                        "gpu_reserved": torch.cuda.memory_reserved(device)
                        if device.type == "cuda"
                        else 0,
                    },
                    "schedule": {key: c.counts[key] for key in SCHEDULE_COUNTS if key in c.counts},
                },
            )

        def maintain():
            nonlocal pending, next_status, initial
            if not mp.parent_process().is_alive():
                stop.set()
            jobs.flush()
            pool.set_target(target.value)
            try:
                value = publications.get_nowait()
            except queue.Empty:
                pass
            else:
                if pending is not None:
                    jobs.coalesced_publications += 1
                pending = torch.load(io.BytesIO(value), weights_only=True, map_location="cpu")
                c.trainer.updates = pending["updates"]
                c.generation = pending["generation"]
            if pending is not None:
                jobs.release_unused()
                if len(jobs.root_versions) < 4:
                    initial = pending
                    model = load_model(pending, device)
                    c.trainer.model = model
                    jobs.historical = None
                    jobs.release_unused()
                    jobs.next_window(
                        pending["version"], historical(pending["history"]), snapshot=model
                    )
                    if jobs.continuous:
                        jobs.target = float("inf")
                    pool.model_hash = jobs.version
                    if pool.input_processes is not None:
                        pool.input_processes.collection_version(jobs.collection_version())
                    pending = None
            now = time.perf_counter()
            if now >= next_status:
                publish_status()
                next_status = now + 0.5

        publish_status()
        # 握手与模型加载属于启动；父进程开启正式采样后才接纳第一局。
        while not admission.wait(0.1) and not stop.is_set():
            if not mp.parent_process().is_alive():
                stop.set()
        if not stop.is_set():
            profile_context = nullcontext()
            if profile_options is not None:
                from ..diagnostics import profiling

                mode, directory, wait_batches = profile_options
                profile_path = Path(directory)
                profile_path.mkdir(parents=True, exist_ok=True)
                profile_context = profiling(mode, profile_path, profile_report, wait_batches)
            with profile_context:
                pool.run(jobs, cancel_event=stop, maintenance=maintain)
        publish_status()
    except BaseException:
        error = traceback.format_exc()
    finally:
        try:
            if pool is not None:
                pool.close()
            if jobs is not None:
                jobs.close()
                jobs.flush(block=True)
                publish_status()
        except BaseException:
            error = error or traceback.format_exc()
        if profile_path is not None:
            (profile_path / "report.json").write_text(
                json.dumps(profile_report, indent=2) + "\n", encoding="utf-8"
            )
        # 结案队列保持 FIFO；父进程在关闭期间继续取走，直到读到最终标记。
        if mp.parent_process().is_alive():
            output.put(pickle.dumps(("closed", None, error), protocol=5))
        else:
            output.cancel_join_thread()


class SamplingPipeline:
    """学习进程的轻量句柄；只消费结案和状态，不参与逐请求的组批与回传。"""

    def __init__(self, controller, count):
        self.context = mp.get_context("spawn")
        self.publications = self.context.Queue(maxsize=1)
        self.output = self.context.Queue(maxsize=4)
        self.status = self.context.Queue(maxsize=1)
        self.stop = self.context.Event()
        self.admission = self.context.Event()
        self.target = self.context.Value("i", count)
        self.target_environments = count
        self.pending = None
        self.closed = self.finished = False
        self.error = None
        self.reported_error = False
        self._metrics = {"active": [], "inflight_bytes": 0, "inflight_samples": 0}
        self.accounted = {}
        self.schedule = {}
        self.last_version = controller.version
        self.learner_updates = controller.trainer.updates
        self.coalesced = 0
        self.receiver = MemoryJobs.results(controller, self)
        bootstrap = {
            "engine": str(controller.engine),
            "device": str(controller.device),
            "config": dict(controller.config),
            "counts": dict(controller.counts),
            "starts": controller.starts,
            "continuous": controller.continuous_selfplay,
            "frame_limit": min(32 * 2**20, max(4 * 2**20, available_memory() // (16 * count))),
            "weights": self.weights(controller),
            "profile": getattr(controller, "sampling_profile", None),
        }
        self.process = self.context.Process(
            target=sampling_process,
            args=(
                bootstrap,
                self.publications,
                self.output,
                self.status,
                self.stop,
                self.target,
                self.admission,
            ),
            name="haojie-sampling",
        )
        self.process.start()
        began = time.perf_counter()
        try:
            while not self._metrics.get("sampling_process"):
                self.refresh(controller)
                if controller.cancel.is_set():
                    raise InterruptedError("采样进程启动已取消")
                if not self.process.is_alive() or time.perf_counter() - began > 120:
                    raise RuntimeError("采样进程未完成启动")
                time.sleep(0.01)
        except BaseException:
            self.close(controller)
            raise
        self.startup_seconds = time.perf_counter() - began

    def weights(self, c):
        history = None
        if c.config["history_percent"]:
            names = c.store.names("history")
            offset = c.generation % max(1, len(names))
            for name in names[offset:] + names[:offset]:
                old = torch.load(c.store.root / name, weights_only=True, map_location="cpu")
                if old["version"] != c.version:
                    history = old
                    break
        return encode_weights({**c._weights(), "generation": c.generation, "history": history})

    @property
    def environments(self):
        return self._metrics.get("environments", 0)

    @property
    def slots(self):
        return {row["slot"]: {**row, "id": row["task"]} for row in self._metrics.get("active", [])}

    def set_target(self, value):
        self.target_environments = self.target.value = value

    def metrics(self):
        result = dict(self._metrics)
        result["coalesced_publications"] = result.get("coalesced_publications", 0) + self.coalesced
        if result.get("root_snapshots"):
            result["oldest_root_update_lag"] = result.get("oldest_root_update_lag", 0) + max(
                0, self.learner_updates - result.get("learner_updates_seen", self.learner_updates)
            )
        result["publisher_pending"] = self.pending is not None
        result["terminal_queue_size"] = self.output.qsize() if not self.closed else 0
        return result

    def publication_metrics(self):
        metrics = self.metrics()
        return {key: metrics[key] for key in PUBLICATION_FIELDS if key in metrics}

    def account(self, c):
        """按采样进程的同一时钟累计，包含学习与保存期间继续完成的工作。"""
        before, after = self.accounted, self.metrics()
        delta = {key: after.get(key, 0) - before.get(key, 0) for key in c.sampling_totals}
        with c.lock:
            for key, value in delta.items():
                c.sampling_totals[key] += value
        self.accounted = after
        return before, after, delta

    def publish(self, c):
        self.learner_updates = c.trainer.updates
        if self.finished or self.closed or self.stop.is_set():
            return
        if c.version != self.last_version:
            self.coalesced += self.pending is not None
            self.pending = self.weights(c)
            self.last_version = c.version
        if self.pending is not None:
            try:
                self.publications.put_nowait(self.pending)
            except queue.Full:
                pass
            else:
                self.pending = None

    def refresh(self, c):
        while True:
            try:
                value = self.status.get_nowait()
            except queue.Empty:
                break
            self._metrics, self.schedule = value["metrics"], value["schedule"]
        for key, value in self.schedule.items():
            if isinstance(value, dict):
                current = c.counts.setdefault(key, {})
                for name, count in value.items():
                    current[name] = max(current.get(name, 0), count)
            else:
                c.counts[key] = max(c.counts.get(key, 0), value)
        self.publish(c)

    def receive(self, c, timeout=0):
        try:
            packet = self.output.get(timeout=timeout)
        except queue.Empty:
            if not self.process.is_alive() and not self.finished:
                raise RuntimeError(f"采样进程异常退出：{self.process.exitcode}")
            return False
        kind, task, message = pickle.loads(packet)
        jobs = self.receiver
        if kind == "done":
            c.counts["tasks"] = max(c.counts["tasks"], task["id"])
            jobs.finish(task, message)
        elif kind == "unknown":
            jobs.unfinished(task)
        elif kind == "resource":
            jobs.resource_error(message)
        elif kind == "closed":
            self.finished, self.error = True, message
        return True

    def raise_if_error(self):
        if self.error and not self.reported_error:
            self.reported_error = True
            raise RuntimeError(self.error)

    def wait(self, c, jobs):
        self.receiver = jobs
        self.admission.set()
        began = time.perf_counter()
        before = jobs.finished
        while not c.cancel.is_set():
            self.refresh(c)
            received = self.receive(c, 0.1)
            if received:
                self.refresh(c)
                self.raise_if_error()
                if self.finished or (
                    c.continuous_selfplay
                    and (
                        (jobs.finished > before and c.samples.rows)
                        or jobs.finished - before >= c.config["games_per_round"]
                    )
                ):
                    break
            if time.perf_counter() - began >= 0.5:
                c._progress(self.metrics())
                began = time.perf_counter()
        if c.cancel.is_set():
            self.stop.set()
        self.refresh(c)
        c._progress(self.metrics())

    def close(self, c):
        if self.closed:
            return
        self.stop.set()
        began = time.perf_counter()
        try:
            while not self.finished and self.process.is_alive():
                self.receive(c, 0.1)
                self.refresh(c)
                if time.perf_counter() - began > 30:
                    self.process.terminate()
                    raise TimeoutError("采样进程未在关闭时限内结束")
            # 进程退出与 Queue feeder 可相邻发生；退出前的结案仍全部接纳。
            while not self.finished:
                if not self.receive(c, 0.1):
                    break
            self.process.join(timeout=5)
            self.refresh(c)
            if not self.process.is_alive():
                self._metrics.update(gpu_allocated=0, gpu_reserved=0, sampling_process_alive=False)
            self.account(c)
            self.raise_if_error()
        finally:
            self.closed = True
            for channel in (self.publications, self.output, self.status):
                channel.cancel_join_thread()
                channel.close()
