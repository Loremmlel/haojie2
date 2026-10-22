"""单一后台控制器拥有训练生命周期；HTTP仅提交意图，GPU阶段串行执行。"""

import copy
import hashlib
import json
import random
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import asdict

import torch

from ..batching.stream import update_groups
from ..data import collate_examples
from ..model import NETWORK_VERSION, ModelConfig, PolicyValueNet
from ..native.client import Client
from ..native.model import metadata, model_from
from ..native.pool import Pool
from ..runtime import Trainer, cpu_state, default_precision, resolve_device
from .config import METHOD as METHOD
from .config import defaults, validate
from .config.resources import available_memory, torch_threads
from .evaluation import evaluate_slice
from .memory import SamplePool
from .opponents import ready
from .sampling import MemoryJobs
from .storage import FORMAT, restore_recovery


def identity(model):
    digest = hashlib.sha256()
    for name, value in cpu_state(model.state_dict()).items():
        digest.update(name.encode())
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


class Controller:
    """HTTP 提交运行意图，后台线程串行拥有模型、采样器和检查点提交。

    暂停通过事件在阶段边界取消，样本池仅保留在内存；恢复和新实验成功后
    才切换持久化身份。设置校验和恢复文件校验分别由 config、storage 负责。
    """

    def __init__(
        self, engine, store, *, device="auto", tiny=False, imported=None, starts=None, threads=None
    ):
        self.engine, self.store = engine, store
        self.root_store = store
        manifest = store.root / "active.json"
        startup_notice = None
        self.experiment = "legacy"
        try:
            if manifest.exists():
                selected = json.loads(manifest.read_text(encoding="utf-8"))["experiment"]
                self.store = store if selected == "legacy" else store.scope(selected)
                self.experiment = selected
        except (ValueError, KeyError, TypeError) as error:
            startup_notice = f"活动清单无效：{error}；请在资源页选择已有实验，原文件保留"
        self.new_request = self.switch_request = None
        self.parent = None
        self.evaluation = None
        self.candidate = None
        self.update_budget = 0
        self.execution_precision = "fp32"
        self.readiness = {"training": False, "engine": "等待检查", "teacher": ready()}
        self.requested_threads = threads
        self.device = resolve_device(device)
        self.config = defaults(self.device)
        self.tiny, self.imported, self.starts = tiny, imported, starts
        self.lock = threading.RLock()
        self.snapshot_requests = deque()
        self.phase, self.phase_started = "idle", time.monotonic()
        self.phase_seconds = dict.fromkeys(
            ("idle", "sampling", "updating", "evaluating", "saving"), 0.0
        )
        self.last_round = {}
        self.sampling_totals = dict.fromkeys(
            (
                "requests",
                "inference_requests",
                "forwards",
                "inference_seconds",
                "collation_seconds",
                "tensor_bytes",
            ),
            0,
        )
        self.cancel, self.eval_cancel, self.shutdown = (threading.Event() for _ in range(3))
        self.wake = threading.Event()
        self.state, self.error = "idle", None
        self.want_run = self.want_save = self.want_eval = False
        self.want_restore = self.want_export = False
        self.trainer = self.pool = None
        self.sampling_jobs = None
        # 生产 MC-Q 使用跨轮对局；固定任务集基准可显式关闭以核对相同轨迹。
        self.continuous_selfplay = True
        self.version = None
        self.meta = None
        self.rng = random.Random(self.config["seed"])
        self._reset_progress()
        self.samples = SamplePool(self.config["pool_mib"] * 2**20)
        self.logs = deque(maxlen=100)
        if startup_notice:
            self.log(startup_notice)
        self.last_activity = None
        self.runtime = {"active": [], "inflight_bytes": 0}
        self.update_progress = 0
        self.telemetry = {}
        self.external = []
        self.last_metrics = time.monotonic()
        self.last_completed = time.time()
        self.thread = threading.Thread(target=self._loop, name="haojie-controller", daemon=True)
        self.thread.start()

    def _reset_progress(self):
        """首次启动和新实验使用相同计数起点；磁盘恢复随后覆盖这些默认值。"""
        self.counts = dict.fromkeys(
            (
                "games",
                "truncated",
                "discarded",
                "tasks",
                "eval_tasks",
                "seen_samples",
                "retained_samples",
                "oversize_samples",
            ),
            0,
        )
        self.losses, self.evaluations = [], []
        self.saved_step = self.saved_games = self.eval_games = self.eval_step = (
            self.history_step
        ) = 0
        self.saved_at = None
        self.generation = self.round_number = 0
        self.restored = False

    def log(self, message):
        with self.lock:
            self.logs.append({"time": time.time(), "message": str(message)[:2000]})

    def status(self):
        with self.lock:
            updates = self.trainer.updates if self.trainer else self.saved_step
            pool = self.pool
            actual_environments = len(pool.clients) if pool and not pool.closed else 0
            target_environments = self.config["environments"]
            if not (self.continuous_selfplay and self.config["method"] == METHOD):
                target_environments = min(target_environments, self.config["games_per_round"])
            return copy.deepcopy(
                {
                    "state": self.state,
                    "error": self.error,
                    "device": str(self.device),
                    "config": self.config,
                    "concurrency": {
                        "requested": self.config["environments"],
                        "actual": actual_environments,
                        "pending": bool(
                            actual_environments and actual_environments != target_environments
                        ),
                    },
                    "counts": self.counts,
                    "version": self.version,
                    "experiment": self.experiment,
                    "experiments": self.root_store.experiments(),
                    "parent": self.parent,
                    "generation": self.generation,
                    "readiness": self.readiness,
                    "execution": {
                        "sampling": self.config["sampling_precision"],
                        "updating": self.execution_precision,
                        "optimizer": "auto",
                    },
                    "update_budget": self.update_budget,
                    "evaluation_job": self.evaluation.summary() if self.evaluation else None,
                    "candidate": {k: v for k, v in self.candidate.items() if k != "weights"}
                    if self.candidate
                    else None,
                    "updates": updates,
                    "saved_step": self.saved_step,
                    "saved_at": self.saved_at,
                    "unsaved_updates": updates - self.saved_step,
                    "unsaved_games": self.counts["games"] - self.saved_games,
                    "restored": self.restored,
                    "recovery_note": (
                        "退出后仅恢复最近完整磁盘状态；未保存更新与RAM样本允许丢失，"
                        "精确损失数量无法跨崩溃重建。"
                    ),
                    "research_fixture": bool(self.starts),
                    "pool": self.samples.snapshot(),
                    "runtime": self.runtime,
                    "update_progress": self.update_progress,
                    "save_pending": self.want_save,
                    "eval_pending": self.want_eval,
                    "next_eval_games": max(
                        0, self.config["eval_games"] - (self.counts["games"] - self.eval_games)
                    ),
                    "next_save_seconds": max(
                        0,
                        self.config["save_seconds"]
                        - (time.time() - (self.saved_at or self.last_completed)),
                    ),
                    "evaluation_ready": ready(),
                    "disk": self.store.snapshot(),
                    "recoveries": self.store.names("recovery"),
                    "history": self.store.names("history"),
                    "best": self.store.names("best"),
                    "exports": self.store.names("export"),
                    "losses": self.losses,
                    "evaluations": self.evaluations,
                    "logs": list(self.logs),
                    "resources": self.telemetry,
                    "performance": self.performance(),
                    "external": self.external,
                    "learning_status": (
                        "可开始收益驱动训练"
                        if self.readiness["training"]
                        else "训练引擎尚未就绪，详见资源页"
                    )
                    + "；自然棋力："
                    + self._strength_note(),
                }
            )

    def performance(self):
        with self.lock:
            seconds = self.phase_seconds.copy()
            seconds[self.phase] += time.monotonic() - self.phase_started
            return {
                "phase": self.phase,
                "phase_seconds": seconds,
                "sampling": self.sampling_totals.copy(),
                "last_round": self.last_round.copy(),
            }

    @contextmanager
    def _phase(self, name):
        """阶段墙钟独占记账；采样期间的保存归入 saving，避免重复相加。"""
        with self.lock:
            now, previous = time.monotonic(), self.phase
            self.phase_seconds[previous] += now - self.phase_started
            self.phase, self.phase_started = name, now
        try:
            yield
        finally:
            with self.lock:
                now = time.monotonic()
                self.phase_seconds[name] += now - self.phase_started
                self.phase, self.phase_started = previous, now

    def snapshot(self, callback):
        """在训练线程安全点保存，再冻结本地文件；压缩和远端写入由调用者执行。

        Future 返回回调结果；HTTP 不接受或执行回调，备份错误不冒充训练失败。
        """
        future = Future()
        with self.lock:
            if self.shutdown.is_set():
                future.set_exception(RuntimeError("训练服务已关闭"))
            else:
                self.snapshot_requests.append((callback, future))
                self.wake.set()
        return future

    def _snapshots(self):
        while True:
            with self.lock:
                if not self.snapshot_requests:
                    return
                callback, future = self.snapshot_requests.popleft()
            if not future.set_running_or_notify_cancel():
                continue
            try:
                with self._phase("saving"):
                    self._save()
                    result = callback(self)
                future.set_result(result)
            except Exception as error:
                self.log(f"备份快照失败：{error}")
                future.set_exception(error)

    def _reject_snapshots(self, error):
        with self.lock:
            while self.snapshot_requests:
                _, future = self.snapshot_requests.popleft()
                if future.set_running_or_notify_cancel():
                    future.set_exception(error)

    def _strength_note(self):
        if not self.evaluations:
            return "尚未评测"
        last = self.evaluations[-1]
        tiers = ("easy", "medium", "hard")
        if any(last["results"][key]["n"] < 4 for key in tiers):
            return "样本不足，详见评测"
        earlier = [
            row
            for row in self.evaluations[:-1]
            if row["series"] == last["series"]
            and row["model"] != last["model"]
            and all(row["results"][key]["n"] == last["results"][key]["n"] for key in tiers)
        ]
        if not earlier:
            return "已有对战结果，尚无同条件前后对照"
        delta = sum(
            last["results"][key]["score_rate"] - earlier[0]["results"][key]["score_rate"]
            for key in tiers
        )
        return (
            "本次观测"
            + ("改善" if delta > 0 else "下降" if delta < 0 else "无变化")
            + "（小样本，不代表显著提升）"
        )

    def command(self, action, values=None):
        with self.lock:
            if action == "new-experiment":
                if self.state not in ("idle", "paused", "error"):
                    raise ValueError("请先暂停，再新建实验")
                if self.new_request:
                    return
                values = values or {}
                source = values.get("source", "fresh")
                if source not in ("fresh", "current", "recovery"):
                    raise ValueError("继承来源无效")
                device = values.get("device", str(self.device))
                selected_device = resolve_device(device)
                config = {**defaults(selected_device), **values.get("config", {})}
                validate(config)
                self.new_request = (source, selected_device, config)
                self.state, self.error = "starting", None
            elif action == "start":
                if self.state in ("idle", "paused", "error"):
                    self.want_run, self.error = True, None
                    self.cancel.clear()
                    self.state = "starting"
            elif action in ("switch-experiment", "delete-experiment"):
                if self.state not in ("idle", "paused", "error"):
                    raise ValueError("请先暂停，再管理实验")
                target = (values or {}).get("experiment")
                entries = {entry["id"]: entry for entry in self.root_store.experiments()}
                if target not in entries or target == self.experiment:
                    raise ValueError("请选择一个非当前实验")
                if action == "switch-experiment":
                    if not entries[target]["recoveries"]:
                        raise ValueError("此实验没有完整恢复点，不能继续")
                    self.switch_request = target
                    self.state = "starting"
                else:
                    if target == "legacy":
                        raise ValueError("旧根目录受保护，不能通过实验删除入口移除")
                    prefix = f"experiments/{target}/"
                    for name in list(self.root_store.files):
                        if name.startswith(prefix):
                            self.root_store.remove(name)
                    self.log(f"已删除明确选中的非当前实验 {target}；其他实验和未纳管文件保留")
            elif action == "pause":
                self.want_run = False
                self.want_eval = False
                self.cancel.set()
                self.eval_cancel.set()
                if self.state not in ("idle", "paused"):
                    self.state = "pausing"
            elif action == "save":
                self.want_save = True
            elif action == "evaluate":
                if not ready():
                    raise ValueError("评测未就绪，请安装随包 mini-racer 依赖")
                self.want_eval = not bool(self.evaluation)
                if not self.want_run:
                    self.cancel.clear()
            elif action == "cancel-evaluation":
                self.want_eval = False
                self.eval_cancel.set()
                if not self.evaluation:
                    self.eval_games = self.counts["games"]
                    self.eval_step = self.trainer.updates if self.trainer else 0
                if self.candidate:
                    self.candidate["confirm"] = False
            elif action == "confirm-candidate":
                if not self.candidate:
                    raise ValueError("尚无冻结趋势候选；请先完成一轮评测")
                if self.evaluation:
                    raise ValueError("已有评测任务，请等待或取消后确认")
                if time.time() - self.candidate["created"] > 86400:
                    self.candidate = None
                    raise ValueError("候选已超过24小时；请重新立即评测")
                self.want_eval = True
                self.candidate["confirm"] = True
                if not self.want_run:
                    self.cancel.clear()
            elif action == "export":
                if not self.trainer:
                    raise ValueError("请先新建或继续训练以加载模型")
                self.want_export = True
            elif action == "restore":
                if self.state not in ("idle", "paused", "error"):
                    raise ValueError("请先暂停，再恢复磁盘状态")
                if not self.store.names("recovery"):
                    raise ValueError("尚无恢复点；请新建训练")
                self.want_restore = True
                self.state, self.error = "starting", None
            elif action == "cleanup":
                if self.state not in ("idle", "paused", "error"):
                    raise ValueError("请先暂停，再回收权重")
                for category in ("history", "export"):
                    for name in self.store.names(category)[:-1]:
                        self.root_store.remove(name)
                self.log("已回收可替换的旧历史/导出权重；恢复点和评测基准保留")
            elif action == "concurrency":
                if self.state in ("starting", "pausing"):
                    raise ValueError("请等待启动或暂停完成，再调整并发")
                if not isinstance(values, dict) or set(values) != {"environments"}:
                    raise ValueError("并发设置只接受 environments 字段")
                config = {**self.config, **values}
                validate(config)
                self.config = config
                self.log(f"并发目标已设为 {config['environments']}，下一采样窗口生效")
            elif action == "settings":
                if self.state not in ("idle", "paused", "error"):
                    raise ValueError("请先暂停再修改设置")
                if not isinstance(values, dict):
                    raise ValueError("设置必须是字段对象")
                config = {**self.config, **(values or {})}
                validate(config)
                if self.trainer and config["method"] != self.config["method"]:
                    raise ValueError("已初始化会话不能切换学习目标；请继续当前方法")
                self.config = config
                self.samples.resize(config["pool_mib"] * 2**20)
            else:
                raise ValueError("未知操作")
            self.wake.set()

    def _initialize(self):
        # CLI 的自动线程策略随新实验设备更新；嵌入调用保留其自行选择的进程线程数。
        if self.requested_threads is not None:
            threads = torch_threads(str(self.device), self.requested_threads)
            if torch.get_num_threads() != threads:
                torch.set_num_threads(threads)
        if self.trainer:
            self._device(self.device)
            return
        with Client(self.engine) as client:
            self.ready = client.ready
        if "mc-context-v2" not in self.ready.get("capabilities", []):
            raise ValueError("原生引擎缺少内存出口；请更新本次正式构建")
        self.meta = {
            **metadata(self.ready),
            "stream_format": FORMAT,
            "policy_source": self.config["method"],
        }
        recoveries = self.store.names("recovery")
        payload = None
        if recoveries and not self.imported:
            trainer, payload = restore_recovery(self.store, self.meta, self.device, self.log)
            self.config = payload["config"]
            self.meta["policy_source"] = self.config["method"]
        elif self.imported:
            model, _ = model_from(self.imported, self.ready)
        else:
            torch.manual_seed(self.config["seed"])
            model = PolicyValueNet(ModelConfig.tiny() if self.tiny else ModelConfig())
        if payload is None:
            precision = default_precision(self.device)
            trainer = Trainer(
                model,
                self.device,
                precision=precision,
                optimizer="auto",
                objective=self.config["method"],
            )
        if payload:
            accelerator = payload.get("accelerator_rng")
            if accelerator and accelerator["device"] == self.device.type:
                getattr(torch, self.device.type).set_rng_state(accelerator["state"], self.device)
            self.counts = payload["counts"]
            self.saved_step = trainer.updates
            self.saved_games = self.counts["games"]
            self.saved_at = payload["saved_at"]
            self.rng.setstate(payload["random"])
            self.round_number = payload.get("round_number", 0)
            self.generation = payload.get("generation", 0)
            self.parent = payload.get("parent")
            self.eval_games, self.eval_step = payload.get("eval_cursor", [0, 0])
            self.history_step = payload.get("history_step", 0)
            self.losses, self.evaluations = (
                payload.get("losses", []),
                payload.get("evaluations", []),
            )
            self.restored = True
            self.log(f"恢复至更新 {self.saved_step}；样本池为空，恢复点之后未保存的进度不在磁盘中")
        self.trainer = trainer
        self.execution_precision = trainer.precision
        self.samples = SamplePool(self.config["pool_mib"] * 2**20)
        self.samples.generation = self.generation
        if payload:
            self.samples.restore_totals(payload.get("sample_totals", {}))
        self.version = identity(self.trainer.model)
        if payload:
            try:
                recent = self.store.read_metrics()
                # 同一已恢复权重的评测可独立落小摘要，不因没有新更新而必须重存模型。
                if (
                    recent.get("version") == self.version
                    and recent.get("updates") == trainer.updates
                ):
                    self.losses = recent["losses"]
                    self.evaluations = recent["evaluations"]
            except (ValueError, KeyError, OSError) as error:
                self.log(f"指标摘要未恢复：{error}；学习状态仍使用有效恢复点")
        if not self.store.names("history"):
            self._history()
        if not self.store.names("baseline"):
            self.store.save_tensor("baseline", 0, self._weights(), 1)
        self.log(f"学习器就绪：{self.config['method']}；行为版本 {self.version[:12]}")

    def _device(self, device):
        self.trainer.model.to(device)
        self.trainer.device = torch.device(device)
        for state in self.trainer.optimizer.state.values():
            for key, value in state.items():
                # AdamW 的非 capturable 步数保持 CPU。
                if isinstance(value, torch.Tensor) and key != "step":
                    state[key] = value.to(device)

    def _weights(self):
        return {
            "format": "haojie-stream-weights-v1",
            "network": NETWORK_VERSION,
            "version": self.version,
            "config": asdict(self.trainer.model.config),
            "metadata": self.meta,
            "updates": self.trainer.updates,
            "model": cpu_state(self.trainer.model.state_dict()),
        }

    def _history(self):
        for name in self.store.names("history"):
            old = torch.load(self.store.root / name, weights_only=True, map_location="cpu")
            if old["version"] == self.version:
                self.history_step = self.trainer.updates
                return
        self.store.save_tensor(
            "history", self.trainer.updates, self._weights(), self.config["history_size"]
        )
        self.history_step = self.trainer.updates

    def _save(self):
        with self._phase("saving"):
            self._save_state()

    def _save_state(self):
        self.want_save = False
        health = self.trainer.health()
        if not health["finite_parameters"] or not health["finite_optimizer"]:
            raise FloatingPointError("拒绝保存非有限学习状态；请重启服务恢复最后有效点")
        payload = {
            "format": FORMAT,
            "trainer": self.trainer.checkpoint(self.meta),
            "config": self.config,
            "counts": self.counts,
            "saved_at": time.time(),
            "random": self.rng.getstate(),
            "accelerator_rng": None
            if self.device.type == "cpu"
            else {
                "device": self.device.type,
                "state": getattr(torch, self.device.type).get_rng_state(self.device),
            },
            "round_number": self.round_number,
            "generation": self.generation,
            "parent": self.parent,
            "eval_cursor": [self.eval_games, self.eval_step],
            "history_step": self.history_step,
            "sample_totals": self.samples.snapshot(),
            "losses": self.losses,
            "evaluations": self.evaluations,
        }
        self.store.save_tensor("recovery", self.trainer.updates, payload, 2)
        self.saved_step, self.saved_games = self.trainer.updates, self.counts["games"]
        self.saved_at = payload["saved_at"]
        self.log(f"恢复点保存完成：更新 {self.saved_step}")

    def _due_save(self):
        if self.want_save:
            return True
        return (
            self.want_run
            and self.trainer.updates > self.saved_step
            and (
                time.time() - (self.saved_at or self.last_completed) >= self.config["save_seconds"]
                or self.config["save_games"]
                and self.counts["games"] - self.saved_games >= self.config["save_games"]
            )
        )

    def _progress(self, metrics):
        self.runtime = {
            **metrics,
            "inflight_bytes": sum(t["trajectory"].used for t in self.pool.slots.values()),
            "inflight_samples": sum(t["trajectory"].seen for t in self.pool.slots.values()),
            "historical_models": max(0, len(self.sampling_jobs.inferences) - 1)
            if self.sampling_jobs
            else 0,
        }
        if available_memory() < 256 * 2**20:
            raise MemoryError("可用RAM不足256MiB；已暂停并丢弃在途局")
        self._snapshots()
        if self._due_save() and not self.cancel.is_set():
            self._save()
        if time.monotonic() - self.last_metrics >= 60:
            self.store.reconcile()
            self._metrics()
            self.last_metrics = time.monotonic()

    def _metrics(self):
        self.store.metrics(
            {
                "version": self.version,
                "updates": self.trainer.updates,
                "losses": self.losses,
                "evaluations": self.evaluations,
                "logs": list(self.logs),
                "performance": self.performance(),
            }
        )

    def _new_pool(self, count):
        from ..native.execution import PolicyInference

        return Pool(
            self.engine,
            count,
            self.device,
            batch_wait_ms=self.config["batch_wait_ms"],
            max_batch_size=self.config["inference_batch_size"],
            inference=(
                PolicyInference(self.trainer.model, self.device, self.config["sampling_precision"]),
                self.version,
            ),
            frame_limit=min(32 * 2**20, max(4 * 2**20, available_memory() // (16 * count))),
        )

    def _historical(self, jobs=None):
        # 历史模型按整局冻结，达到驻留上限时继续选择已有版本，不阻塞当前策略更新。
        if jobs:
            jobs.release_unused()
        if not self.config["history_percent"]:
            return None
        if jobs:
            versions = [version for version in jobs.inferences if version != jobs.current]
            if len(versions) >= self.config["history_size"]:
                version = versions[self.generation % len(versions)]
                return jobs.inferences[version].model, version
        names = self.store.names("history")
        # 先去除同权重，再按训练轮次选择；评测/优化抽样不改变历史轮转。
        for name in (
            names[self.generation % max(1, len(names)) :]
            + names[: self.generation % max(1, len(names))]
        ):
            weight = torch.load(self.store.root / name, weights_only=True, map_location="cpu")
            if weight["version"] != self.version:
                if jobs and weight["version"] in jobs.inferences:
                    return jobs.inferences[weight["version"]].model, weight["version"]
                old = PolicyValueNet(self.trainer.model.config)
                old.load_state_dict(weight["model"])
                return old.to(self.device).eval(), weight["version"]
            del weight
        return None

    def _sample(self):
        self.state = "sampling"
        continuous = self.continuous_selfplay and self.config["method"] == METHOD
        # 仅训练线程在窗口边界重建；已完成样本和学习状态保留，未终局按原取消协议记账。
        count = self.config["environments"]
        count = count if continuous else min(count, self.config["games_per_round"])
        if self.pool and len(self.pool.clients) != count:
            discarded = len(self.pool.slots)
            self._close_pool()
            self.log(f"并发切换为 {count}；取消 {discarded} 个在途局，已完成样本保留")
        if self.pool is None:
            self.pool = self._new_pool(count)
        self.pool.model_hash = self.version
        jobs = self.sampling_jobs if continuous else None
        historical = self._historical(jobs)
        if jobs is None:
            jobs = MemoryJobs(self, self.version, historical, continuous=continuous)
            self.sampling_jobs = jobs
        else:
            jobs.next_window(self.version, historical)
        before = self.pool.metrics()
        began = self.performance()["phase_seconds"]["sampling"]
        games, samples = self.counts["games"], self.counts["retained_samples"]
        finished = jobs.finished
        try:
            self.pool.run(
                jobs,
                cancel_event=self.cancel,
                keep_open=True,
                snapshot=self._progress,
                # 已有真实终局样本即可学习；其他局仍在下一决策根停驻并跨更新保留。
                # 完成数上限只负责让无样本的截断窗口也能归还控制，不再作为学习门槛。
                yield_when=(
                    lambda: (
                        (jobs.finished > finished and bool(self.samples.rows))
                        or jobs.finished >= jobs.target
                    )
                )
                if continuous
                else None,
            )
        finally:
            after = self.pool.metrics()
            elapsed = self.performance()["phase_seconds"]["sampling"] - began
            delta = {key: after[key] - before[key] for key in self.sampling_totals}
            with self.lock:
                for key, value in delta.items():
                    self.sampling_totals[key] += value
                self.last_round = {
                    **delta,
                    "seconds": elapsed,
                    "games": self.counts["games"] - games,
                    "retained_samples": self.counts["retained_samples"] - samples,
                    "mean_batch": delta["inference_requests"] / max(1, delta["forwards"]),
                    "requests_per_second": delta["requests"] / max(0.001, elapsed),
                    "environments": len(self.pool.clients),
                    "completed_tasks": jobs.finished - finished,
                    "carried_games": len(self.pool.slots),
                    "mean_active_environments": (
                        after["active_slot_seconds"] - before["active_slot_seconds"]
                    )
                    / max(0.001, elapsed),
                }
            self.runtime = {
                **after,
                "inflight_bytes": sum(t["trajectory"].used for t in self.pool.slots.values()),
                "inflight_samples": sum(t["trajectory"].seen for t in self.pool.slots.values()),
                "historical_models": max(0, len(jobs.inferences) - 1) if continuous else 0,
            }
            if not continuous:
                self.sampling_jobs = None

    def _update(self):
        self.state = "updating"
        self.trainer.model.train()
        self.update_progress = 0
        self.update_budget = self.config["updates_per_round"] or self.samples.budget(
            self.config["batch_size"], self.config["update_limit"]
        )
        for step in range(self.update_budget):
            if self.cancel.is_set() or not self.samples.rows:
                break
            examples = self.samples.batch(self.config["batch_size"], self.rng)
            groups = (
                update_groups(examples, self.trainer.model.config)
                if self.config["method"].startswith("decomposed-mc-q-")
                else [examples]
            )
            batches = [
                {
                    k: v.to(self.device, non_blocking=self.device.type == "cuda")
                    for k, v in collate_examples(group, self.trainer.model.config).items()
                }
                for group in groups
            ]
            self.trainer.step_batches(batches)
            self.update_progress = step + 1
            if self.snapshot_requests:
                self._snapshots()
        self.version = identity(self.trainer.model)
        if self.update_progress:
            self.generation += 1
            self.samples.expire(self.generation)
            self.losses.append(
                {"step": self.trainer.updates, "time": time.time(), **self.trainer.last_losses}
            )
            self.losses = self._compact(self.losses)
            health = self.trainer.health()
            if not health["finite_parameters"] or not health["finite_optimizer"]:
                raise FloatingPointError("模型/优化器包含非有限参数")
            if (
                not self.cancel.is_set()
                and self.trainer.updates - self.history_step >= self.config["history_every"]
            ):
                self._history()
        self.update_progress = 0

    @staticmethod
    def _compact(rows):
        return rows if len(rows) <= 256 else rows[:128:2] + rows[128:]

    def _evaluate(self):
        # 一组四局覆盖三档教师和基准，随后归还学习器，避免确认被长采样轮无限稀释。
        for _ in range(4):
            evaluate_slice(self)
            self._snapshots()
            if not self.evaluation or self.cancel.is_set() or self.eval_cancel.is_set():
                break

    def _due_evaluation(self):
        return (
            self.want_run
            and not self.cancel.is_set()
            and ready()
            and self.trainer.updates > self.eval_step
            and self.counts["games"] - self.eval_games >= self.config["eval_games"]
        )

    def _close_pool(self):
        if self.pool:
            try:
                if self.sampling_jobs:
                    for task in self.pool.slots.values():
                        self.sampling_jobs.unfinished(task)
                    for inference in self.sampling_jobs.inferences.values():
                        inference.close()
            finally:
                try:
                    self.pool.close()
                finally:
                    self.pool = self.sampling_jobs = None

    def _pause(self):
        failed = self.state == "error"
        self._close_pool()
        if self.evaluation and (self.cancel.is_set() or self.eval_cancel.is_set()):
            evaluate_slice(self)
        if self.trainer:
            self.trainer.optimizer.zero_grad(set_to_none=True)
            self._device("cpu")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        self.runtime = {"active": [], "inflight_bytes": 0}
        self.state = "error" if failed else "paused"

    def _loop(self):
        try:
            with Client(self.engine) as client:
                valid = "mc-context-v2" in client.ready.get("capabilities", [])
            self.readiness = {
                "training": valid,
                "engine": "就绪" if valid else "请运行 npm run native:build 更新引擎",
                "teacher": ready(),
            }
        except Exception as error:
            self.readiness = {"training": False, "engine": str(error), "teacher": ready()}
        while not self.shutdown.is_set():
            try:
                if self.candidate and time.time() - self.candidate["created"] > 86400:
                    if not self.evaluation:
                        self.want_eval = False
                    self.candidate = None
                    self.log("趋势候选超过24小时，已释放冻结权重；可重新评测")
                if self.switch_request:
                    self._switch_experiment()
                if self.new_request:
                    self._new_experiment()
                if self.want_restore:
                    self.want_restore = False
                    self._close_pool()
                    self.trainer = self.version = self.imported = None
                    self.cancel.clear()
                    self._initialize()
                    self._pause()
                if not (
                    self.want_run
                    or self.want_save
                    or self.want_eval
                    or self.evaluation
                    or self.want_export
                    or self.snapshot_requests
                ):
                    if self.state == "pausing":
                        self._pause()
                    self.wake.wait(0.25)
                    self.wake.clear()
                    continue
                self._initialize()
                self._snapshots()
                if self.want_export:
                    self.want_export = False
                    name = self.store.save_tensor(
                        "export", self.trainer.updates, self._weights(), 1
                    )
                    self.log(f"导出完成（受管，仅保留最新一份）：{self.store.root / name}")
                if self.want_save:
                    self._save()
                # 更新完成后即使先暂停，继续时也先兑现到期评测，不再额外等待长采样轮。
                if self._due_evaluation():
                    self.want_eval = True
                if (self.want_eval or self.evaluation) and not self.cancel.is_set():
                    with self._phase("evaluating"):
                        self._evaluate()
                if self.want_run and not self.cancel.is_set():
                    # 暂停后剩余或上次受更新上限保留的终局样本，先消费再继续采样。
                    if not (
                        self.continuous_selfplay
                        and self.config["method"] == METHOD
                        and self.samples.rows
                    ):
                        with self._phase("sampling"):
                            self._sample()
                    if not self.cancel.is_set():
                        with self._phase("updating"):
                            self._update()
                        if self._due_save():
                            self._save()
                if (not self.want_run and not self.evaluation) or self.cancel.is_set():
                    self._pause()
            except Exception as error:
                self._reject_snapshots(error)
                self.want_run = self.want_eval = self.want_save = self.want_export = False
                self.cancel.set()
                self.eval_cancel.set()
                self.error, self.state = str(error), "error"
                self.log(error)
                self._pause()
        self._pause()

    def _new_experiment(self):
        source, device, config = self.new_request
        self.new_request = None
        weights, parent = None, None
        if source == "current":
            if not self.trainer:
                raise ValueError("当前没有内存模型，请选择恢复点继承")
            weights = copy.deepcopy(self.trainer.model).cpu()
            parent = {
                "experiment": self.experiment,
                "version": self.version,
                "updates": self.trainer.updates,
                "optimizer": False,
            }
        elif source == "recovery":
            names = self.store.names("recovery")
            if not names:
                raise ValueError("没有可继承的恢复点")
            with Client(self.engine) as client:
                expected = metadata(client.ready)
            for name in reversed(names):
                try:
                    payload = torch.load(
                        self.store.root / name, weights_only=True, map_location="cpu"
                    )
                    model = payload["trainer"]
                    if model["network"] != NETWORK_VERSION or any(
                        model["metadata"].get(key) != expected[key]
                        for key in ("ruleset", "encoding", "schema", "rules_package_sha256")
                    ):
                        raise ValueError("权重网络/规则/编码不兼容")
                    weights = PolicyValueNet(ModelConfig(**model["config"]))
                    weights.load_state_dict(model["model"])
                    if not all(torch.isfinite(p).all() for p in weights.parameters()):
                        raise ValueError("权重包含非有限参数")
                    break
                except Exception as error:
                    self.log(f"无法继承 {name}：{error}")
                    weights = None
            if weights is None:
                raise ValueError("没有兼容且有效的权重；请新建随机初始化实验")
            parent = {
                "experiment": self.experiment,
                "recovery": name,
                "method": payload["config"]["method"],
                "optimizer": False,
            }
        if self.trainer and self.trainer.updates > self.saved_step:
            self._save()
        self._close_pool()
        previous = self.__dict__.copy()
        try:
            self._activate_experiment(weights, parent, device, config)
        except Exception:
            # 只有有效初始恢复点和清单均发布后才切换；失败产物仍由全局配额计账。
            self.__dict__.update(previous)
            raise

    def _switch_experiment(self):
        target, self.switch_request = self.switch_request, None
        if self.trainer and self.trainer.updates > self.saved_step:
            self._save()
        self._close_pool()
        previous = self.__dict__.copy()
        try:
            self.experiment = target
            self.store = self.root_store if target == "legacy" else self.root_store.scope(target)
            self.trainer = self.version = self.imported = self.evaluation = self.candidate = None
            self.cancel.clear()
            self._initialize()
            self.root_store.write("active.json", json.dumps({"experiment": target}).encode())
            self.error = None
            self._pause()
            self.log(f"已打开实验 {target}，从有效恢复点继续同一优化器")
        except Exception:
            self.__dict__.update(previous)
            raise

    def _activate_experiment(self, weights, parent, device, config):
        self.experiment = uuid.uuid4().hex[:12]
        self.store = self.root_store.scope(self.experiment)
        self.trainer = self.meta = self.version = None
        self.imported = None
        self.device, self.config, self.parent = device, config, parent
        self._reset_progress()
        self.evaluation = self.candidate = None
        self.rng = random.Random(config["seed"])
        self.cancel.clear()
        self._initialize()
        if weights:
            self.trainer.model.load_state_dict(weights.state_dict())
            self.version = identity(self.trainer.model)
            # 继承后替换本实验初始化权重；旧实验原件保留。
            self._history()
            self.store.save_tensor("baseline", 0, self._weights(), 1)
        self._save()
        self.root_store.write("active.json", json.dumps({"experiment": self.experiment}).encode())
        self.state = "paused"
        self.log(
            f"新实验 {self.experiment} 已就绪；"
            + ("仅继承权重，优化器和曲线重新建立" if parent else "默认模型随机初始化")
        )

    def close(self):
        self.shutdown.set()
        self.cancel.set()
        self.eval_cancel.set()
        self.wake.set()
        self.thread.join(timeout=20)
        self._reject_snapshots(RuntimeError("训练服务已关闭，快照未执行"))
        if self.thread.is_alive():
            raise TimeoutError("训练线程仍在结束当前操作，请等待进程退出后再打开同一目录")
