"""单一后台控制器拥有训练生命周期；HTTP仅提交意图，GPU阶段串行执行。"""

import copy
import hashlib
import json
import random
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict

import psutil
import torch

from ..data import collate_examples
from ..model import NETWORK_VERSION, ModelConfig, PolicyValueNet
from ..native.client import Client
from ..native.pipeline import metadata, model_from
from ..native.pool import Pool
from ..runtime import Trainer, resolve_device
from .evaluation import evaluate_slice
from .memory import SamplePool
from .opponents import ready
from .sampling import MemoryJobs

METHOD = "decomposed-mc-q-v2"
FORMAT = "haojie-stream-recovery-v2"


def defaults():
    return {
        "environments": 4,
        "games_per_round": 4,
        "updates_per_round": 0,
        "update_limit": 256,
        "batch_size": 8,
        "classic_percent": 50,
        "history_percent": 50,
        "current_first_percent": 50,
        "pool_mib": min(4096, max(64, int(psutil.virtual_memory().available / 4 / 2**20))),
        "save_seconds": 600,
        "save_games": 0,
        "eval_games": 16,
        "eval_pairs": 1,
        "eval_seconds": 180,
        "confirmation_pairs": 5,
        "max_commands": 20000,
        "max_plies": 1000,
        "history_every": 200,
        "history_size": 4,
        "seed": 20261004,
        "method": METHOD,
        "temperature": 0.25,
    }


def validate(config):
    limits = {
        "environments": (1, 16),
        "games_per_round": (1, 128),
        "updates_per_round": (0, 1000),
        "update_limit": (1, 1000),
        "classic_percent": (0, 100),
        "history_percent": (0, 100),
        "current_first_percent": (0, 100),
        "confirmation_pairs": (5, 20),
        "batch_size": (1, 32),
        "pool_mib": (16, 4096),
        "save_seconds": (10, 86400),
        "save_games": (0, 100000),
        "eval_games": (1, 100000),
        "eval_pairs": (1, 20),
        "eval_seconds": (2, 7200),
        "max_commands": (1, 20000),
        "max_plies": (1, 1000),
        "history_every": (1, 100000),
        "history_size": (4, 8),
        "seed": (1, 999999999),
    }
    if set(config) != set(defaults()):
        raise ValueError("设置字段不匹配")
    for key, (low, high) in limits.items():
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError(f"{key} 必须为 {low} 至 {high} 的整数")
    if config["method"] not in (METHOD, "sampled-action-imitation"):
        raise ValueError("未知训练方法")
    if (
        not isinstance(config["temperature"], (int, float))
        or not 0.05 <= config["temperature"] <= 2
    ):
        raise ValueError("温度必须为0.05至2")


def identity(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class Controller:
    def __init__(self, engine, store, *, device="auto", tiny=False, imported=None, starts=None):
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
        self.new_request = None
        self.parent = None
        self.generation = 0
        self.evaluation = None
        self.candidate = None
        self.update_budget = 0
        self.execution_precision = "fp32"
        self.readiness = {"training": False, "engine": "等待检查", "teacher": ready()}
        self.device = resolve_device(device)
        self.config = defaults()
        self.tiny, self.imported, self.starts = tiny, imported, starts
        self.lock = threading.RLock()
        self.cancel, self.eval_cancel, self.shutdown = (threading.Event() for _ in range(3))
        self.wake = threading.Event()
        self.state, self.error = "idle", None
        self.want_run = self.want_save = self.want_eval = False
        self.trainer = self.pool = None
        self.version = None
        self.meta = None
        self.rng = random.Random(self.config["seed"])
        self.counts = dict.fromkeys(
            (
                "games",
                "truncated",
                "discarded",
                "tasks",
                "seen_samples",
                "retained_samples",
                "oversize_samples",
            ),
            0,
        )
        self.samples = SamplePool(self.config["pool_mib"] * 2**20)
        self.logs, self.losses, self.evaluations = deque(maxlen=100), [], []
        if startup_notice:
            self.log(startup_notice)
        self.saved_step = self.saved_games = self.eval_games = self.eval_step = (
            self.history_step
        ) = 0
        self.saved_at = self.last_activity = None
        self.round_number = 0
        self.runtime = {"active": [], "inflight_bytes": 0}
        self.update_progress = 0
        self.telemetry = {}
        self.external = []
        self.last_metrics = time.monotonic()
        self.restored = False
        self.last_completed = time.time()
        self.thread = threading.Thread(target=self._loop, name="haojie-controller", daemon=True)
        self.thread.start()

    def log(self, message):
        with self.lock:
            self.logs.append({"time": time.time(), "message": str(message)[:2000]})

    def status(self):
        with self.lock:
            updates = self.trainer.updates if self.trainer else self.saved_step
            return copy.deepcopy(
                {
                    "state": self.state,
                    "error": self.error,
                    "device": str(self.device),
                    "config": self.config,
                    "counts": self.counts,
                    "version": self.version,
                    "experiment": self.experiment,
                    "experiments": self.root_store.experiments(),
                    "parent": self.parent,
                    "generation": self.generation,
                    "readiness": self.readiness,
                    "execution": {
                        "sampling": "fp32",
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
                config = {**defaults(), **values.get("config", {})}
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
            for name in reversed(recoveries):
                try:
                    payload = torch.load(
                        self.store.root / name, weights_only=True, map_location="cpu"
                    )
                    if payload["format"] != FORMAT or payload["trainer"]["metadata"] != {
                        **self.meta,
                        "policy_source": payload["config"]["method"],
                    }:
                        raise ValueError("恢复点规则/编码不匹配")
                    validate(payload["config"])
                    restored_trainer = Trainer(
                        PolicyValueNet(ModelConfig(**payload["trainer"]["config"])),
                        self.device,
                        precision=payload["trainer"]["precision"],
                        optimizer="auto",
                        objective=payload["config"]["method"],
                    )
                    restored_trainer.restore_payload(
                        payload["trainer"], payload["trainer"]["metadata"]
                    )
                    if not all(
                        restored_trainer.health()[key]
                        for key in ("finite_parameters", "finite_optimizer")
                    ):
                        raise ValueError("恢复点含非有限学习状态")
                    break
                except Exception as error:
                    self.log(f"恢复点 {name} 无效：{error}")
                    payload = None
            if payload is None:
                raise ValueError("无有效恢复点；未覆盖已有文件")
            self.config = payload["config"]
            self.meta["policy_source"] = self.config["method"]
            model = restored_trainer.model
        elif self.imported:
            model, _ = model_from(self.imported, self.ready)
        else:
            torch.manual_seed(self.config["seed"])
            model = PolicyValueNet(ModelConfig.tiny() if self.tiny else ModelConfig())
        trainer = (
            restored_trainer
            if payload
            else Trainer(
                model,
                self.device,
                precision="bf16"
                if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
                else "fp32",
                optimizer="auto",
                objective=self.config["method"],
            )
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
            "version": self.version,
            "config": asdict(self.trainer.model.config),
            "metadata": self.meta,
            "updates": self.trainer.updates,
            "model": {k: v.detach().cpu() for k, v in self.trainer.model.state_dict().items()},
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
        }
        if psutil.virtual_memory().available < 256 * 2**20:
            raise MemoryError("可用RAM不足256MiB；已暂停并丢弃在途局")
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
            }
        )

    def _new_pool(self, count):
        from ..native.execution import PolicyInference

        return Pool(
            self.engine,
            None,
            count,
            self.device,
            inference=(PolicyInference(self.trainer.model, self.device), self.version),
            frame_limit=min(
                32 * 2**20, max(4 * 2**20, psutil.virtual_memory().available // (16 * count))
            ),
        )

    def _sample(self):
        self.state = "sampling"
        if self.pool is None:
            self.pool = self._new_pool(self.config["environments"])
        self.pool.model_hash = self.version
        historical = None
        names = self.store.names("history")
        # 先去除同权重，再按训练轮次选择；评测/优化抽样不改变历史轮转。
        for name in (
            names[self.generation % max(1, len(names)) :]
            + names[: self.generation % max(1, len(names))]
        ):
            weight = torch.load(self.store.root / name, weights_only=True, map_location="cpu")
            if weight["version"] != self.version:
                old = PolicyValueNet(self.trainer.model.config)
                old.load_state_dict(weight["model"])
                historical = (old.to(self.device).eval(), weight["version"])
                break
            del weight
        jobs = MemoryJobs(self, self.version, historical)
        self.pool.run(jobs, cancel_event=self.cancel, keep_open=True, snapshot=self._progress)
        self.runtime["inflight_bytes"] = 0
        self.runtime["active"] = []

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
            batch = collate_examples(examples, self.trainer.model.config)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.trainer.step(batch)
            self.update_progress = step + 1
        self.version = identity(self.trainer.model)
        if self.update_progress:
            self.generation += 1
            self.samples.expire(self.generation)
            self.losses.append(
                {"step": self.trainer.updates, "time": time.time(), **self.trainer.last_losses}
            )
            self.losses = self._compact(self.losses)
            if not all(self.trainer.health()[k] for k in ("finite_parameters", "finite_optimizer")):
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
            if not self.evaluation or self.cancel.is_set() or self.eval_cancel.is_set():
                break

    def _close_pool(self):
        if self.pool:
            self.pool.close()
            self.pool = None

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
                if getattr(self, "switch_request", None):
                    self._switch_experiment()
                if self.new_request:
                    self._new_experiment()
                if getattr(self, "want_restore", False):
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
                    or getattr(self, "want_export", False)
                ):
                    if self.state == "pausing":
                        self._pause()
                    self.wake.wait(0.25)
                    self.wake.clear()
                    continue
                self._initialize()
                if getattr(self, "want_export", False):
                    self.want_export = False
                    name = self.store.save_tensor(
                        "export", self.trainer.updates, self._weights(), 1
                    )
                    self.log(f"导出完成（受管，仅保留最新一份）：{self.store.root / name}")
                if self.want_save:
                    self._save()
                if (self.want_eval or self.evaluation) and not self.cancel.is_set():
                    self._evaluate()
                if self.want_run and not self.cancel.is_set():
                    self._sample()
                    if not self.cancel.is_set():
                        self._update()
                        if self._due_save():
                            self._save()
                        if (
                            not self.cancel.is_set()
                            and self.want_run
                            and ready()
                            and self.trainer.updates > self.eval_step
                            and self.counts["games"] - self.eval_games >= self.config["eval_games"]
                        ):
                            self.want_eval = True
                if (not self.want_run and not self.evaluation) or self.cancel.is_set():
                    self._pause()
            except Exception as error:
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
        self.counts = {
            key: 0
            for key in (
                "games",
                "truncated",
                "discarded",
                "tasks",
                "eval_tasks",
                "seen_samples",
                "retained_samples",
                "oversize_samples",
            )
        }
        self.saved_step = self.saved_games = self.eval_games = self.eval_step = (
            self.history_step
        ) = self.generation = self.round_number = 0
        self.saved_at = None
        self.losses, self.evaluations = [], []
        self.evaluation = self.candidate = None
        self.restored = False
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
