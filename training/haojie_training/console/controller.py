"""单一后台控制器拥有训练生命周期；HTTP仅提交意图，GPU阶段串行执行。"""

import copy
import hashlib
import json
import random
import threading
import time
from collections import deque
from dataclasses import asdict

import psutil
import torch

from ..data import collate_examples
from ..model import ModelConfig, PolicyValueNet
from ..native.client import Client
from ..native.pipeline import metadata, model_from
from ..native.pool import Pool
from ..runtime import Trainer, resolve_device
from .memory import SamplePool
from .opponents import Teacher, ready, schedule, summarize
from .sampling import MemoryJobs

METHOD = "decomposed-mc-q-v1"
FORMAT = "haojie-stream-recovery-v1"


def defaults():
    return {
        "environments": 4,
        "games_per_round": 4,
        "updates_per_round": 8,
        "batch_size": 4,
        "pool_mib": min(4096, max(64, int(psutil.virtual_memory().available / 4 / 2**20))),
        "save_seconds": 600,
        "save_games": 0,
        "eval_games": 200,
        "eval_pairs": 1,
        "eval_seconds": 600,
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
        "updates_per_round": (1, 1000),
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
                    "losses": self.losses,
                    "evaluations": self.evaluations,
                    "logs": list(self.logs),
                    "resources": self.telemetry,
                    "external": self.external,
                    "learning_status": "自对弈数据与训练接口已接通，策略改进实验尚未完成",
                }
            )

    def command(self, action, values=None):
        with self.lock:
            if action == "start":
                if self.state in ("idle", "paused", "error"):
                    self.want_run, self.error = True, None
                    self.cancel.clear()
                    self.state = "starting"
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
                self.want_eval = True
                if not self.want_run:
                    self.cancel.clear()
            elif action == "cancel-evaluation":
                self.want_eval = False
                self.eval_cancel.set()
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
                self.samples.limit = config["pool_mib"] * 2**20
                while self.samples.used > self.samples.limit and self.samples.rows:
                    _, size, _ = self.samples.rows.popleft()
                    self.samples.used -= size
                    self.samples.evicted += 1
            else:
                raise ValueError("未知操作")
            self.wake.set()

    def _initialize(self):
        if self.trainer:
            self._device(self.device)
            return
        with Client(self.engine) as client:
            self.ready = client.ready
        if "memory-examples-v1" not in self.ready.get("capabilities", []):
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
                        optimizer="foreach",
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
            else Trainer(model, self.device, optimizer="foreach", objective=self.config["method"])
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
            self.eval_games, self.eval_step = payload.get("eval_cursor", [0, 0])
            self.history_step = payload.get("history_step", 0)
            self.losses, self.evaluations = (
                payload.get("losses", []),
                payload.get("evaluations", []),
            )
            self.restored = True
            self.log(f"恢复至更新 {self.saved_step}；样本池为空，恢复点之后未保存的进度不在磁盘中")
        self.trainer = trainer
        self.samples = SamplePool(self.config["pool_mib"] * 2**20)
        self.version = identity(self.trainer.model)
        if not self.store.names("history"):
            self._history()
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
            "eval_cursor": [self.eval_games, self.eval_step],
            "history_step": self.history_step,
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
        }
        if psutil.virtual_memory().available < 256 * 2**20:
            raise MemoryError("可用RAM不足256MiB；已暂停并丢弃在途局")
        if self._due_save() and not self.cancel.is_set():
            self._save()
        if time.monotonic() - self.last_metrics >= 60:
            self.store.reconcile()
            self.store.metrics(
                {"losses": self.losses, "evaluations": self.evaluations, "logs": list(self.logs)}
            )
            self.last_metrics = time.monotonic()

    def _new_pool(self, count):
        from ..native.execution import PolicyInference

        return Pool(
            self.engine,
            None,
            count,
            self.device,
            inference=(PolicyInference(self.trainer.model, self.device), self.version),
            frame_limit=2 * 1024 * 1024,
        )

    def _sample(self):
        self.state = "sampling"
        if self.pool is None:
            self.pool = self._new_pool(self.config["environments"])
        self.pool.model_hash = self.version
        historical = None
        names = self.store.names("history")
        if names:
            weight = torch.load(
                self.store.root / self.rng.choice(names), weights_only=True, map_location="cpu"
            )
            if weight["version"] != self.version:
                old = PolicyValueNet(self.trainer.model.config)
                old.load_state_dict(weight["model"])
                historical = (old.to(self.device).eval(), weight["version"])
            del weight
        jobs = MemoryJobs(self, self.version, historical)
        self.pool.run(jobs, cancel_event=self.cancel, keep_open=True, snapshot=self._progress)
        self.runtime["inflight_bytes"] = 0
        self.runtime["active"] = []

    def _update(self):
        self.state = "updating"
        self.trainer.model.train()
        for step in range(self.config["updates_per_round"]):
            if self.cancel.is_set() or not self.samples.rows:
                break
            examples = self.samples.batch(self.config["batch_size"], self.rng)
            batch = collate_examples(examples, self.trainer.model.config)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.trainer.step(batch)
            self.update_progress = step + 1
        self.version = identity(self.trainer.model)
        if self.update_progress:
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
        self.want_eval = False
        self.eval_cancel.clear()
        self.state = "evaluating"
        self.round_number += 1
        plan = schedule(self.config["eval_pairs"], self.round_number)
        began = time.monotonic()
        self.log(f"开始第 {self.round_number} 轮评测：冻结 {self.version[:12]}，{len(plan)} 局")
        self._close_pool()
        for game in plan:
            game["result"] = "unfinished"
            if (
                self.cancel.is_set()
                or self.eval_cancel.is_set()
                or time.monotonic() - began >= self.config["eval_seconds"]
            ):
                continue
            teacher = None
            try:
                teacher = Teacher(game["difficulty"])
                game["budget"] = teacher.budget
                self.pool = self._new_pool(1)
                jobs = MemoryJobs(self, self.version, evaluation=game, teacher=teacher)

                # 评测超时取消及训练暂停共享短轮询边界；不中断正在完成的优化步。
                class Cancel:
                    def is_set(inner):
                        return (
                            self.cancel.is_set()
                            or self.eval_cancel.is_set()
                            or time.monotonic() - began >= self.config["eval_seconds"]
                        )

                self.pool.run(jobs, cancel_event=Cancel())
                if jobs.result:
                    outcome = jobs.result["outcome"]
                    game["reason"] = outcome["reason"]
                    if outcome["terminated"]:
                        game["result"] = (
                            "draw"
                            if outcome["winner"] == "draw"
                            else "win"
                            if outcome["winner"] == game["side"]
                            else "loss"
                        )
            except InterruptedError:
                pass
            except Exception as error:
                game.update(result="error", error=str(error)[:1000])
            finally:
                self._close_pool()
                if teacher:
                    teacher.close()
        from .opponents import ASSETS

        opponent = json.loads((ASSETS / "teacher.json").read_text(encoding="utf-8"))["sha256"]
        series = hashlib.sha256(
            json.dumps(
                {
                    "opponent": opponent,
                    "budget": "production-work",
                    "rules": self.meta,
                    "pairs": self.config["eval_pairs"],
                    "seconds": self.config["eval_seconds"],
                    "commands": self.config["max_commands"],
                    "plies": self.config["max_plies"],
                    "temperature": self.config["temperature"],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        report = {
            "series": series,
            "opponent": opponent,
            "budget": "production-work",
            "model": self.version,
            "updates": self.trainer.updates,
            "games_trained": self.counts["games"],
            "time": time.time(),
            "results": summarize(plan),
            "games": plan,
            "fixture": False,
        }
        self.evaluations.append(report)
        for old in self.evaluations[:-2]:
            old.pop("games", None)
        self.evaluations = self._compact(self.evaluations)
        self.eval_games, self.eval_step = self.counts["games"], self.trainer.updates
        self.store.metrics(
            {"losses": self.losses, "evaluations": self.evaluations, "logs": list(self.logs)}
        )
        # 明确规则：至少每档20个有效对局、整轮完成，才允许“最佳”身份。
        results = report["results"].values()
        if all(r["n"] >= 20 and r["completion"] == 1 for r in results):
            score = sum(r["score_rate"] for r in results) / 3
            best = self.store.names("best")
            previous = torch.load(self.store.root / best[-1], weights_only=True) if best else None
            if (
                not previous
                or previous["evaluation"]["series"] != series
                or score > previous["score"]
            ):
                self.store.save_tensor(
                    "best",
                    self.trainer.updates,
                    {**self._weights(), "score": score, "evaluation": report},
                    1,
                )
        self.log("本轮评测结束；未完成/错误局未计胜负，未用于训练")

    def _close_pool(self):
        if self.pool:
            self.pool.close()
            self.pool = None

    def _pause(self):
        self._close_pool()
        if self.trainer:
            self.trainer.optimizer.zero_grad(set_to_none=True)
            self._device("cpu")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        self.runtime = {"active": [], "inflight_bytes": 0}
        if self.state != "error":
            self.state = "paused"

    def _loop(self):
        while not self.shutdown.is_set():
            try:
                if not (self.want_run or self.want_save or self.want_eval):
                    if self.state == "pausing":
                        self._pause()
                    self.wake.wait(0.25)
                    self.wake.clear()
                    continue
                self._initialize()
                if self.want_save:
                    self._save()
                if self.want_eval and not self.cancel.is_set():
                    self._evaluate()
                if self.want_run and not self.cancel.is_set():
                    self._sample()
                    if not self.cancel.is_set():
                        self._update()
                        if self._due_save():
                            self._save()
                        if (
                            self.trainer.updates > self.eval_step
                            and self.counts["games"] - self.eval_games >= self.config["eval_games"]
                        ):
                            self.want_eval = True
                if not self.want_run or self.cancel.is_set():
                    self._pause()
            except Exception as error:
                self.want_run = self.want_eval = self.want_save = False
                self.error, self.state = str(error), "error"
                self.log(error)
                self._pause()
        self._pause()

    def close(self):
        self.shutdown.set()
        self.cancel.set()
        self.eval_cancel.set()
        self.wake.set()
        self.thread.join(timeout=20)
