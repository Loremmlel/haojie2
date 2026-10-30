"""CPU 协调状态：对局、阶段、回放、模型引用与恢复点。"""

import copy
import json
import os
from pathlib import Path
import time
from dataclasses import asdict

import numpy as np
import torch

from ..native_pool import Request, search_values
from ..native import Environment
from .targets import Calibration
from ..opponents import Opponents
from .storage import CheckpointPickle, Replay, Weights, atomic_json, clean_storage
from .compute import Compute


class Trainer(Compute):
    def __init__(self, directory, config, resume=False):
        self.directory = Path(directory).resolve()
        self.config = config
        config.validate()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.models = self.directory / "models"
        self.models.mkdir(exist_ok=True)
        self.initialize_network("cpu")
        self.rng = np.random.default_rng(config.seed)
        self.replay = Replay(config.replay_games, max_bytes=config.replay_mib * 1024**2)
        self.weights = Weights(self.models)
        self.pool = Opponents()
        self.published = 0
        self.inference_models = {}
        self.contexts = {}
        self.phase = "collect" if config.warmup_games else "self"
        self.demonstration_inflight = 0
        self.imitation_updates = 0
        self.calibration = Calibration()
        self.bindings = {}
        self.credit = 0.0
        self.next_game = 0
        self.saved_workers = None
        self.pending_versions = {}
        self.stop_requested = False
        self.stats = {
            "updates": 0,
            "completed": 0,
            "truncated": 0,
            "ply_truncated": 0,
            "command_truncated": 0,
            "demonstrations": 0,
            "selfplay_terminal": 0,
            "samples": 0,
            "trained_samples": 0,
            "value_samples": 0,
            "soft_value_samples": 0,
            "inference_roots": 0,
            "inference_decodes": 0,
            "inference_batches": 0,
            "inference_seconds": 0.0,
            "learning_seconds": 0.0,
            "commands": 0,
            "parameters": self.model.description()["parameters"],
        }
        checkpoint = self.directory / "checkpoint.pt"
        if resume:
            # 仅加载用户指定目录中本程序写出的完整恢复快照。
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved["format"] != "haojie-classic-training-v3":
                raise ValueError("续训只接受 haojie-classic-training-v3 检查点")
            self.model.load_state_dict(saved["model"])
            self.optimizer.load_state_dict(saved["optimizer"])
            for group in self.optimizer.param_groups:
                group["foreach"] = False
                group["lr"] = config.learning_rate
            for state in self.optimizer.state.values():
                state["step"] = state["step"].to(device="cpu", dtype=torch.int64)
            self.rng.bit_generator.state = saved["rng"]
            torch.set_rng_state(saved["torch_rng"])
            self.remote_torch_rng = saved["torch_rng"]
            self.remote_cuda_rng = saved["cuda_rng"]
            if self.device.type == "cuda" and saved["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
            self.stats.update(saved["stats"])
            self.phase = saved["phase"]
            self.demonstration_inflight = saved["demonstration_inflight"]
            self.imitation_updates = saved["imitation_updates"]
            self.calibration = Calibration(saved["calibration"])
            self.pool = Opponents(saved["pool"])
            self.published, self.next_game = saved["published"], saved["next_game"]
            self.credit = saved["credit"]
            self.saved_workers = saved["workers"]
            self.pending_versions = {
                lane["game"]["id"]: lane["root_version"]
                for lanes in self.saved_workers.values()
                for lane in lanes
                if lane["root_version"] is not None and lane["root_version"] >= 0
            }
            self.replay = Replay(
                config.replay_games, saved["replay"], config.replay_mib * 1024**2
            )
            self.weights = Weights(
                self.models, saved["weights"], (e["version"] for e in self.pool.entries)
            )
        else:
            if checkpoint.exists() or any(self.models.iterdir()):
                raise ValueError("目录已有训练，请使用 --resume 或新目录")
            self.publish()

    def publish(self):
        self.published = self.stats["updates"]
        if hasattr(self, "remote"):
            self.remote.publish(self.published)
            return
        self.weights.publish(self.published, self.model)
        self.prune_models()

    def archive(self, version):
        if hasattr(self, "remote"):
            self.remote.call("archive", version=version)
        else:
            self.weights.archive(version)

    def game(self):
        if self.phase == "imitation":
            return None
        if (
            self.phase == "collect"
            and self.stats["demonstrations"] + self.demonstration_inflight
            >= self.config.warmup_games
        ):
            return None
        number = self.next_game
        self.next_game += 1
        game = {
            "id": number,
            "seed": int(self.rng.integers(1, 2**32)),
            "learner": 1 + number % 2,
            "opponent": "self",
            "difficulty": "easy",
        }
        if self.phase == "collect":
            game["opponent"] = "demo"
            self.demonstration_inflight += 1
        else:
            draw = self.rng.random()
            if draw < self.config.teacher_fraction:
                game["opponent"] = "teacher"
            elif (
                draw < self.config.teacher_fraction + self.config.history_fraction
                and self.pool.entries
            ):
                version = self.pool.choose(self.rng)
                if version != self.published:
                    game.update(opponent="history", version=version)
        # 开局前决定是否保留完整局；恢复沿用该局的决定，不按优势临时放行。
        game["ply_limit"] = (
            0
            if game["opponent"] == "demo"
            or self.rng.random() < self.config.full_game_fraction
            else self.config.max_ply
        )
        return game

    def resolve(self, index, request):
        binding = (index, request["lane"])
        if binding in self.bindings and self.bindings[binding][0] != request["root"]:
            self.contexts = {k: v for k, v in self.contexts.items() if k[0] != binding}
        if request["model"] != "current":
            version = int(request["model"])
        elif binding in self.bindings and self.bindings[binding][0] == request["root"]:
            version = self.bindings[binding][1]
        else:
            version = self.published
        self.bindings[binding] = (request["root"], version)
        self.pending_versions.pop(request["root"][0], None)
        self.prune_models()
        return version

    def infer(self, jobs):
        if hasattr(self, "remote"):
            return self.remote.infer(jobs)
        return super().infer(jobs)

    def learn(self):
        if self.phase == "collect" or (
            self.phase == "self" and self.credit < self.config.batch
        ):
            return
        batch = self.replay.batch(self.config.batch, self.rng)
        if not batch:
            return
        self.update(batch)
        self.credit = max(0, self.credit - len(batch))
        if self.stats["updates"] % self.config.publish_updates == 0:
            self.publish()
        if self.phase == "imitation":
            self.imitation_updates += 1
            if self.imitation_updates >= self.config.warmup_updates:
                self.phase = "self"
                self.credit = 0.0
                self.publish()

    def event(self, event):
        if event["type"] == "data":
            data = event["data"]
            self.replay.add(data)
            fresh = data["metadata"]["samples"] - data["credited"]
            data["credited"] = data["metadata"]["samples"]
            self.credit += fresh * self.config.sample_reuse
            self.stats["samples"] += fresh
        elif event["type"] == "done":
            game = event["game"]
            if game["opponent"] == "demo":
                self.demonstration_inflight -= 1
            if event["winner"] is None:
                self.stats["truncated"] += 1
                reason = event.get("truncation", "command_limit")
                self.stats[
                    "ply_truncated" if reason == "ply_limit" else "command_truncated"
                ] += 1
                if game["opponent"] == "self":
                    self.boundary_target(event)
            else:
                self.stats["completed"] += 1
                if game["opponent"] == "demo":
                    self.stats["demonstrations"] += 1
                if game["opponent"] == "self":
                    self.stats["selfplay_terminal"] += 1
                    self.credit += self.config.batch
                    if (
                        self.stats["selfplay_terminal"]
                        < self.config.bootstrap_after_games
                    ):
                        self.calibration.observe(
                            event["data"]["metadata"], self.config.calibration_games
                        )
                if game["opponent"] == "history":
                    self.pool.record(
                        game["version"], event["winner"] == 3 - game["learner"]
                    )
                if self.stats["completed"] % self.config.history_games == 0:
                    self.archive(self.published)
                    self.pool.add(self.published, self.stats["completed"])
            if (
                self.phase == "collect"
                and self.stats["demonstrations"] >= self.config.warmup_games
                and self.demonstration_inflight == 0
            ):
                self.phase = "imitation" if self.config.warmup_updates else "self"

    def boundary_target(self, event):
        value = None
        kind = "calibrated"
        version = None
        if self.stats["selfplay_terminal"] >= self.config.bootstrap_after_games:
            with Environment(event["game"]["seed"]) as env:
                env.call(
                    "restore",
                    snapshot=event["environment"],
                )
                info, bundle = env.encode()
                owner = ("boundary", event["game"]["id"])
                bits = self.infer(
                    [(self.published, Request("encode", info["key"], bundle), owner)]
                )[0]
                value = float(search_values(bits)[0])
                version = self.published
                kind = "bootstrap"
                self.contexts.pop((owner, info["key"], version), None)
        else:
            value = self.calibration.predict(event["metrics"], event["actor"])
        if value is None:
            return
        target = {
            "kind": kind,
            "actor": event["actor"],
            "value": max(-0.98, min(0.98, value)),
            "weight": self.config.bootstrap_weight,
            "version": version,
            "source_games": self.stats["selfplay_terminal"],
        }
        metadata = event["data"]["metadata"]
        metadata["soft_target"] = target
        event["soft_target"] = target
        self.credit += self.config.batch

    def checkpoint_snapshot(self, workers, weights=None):
        # 学习与原生线程均已暂停；未完成搜索在恢复时用原命令 RNG 重算。
        for lanes in workers.values():
            for lane in lanes:
                self.event(
                    {
                        "type": "data",
                        "data": lane["buffer"],
                    }
                )
        payload = {
            "format": "haojie-classic-training-v3",
            "config": asdict(self.config),
            "phase": self.phase,
            "demonstration_inflight": self.demonstration_inflight,
            "imitation_updates": self.imitation_updates,
            "calibration": self.calibration.snapshot(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": self.rng.bit_generator.state,
            "torch_rng": getattr(self, "remote_torch_rng", torch.get_rng_state()),
            "cuda_rng": (
                torch.cuda.get_rng_state_all()
                if self.device.type == "cuda"
                else getattr(self, "remote_cuda_rng", None)
            ),
            "stats": self.stats,
            "pool": self.pool.entries,
            "replay": self.replay.games,
            "published": self.published,
            "next_game": self.next_game,
            "weights": (
                self.weights.snapshot(self.live_versions())
                if weights is None
                else weights
            ),
            "credit": self.credit,
            "workers": workers,
        }
        # 打包数组只读，可跨保存线程共享；复制可变容器并保留回放/在途缓冲别名。
        buffers = self.replay.games + [
            lane["buffer"] for lanes in workers.values() for lane in lanes
        ]
        memo = {
            id(array): array
            for data in buffers
            for chunk in data["chunks"]
            for array in chunk.values()
        }
        return copy.deepcopy(payload, memo)

    def write_checkpoint(self, payload):
        started = time.perf_counter()
        path = self.directory / "checkpoint.pt"
        torch.save(payload, path.with_suffix(".tmp"), pickle_module=CheckpointPickle)
        os.replace(path.with_suffix(".tmp"), path)
        return time.perf_counter() - started, {e["version"] for e in payload["pool"]}

    def checkpoint(self, workers, weights=None):
        payload = self.checkpoint_snapshot(workers, weights)
        _, history = self.write_checkpoint(payload)
        clean_storage(self.directory, history)

    def run(self):
        from .pipeline import run

        return run(self)

    def status(self, status, start, initial, **extra):
        elapsed = time.perf_counter() - start
        value = {
            "status": status,
            **self.stats,
            "elapsed_seconds": elapsed,
            "phase": self.phase,
            "imitation_updates": self.imitation_updates,
            "calibration": self.calibration.validation,
            "commands_per_second": (self.stats["commands"] - initial["commands"])
            / max(elapsed, 0.001),
            "published": self.published,
            "replay_bytes": self.replay.bytes,
            "replay_games": len(self.replay.games),
            "replay_limit_bytes": self.replay.max_bytes,
            "config": asdict(self.config),
            **extra,
        }
        if self.device.type == "cuda":
            value["cuda_peak_bytes"] = torch.cuda.max_memory_allocated()
        atomic_json(self.directory / "status.json", value)
        print(json.dumps(value), flush=True)
