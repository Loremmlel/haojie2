"""CPU 协调状态：对局、阶段、回放、模型引用与恢复点。"""

import json
import os
from pathlib import Path
import time
from dataclasses import asdict

import numpy as np
import torch

from ..search import Request
from ..native import Environment
from .targets import Calibration
from ..opponents import Opponents
from .storage import Replay, atomic_json
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
        self.replay = Replay(config.replay_games)
        self.pool = Opponents()
        self.published = 0
        self.inference_models = {}
        self.contexts = {}
        self.phase = "collect" if config.warmup_games else "self"
        self.demonstration_inflight = 0
        self.imitation_updates = 0
        self.calibration = Calibration()
        self.bindings = {}
        self.seen_samples = {}
        self.credit = 0.0
        self.next_game = 0
        self.saved_workers = None
        self.checkpoint_versions = set()
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
            if saved["format"] != "haojie-classic-training-v2":
                raise ValueError("训练格式已改为分层解码 v2，请保留旧目录并启动新训练")
            old_config = saved["config"]
            for key in ("workers", "games_per_worker"):
                if old_config[key] != getattr(config, key):
                    raise ValueError(
                        f"续训保持 {key}={old_config[key]}，请先按原并发恢复"
                    )
            self.model.load_state_dict(saved["model"])
            self.optimizer.load_state_dict(saved["optimizer"])
            for group in self.optimizer.param_groups:
                group["foreach"] = False
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
            self.replay = Replay(config.replay_games, saved["replay"])
            self.pool = Opponents(saved["pool"])
            self.published, self.next_game = saved["published"], saved["next_game"]
            self.seen_samples, self.credit = saved["seen_samples"], saved["credit"]
            self.saved_workers = saved["workers"]
            self.checkpoint_versions = {saved["published"]} | {
                lane["root_version"]
                for lanes in self.saved_workers.values()
                for lane in lanes
                if lane["root_version"] is not None and lane["root_version"] >= 0
            }
            # 检查点后的数据文件保留；当前恢复分支只关联已提交的元数据前缀。
            for lanes in self.saved_workers.values():
                for lane in lanes:
                    atomic_json(
                        self.directory
                        / "games"
                        / str(lane["game"]["id"])
                        / "game.json",
                        lane["metadata"],
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
        state = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        path = self.models / f"model-{self.published:08d}.pt"
        if not path.exists():
            torch.save(state, path)
        retained = (
            {self.published}
            | self.checkpoint_versions
            | {v for _, v in self.bindings.values()}
            | {e["version"] for e in self.pool.entries}
        )
        for candidate in self.models.glob("model-*.pt"):
            if int(candidate.stem.split("-")[1]) not in retained:
                candidate.unlink()

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
            path = event["path"]
            self.replay.add(path)
            meta = json.loads(Path(path).read_text(encoding="utf-8"))
            fresh = meta["samples"] - self.seen_samples.get(path, 0)
            self.seen_samples[path] = meta["samples"]
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
                            event["path"], self.config.calibration_games
                        )
                if game["opponent"] == "history":
                    self.pool.record(
                        game["version"], event["winner"] == 3 - game["learner"]
                    )
                if self.stats["completed"] % self.config.history_games == 0:
                    self.pool.add(self.published, self.stats["completed"])
            with (self.directory / "results.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
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
                    snapshot=json.loads(
                        (Path(event["path"]).parent / "environment.json").read_text(
                            encoding="utf-8"
                        )
                    ),
                )
                info, bundle = env.encode()
                owner = ("boundary", event["game"]["id"])
                from .ipc import search_values

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
        metadata = json.loads(Path(event["path"]).read_text(encoding="utf-8"))
        metadata["soft_target"] = target
        atomic_json(event["path"], metadata)
        event["soft_target"] = target
        self.credit += self.config.batch

    def checkpoint(self, workers):
        # 学习暂停且所有 worker 已交还槽；未完成搜索在恢复时用原命令 RNG 重算。
        for lanes in workers.values():
            for lane in lanes:
                self.event(
                    {
                        "type": "data",
                        "path": str(
                            self.directory
                            / "games"
                            / str(lane["game"]["id"])
                            / "game.json"
                        ),
                    }
                )
        payload = {
            "format": "haojie-classic-training-v2",
            "config": asdict(self.config),
            "phase": self.phase,
            "demonstration_inflight": self.demonstration_inflight,
            "imitation_updates": self.imitation_updates,
            "calibration": self.calibration.snapshot(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": self.rng.bit_generator.state,
            "torch_rng": getattr(self, "remote_torch_rng", torch.get_rng_state()),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if self.device.type == "cuda"
            else getattr(self, "remote_cuda_rng", None),
            "stats": self.stats,
            "pool": self.pool.entries,
            "replay": self.replay.games,
            "published": self.published,
            "next_game": self.next_game,
            "seen_samples": self.seen_samples,
            "credit": self.credit,
            "workers": workers,
        }
        path = self.directory / "checkpoint.pt"
        torch.save(payload, path.with_suffix(".tmp"))
        os.replace(path.with_suffix(".tmp"), path)
        self.checkpoint_versions = {self.published} | {
            lane["root_version"]
            for lanes in workers.values()
            for lane in lanes
            if lane["root_version"] is not None and lane["root_version"] >= 0
        }

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
            **extra,
        }
        if self.device.type == "cuda":
            value["cuda_peak_bytes"] = torch.cuda.max_memory_allocated()
        atomic_json(self.directory / "status.json", value)
        print(json.dumps(value, ensure_ascii=False), flush=True)
