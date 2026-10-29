"""有界内存回放、历史权重缓存和单个恢复点的文件管理。"""

import json
import os
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    os.replace(temporary, path)


class GameBuffer:
    def __init__(self, game, previous=None):
        self.game = game
        self.samples = []
        self.data = previous or {
            "chunks": [],
            "commands": [],
            "bytes": 0,
            "credited": 0,
        }
        self.metadata = self.data.setdefault(
            "metadata",
            {
                "format": "autoregressive-v2",
                "game": game,
                "samples": 0,
                "timeline": [],
                "winner": None,
                "steps": 0,
                "status": "running",
            },
        )
        if self.metadata.get("format") != "autoregressive-v2":
            raise ValueError("轨迹格式不兼容，请新建训练目录")

    def sample(self, bundle, decisions, actor, ply, version):
        if decisions:
            self.samples.append(
                (bundle, decisions, actor, ply, version, self.metadata["steps"])
            )

    def transition(self, info, command):
        self.metadata["steps"] += 1
        self.metadata["timeline"].append([info["ply"], *info["metrics"]])
        self.data["commands"].append(command)
        self.metadata["winner"] = info["winner"]
        self.metadata["status"] = (
            "terminal" if info["winner"] is not None else "running"
        )

    def flush(self):
        if self.samples:
            bundles, decisions, actors, plies, versions, steps = zip(*self.samples)
            states, records, owners, relations = zip(*bundles)

            def offsets(arrays):
                return np.cumsum([0] + [len(a) for a in arrays])

            all_decisions = [d for sample in decisions for d in sample]
            options, targets, weights = zip(*all_decisions)
            chunk = dict(
                states=np.concatenate(states),
                records=np.concatenate(records),
                owners=np.concatenate(owners),
                relations=np.concatenate([r.reshape(-1, 8) for r in relations]),
                state_offsets=offsets(states),
                record_offsets=offsets(records),
                relation_offsets=np.cumsum([0] + [len(s) ** 2 for s in states]),
                decision_offsets=offsets(decisions),
                option_offsets=offsets(options),
                options=np.concatenate(options),
                targets=np.concatenate(targets),
                weights=np.asarray(weights),
                actors=np.asarray(actors),
                plies=np.asarray(plies),
                versions=np.asarray(versions),
                steps=np.asarray(steps),
            )
            self.data["chunks"].append(chunk)
            self.data["bytes"] += sum(a.nbytes for a in chunk.values())
            self.metadata["samples"] += len(self.samples)
            self.samples.clear()
        return self.data


class Replay:
    def __init__(self, capacity=128, games=None, max_bytes=4 * 1024**3):
        self.capacity = capacity
        self.max_bytes = max_bytes
        self.games = []
        self.bytes = 0
        for game in games or []:
            Replay.add(self, game)

    def add(self, game):
        if any(
            sum(a.nbytes for a in c.values()) > self.max_bytes for c in game["chunks"]
        ):
            raise ValueError("单个回放片段超过内存预算，请增大 --replay-mib")
        if not any(g is game for g in self.games):
            self.games.append(game)
        self.bytes = sum(g["bytes"] for g in self.games)
        while len(self.games) > self.capacity or self.bytes > self.max_bytes:
            old = self.games.pop(0)
            self.bytes -= old["bytes"]
            old["chunks"].clear()
            old["bytes"] = 0

    def batch(self, size, rng):
        available = []
        for game in self.games:
            if game["chunks"]:
                available.append((game, game["metadata"]))
        if not available:
            return []
        terminal = [
            p
            for p in available
            if p[1]["winner"] is not None and p[1]["game"]["opponent"] == "self"
        ]
        result = []
        for sample_index in range(size):
            population = terminal if terminal and sample_index % 2 == 0 else available
            game, meta = population[int(rng.integers(len(population)))]
            data = game["chunks"][int(rng.integers(len(game["chunks"])))]
            i = int(rng.integers(len(data["actors"])))

            def segment(key, offsets, index=i):
                return data[key][data[offsets][index] : data[offsets][index + 1]]

            states = segment("states", "state_offsets")
            bundle = (
                states,
                segment("records", "record_offsets"),
                segment("owners", "record_offsets"),
                segment("relations", "relation_offsets").reshape(
                    len(states), len(states), 8
                ),
            )
            decisions = [
                (
                    segment("options", "option_offsets", d),
                    segment("targets", "option_offsets", d),
                    float(data["weights"][d]),
                )
                for d in range(
                    data["decision_offsets"][i], data["decision_offsets"][i + 1]
                )
            ]
            actor = int(data["actors"][i])
            winner = meta["winner"]
            valid = winner is not None and meta["game"]["opponent"] == "self"
            value = (
                0.0 if winner in (None, "draw") else (1.0 if winner == actor else -1.0)
            )
            value_weight = 1.0 if valid else 0.0
            soft = (
                meta.get("soft_target")
                if meta["game"]["opponent"] == "self" and winner is None
                else None
            )
            if soft:
                value = soft["value"] * (1 if actor == soft["actor"] else -1)
                value_weight = soft["weight"]
            timeline = meta["timeline"]
            step = int(data["steps"][i])
            future = next(
                (v for v in timeline[step:] if v[0] >= data["plies"][i] + 2), None
            )
            current = timeline[step - 1][1:] if step else [0.0] * 4
            aux = (
                ((np.asarray(future[1:]) - current) * (1 if actor == 1 else -1)).astype(
                    np.float32
                )
                if future
                else np.zeros(4, np.float32)
            )
            result.append(
                (
                    bundle,
                    decisions,
                    value,
                    value_weight,
                    aux,
                    future is not None,
                    valid,
                    soft is not None,
                )
            )
        return result


class Weights:
    """临时发布只在内存；历史文件按需读入 CPU 缓存。"""

    def __init__(self, directory, states=None, history=(), cache_bytes=256 * 1024**2):
        self.directory = Path(directory)
        self.states = OrderedDict(states or {})
        self.history = set(history)
        self.cache_bytes = cache_bytes

    def get(self, version):
        if version not in self.states:
            if version not in self.history:
                raise KeyError(f"临时模型 {version} 已无存活引用")
            self.states[version] = torch.load(
                self.directory / f"model-{version:08d}.pt",
                map_location="cpu",
                weights_only=True,
            )
        self.states.move_to_end(version)
        return self.states[version]

    def publish(self, version, model):
        self.states[version] = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

    def archive(self, version):
        if version not in self.history:
            path = self.directory / f"model-{version:08d}.pt"
            torch.save(self.get(version), path.with_suffix(".tmp"))
            os.replace(path.with_suffix(".tmp"), path)
            self.history.add(version)

    def prune(self, live):
        for version in list(self.states):
            if version not in live and version not in self.history:
                del self.states[version]
        size = sum(
            v.numel() * v.element_size()
            for s in self.states.values()
            for v in s.values()
        )
        for version in list(self.states):
            if size <= self.cache_bytes:
                break
            if version not in live:
                size -= sum(
                    v.numel() * v.element_size() for v in self.states[version].values()
                )
                del self.states[version]

    def snapshot(self, live):
        return {v: self.get(v) for v in live if v not in self.history}


def clean_storage(directory, history):
    """恢复点成功后，清理不属于已保存历史池的孤立归档。"""
    directory = Path(directory).resolve()
    models = directory / "models"
    if not models.is_symlink() and models.resolve().parent == directory:
        for path in models.glob("model-*.pt"):
            version = path.stem.removeprefix("model-")
            if version.isdigit() and int(version) not in history:
                path.unlink()
