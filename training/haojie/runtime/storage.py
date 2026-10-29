"""完整命令轨迹与自回归监督分块落盘；终局和截断软目标分别关联。"""

import json
import os
from pathlib import Path
import numpy as np


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    os.replace(temporary, path)


class GameWriter:
    def __init__(self, directory, game, previous=None):
        self.path = Path(directory) / "games" / str(game["id"])
        self.path.mkdir(parents=True, exist_ok=True)
        self.game = game
        self.samples = []
        self.commands = []
        self.metadata = previous or {
            "format": "autoregressive-v2",
            "game": game,
            "chunks": [],
            "command_segments": [],
            "samples": 0,
            "timeline": [],
            "winner": None,
            "steps": 0,
            "status": "running",
        }
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
        self.commands.append(command)
        self.metadata["winner"] = info["winner"]
        self.metadata["status"] = (
            "terminal" if info["winner"] is not None else "running"
        )

    def filename(self, prefix, suffix):
        index = 0
        while (self.path / f"{prefix}-{index:06d}.{suffix}").exists():
            index += 1
        return f"{prefix}-{index:06d}.{suffix}"

    def flush(self):
        if self.commands:
            name = self.filename("commands", "jsonl")
            with (self.path / name).open("x", encoding="utf-8") as stream:
                for command in self.commands:
                    stream.write(
                        json.dumps(command, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
            self.metadata["command_segments"].append(name)
            self.commands.clear()
        if self.samples:
            name = self.filename("chunk", "npz")
            bundles, decisions, actors, plies, versions, steps = zip(*self.samples)
            states, records, owners, relations = zip(*bundles)

            def offsets(arrays):
                return np.cumsum([0] + [len(a) for a in arrays])

            all_decisions = [d for sample in decisions for d in sample]
            options, targets, weights = zip(*all_decisions)
            with (self.path / name).open("xb") as stream:
                np.savez(
                    stream,
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
                    weights=weights,
                    actors=actors,
                    plies=plies,
                    versions=versions,
                    steps=steps,
                )
            self.metadata["chunks"].append(name)
            self.metadata["samples"] += len(self.samples)
            self.samples.clear()
        atomic_json(self.path / "game.json", self.metadata)
        return str(self.path / "game.json")


class Replay:
    def __init__(self, capacity=128, games=None):
        self.capacity = capacity
        self.games = games or []
        self.cache = {}

    def add(self, path):
        if path not in self.games:
            self.games.append(path)
        self.games = self.games[-self.capacity :]
        self.cache = {
            k: v
            for k, v in self.cache.items()
            if str(Path(k).parent / "game.json") in self.games
        }

    def batch(self, size, rng):
        available = []
        for path in self.games:
            meta = json.loads(Path(path).read_text(encoding="utf-8"))
            if meta["chunks"]:
                available.append((Path(path), meta))
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
            path, meta = population[int(rng.integers(len(population)))]
            filename = str(
                path.parent / meta["chunks"][int(rng.integers(len(meta["chunks"])))]
            )
            if filename not in self.cache:
                if len(self.cache) >= self.capacity * 2:
                    self.cache.pop(next(iter(self.cache)))
                with np.load(filename, allow_pickle=False) as data:
                    self.cache[filename] = {k: data[k] for k in data.files}
            data = self.cache[filename]
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
