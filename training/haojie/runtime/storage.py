"""轨迹分块一次写入；终局及短期未来标签独立关联，不补造截断结果。"""

import json
import os
from pathlib import Path

import numpy as np


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


class GameWriter:
    def __init__(self, directory, game, previous=None):
        self.path = Path(directory) / "games" / str(game["id"])
        self.path.mkdir(parents=True, exist_ok=True)
        self.game = game
        self.samples = []
        self.commands = []
        self.metadata = previous or {"game": game, "chunks": [], "command_segments": [], "samples": 0,
                                     "timeline": [], "winner": None, "steps": 0, "status": "running"}

    def sample(self, states, actions, policy, actor, ply, version):
        self.samples.append((states, actions, policy, actor, ply, version, self.metadata["steps"]))

    def transition(self, info, command):
        self.metadata["steps"] += 1
        self.metadata["timeline"].append([info["ply"], *info["metrics"]])
        self.commands.append(command)
        self.metadata["winner"] = info["winner"]
        self.metadata["status"] = "terminal" if info["winner"] is not None else "running"

    def flush(self):
        if self.commands:
            index = len(self.metadata["command_segments"])
            while (self.path / f"commands-{index:06d}.jsonl").exists():
                index += 1
            name = f"commands-{index:06d}.jsonl"
            with (self.path / name).open("x", encoding="utf-8") as stream:
                for command in self.commands:
                    stream.write(json.dumps(command, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.metadata["command_segments"].append(name)
            self.commands.clear()
        if self.samples:
            index = len(self.metadata["chunks"])
            while (self.path / f"chunk-{index:06d}.npz").exists():
                index += 1
            name = f"chunk-{index:06d}.npz"
            states, actions, policies, actors, plies, versions, steps = zip(*self.samples)
            state_offsets = np.cumsum([0] + [len(v) for v in states])
            action_offsets = np.cumsum([0] + [len(v) for v in actions])
            with (self.path / name).open("xb") as stream:
                np.savez(stream, states=np.concatenate(states), actions=np.concatenate(actions),
                         policies=np.concatenate(policies), state_offsets=state_offsets,
                         action_offsets=action_offsets, actors=actors, plies=plies, versions=versions, steps=steps)
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
        self.games = self.games[-self.capacity:]
        self.cache = {k: v for k, v in self.cache.items() if str(Path(k).parent / "game.json") in self.games}

    def batch(self, size, rng):
        available = []
        for path in self.games:
            metadata = json.loads(Path(path).read_text(encoding="utf-8"))
            if metadata["chunks"]:
                available.append((Path(path), metadata))
        if not available:
            return []
        result = []
        terminal = [(path, meta) for path, meta in available
                    if meta["winner"] is not None and meta["game"]["opponent"] == "self"]
        for sample_index in range(size):
            population = terminal if terminal and sample_index % 2 == 0 else available
            path, meta = population[int(rng.integers(len(population)))]
            chunk = meta["chunks"][int(rng.integers(len(meta["chunks"])))]
            filename = str(path.parent / chunk)
            if filename not in self.cache:
                # 有界缓存避免把整个历史回放常驻内存。
                if len(self.cache) >= self.capacity * 2:
                    self.cache.pop(next(iter(self.cache)))
                with np.load(filename, allow_pickle=False) as data:
                    self.cache[filename] = {k: data[k] for k in data.files}
            data = self.cache[filename]
            i = int(rng.integers(len(data["actors"])))
            so, ao = data["state_offsets"], data["action_offsets"]
            states = data["states"][so[i]:so[i+1]]
            actions = data["actions"][ao[i]:ao[i+1]]
            policy = data["policies"][ao[i]:ao[i+1]]
            actor = int(data["actors"][i])
            winner = meta["winner"]
            valid = winner is not None and meta["game"]["opponent"] == "self"
            value = 0.0 if winner in (None, "draw") else (1.0 if winner == actor else -1.0)
            timeline = meta["timeline"]
            step = int(data["steps"][i])
            future = next((v for v in timeline[step:] if v[0] >= data["plies"][i] + 2), None)
            current = timeline[step - 1][1:] if step else [0., 0., 0., 0.]
            aux = ((np.asarray(future[1:]) - current) * (1 if actor == 1 else -1)).astype(np.float32) if future else np.zeros(4, np.float32)
            result.append((states, actions, policy, value, valid, aux, future is not None))
        return result
