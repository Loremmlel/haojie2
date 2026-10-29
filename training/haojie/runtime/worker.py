"""CPU worker 同时推进多局；只在完整命令边界提交，GPU 请求通过共享槽组批。"""

import copy
import os
import signal
import time
import traceback
from pathlib import Path

import numpy as np

from ..native import Environment
from ..opponents import Teacher
from ..search import search, demonstration
from .ipc import Slot, pack, unpack, search_values
from .storage import GameWriter, atomic_json


class Lane:
    def __init__(self, directory, config, game, restored=None):
        self.config, self.game, self.directory = config, game, directory
        self.environment = Environment(game["seed"])
        self.rng = np.random.default_rng(game["seed"] ^ 0xA53113)
        self.writer = GameWriter(
            directory, game, restored["metadata"] if restored else None
        )
        self.teacher = (
            Teacher(game.get("difficulty", "easy"), self.environment)
            if game["opponent"] in ("teacher", "demo")
            else None
        )
        self.generator = None
        self.request = None
        self.root_rng = None
        self.root_teacher = None
        self.root_version = None
        self.deep = False
        self.info = self.environment.call("info")
        if restored:
            self.environment.call("restore", snapshot=restored["environment"])
            self.rng.bit_generator.state = restored["rng"]
            self.root_version = restored["root_version"]
            self.info = self.environment.call("info")
            if self.teacher and restored["teacher"]:
                self.teacher.restore(restored["teacher"])

    def owner_model(self):
        if (
            self.game["opponent"] == "history"
            and self.info["actor"] != self.game["learner"]
        ):
            return self.game["version"]
        return self.root_version if self.root_version is not None else "current"

    def commit(self, root, policy, version):
        if self.config.temperature == 0:
            index = int(np.argmax(policy))
        else:
            probability = np.power(
                policy.astype(np.float64), 1 / self.config.temperature
            )
            probability /= probability.sum()
            index = int(self.rng.choice(len(probability), p=probability))
        result = self.environment.call("apply", command=root.edges[index].command)
        if result.get("rejected"):
            raise RuntimeError(f"正式命令被拒绝: {result}")
        learning = (
            self.game["opponent"] in ("self", "demo")
            or root.info["actor"] == self.game["learner"]
        )
        if learning and self.deep:
            self.writer.sample(
                root.bundle,
                root.supervision(policy),
                root.info["actor"],
                root.info["ply"],
                version,
            )
        self.writer.transition(result, result["command"])
        self.info = result
        (
            self.generator,
            self.request,
            self.root_rng,
            self.root_version,
            self.root_teacher,
        ) = None, None, None, None, None

    def advance(self, answer=None):
        events = []
        if self.generator is None:
            self.root_rng = copy.deepcopy(self.rng.bit_generator.state)
            teacher_turn = self.game["opponent"] == "demo" or (
                self.game["opponent"] == "teacher"
                and self.info["actor"] != self.game["learner"]
            )
            self.deep = bool(self.rng.random() < self.config.full_fraction)
            if teacher_turn:
                self.root_teacher = self.teacher.snapshot()
                self.chosen_teacher = self.teacher.next()
                if self.game["opponent"] == "teacher":
                    result = self.environment.call("apply", command=self.chosen_teacher)
                    if result.get("rejected"):
                        raise RuntimeError(f"手工对手命令被拒绝: {result}")
                    self.writer.transition(result, result["command"])
                    self.info = result
                    self.root_rng = self.root_teacher = self.root_version = None
                    return None, self.events()
                info, bundle, decisions = demonstration(
                    self.environment, self.chosen_teacher
                )
                result = self.environment.call("apply", command=self.chosen_teacher)
                if result.get("rejected"):
                    raise RuntimeError(f"教师命令被拒绝: {result}")
                self.writer.sample(bundle, decisions, info["actor"], info["ply"], -1)
                self.writer.transition(result, result["command"])
                self.info = result
                self.root_rng = self.root_teacher = self.root_version = None
                return None, self.events()
            self.generator = search(
                self.environment,
                self.rng,
                simulations=self.config.simulations
                if self.deep
                else self.config.fast_simulations,
                depth=self.config.depth,
                noise=self.game["opponent"] != "evaluation",
            )
            answer = None
        try:
            if answer is not None:
                self.root_version = answer[1]
            self.request = (
                next(self.generator) if answer is None else self.generator.send(answer)
            )
            return self.request, events
        except StopIteration as completed:
            root, policy, version = completed.value
            self.commit(root, policy, version)
            return None, self.events()

    def events(self):
        events = []
        terminal = self.info["winner"] is not None
        # ply 是当前单方回合编号；完成上限回合、进入下一回合时截断。
        # 没有分配回合上限的在途局继续按原来的命令硬上限运行。
        ply_limit = self.game.get("ply_limit", 0)
        truncation = (
            "command_limit"
            if self.writer.metadata["steps"] >= self.config.max_commands
            else "ply_limit"
            if ply_limit and self.info["ply"] > ply_limit
            else None
        )
        limited = truncation is not None
        if (
            terminal
            or limited
            or self.writer.metadata["steps"] % self.config.chunk_commands == 0
        ):
            path = self.writer.flush()
            events.append({"type": "data", "path": path})
        if terminal or limited:
            if limited and not terminal:
                self.writer.metadata["status"] = "truncated"
                self.writer.metadata["truncation"] = truncation
                self.writer.flush()
                atomic_json(
                    self.writer.path / "environment.json",
                    self.environment.call("snapshot"),
                )
            events.append(
                {
                    "type": "done",
                    "game": self.game,
                    "winner": self.info["winner"],
                    "steps": self.writer.metadata["steps"],
                    "ply": self.info["ply"],
                    "path": str(self.writer.path / "game.json"),
                    "actor": self.info["actor"],
                    "metrics": self.info["metrics"],
                    "truncation": None if terminal else truncation,
                }
            )
        return events

    def snapshot(self):
        self.writer.flush()
        return {
            "game": self.game,
            "environment": self.environment.call("snapshot"),
            "rng": self.root_rng or self.rng.bit_generator.state,
            "root_version": self.root_version,
            "metadata": self.writer.metadata,
            "teacher": (
                self.root_teacher
                if self.root_rng is not None
                and (
                    self.game["opponent"] == "demo"
                    or self.info["actor"] != self.game["learner"]
                )
                else self.teacher.snapshot()
            )
            if self.teacher
            else None,
        }

    def close(self):
        self.environment.close()
        if self.teacher:
            self.teacher.close()


def worker(index, connection, slot_name, directory, config, games, restored):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    slot = Slot(slot_name)
    lanes = {}
    answers = {}
    start = time.perf_counter()
    commands = 0
    try:
        for i, game in enumerate(games):
            if game is not None:
                lanes[i] = Lane(
                    directory, config, game, restored[i] if restored else None
                )
        while True:
            requests, arrays, events = [], [], []
            for i, lane in list(lanes.items()):
                previous = lane.writer.metadata["steps"]
                request, news = lane.advance(answers.pop(i, None))
                commands += lane.writer.metadata["steps"] - previous
                events.extend({**event, "lane": i} for event in news)
                if any(event["type"] == "done" for event in news):
                    lane.close()
                    del lanes[i]
                elif request is not None:
                    requests.append(
                        {
                            "lane": i,
                            "model": lane.owner_model(),
                            "root": f"{lane.game['id']}:{lane.writer.metadata['steps']}",
                            "op": request.op,
                            "context": request.context,
                            "arrays": len(request.arrays),
                        }
                    )
                    arrays.extend(request.arrays)
            needed = sum(a.nbytes for a in arrays)
            if needed > slot.memory.size:
                connection.send(("grow", needed))
                slot.close()
                slot = Slot(connection.recv())
            shapes = pack(slot, arrays)
            connection.send(
                (
                    "work",
                    requests,
                    shapes,
                    events,
                    {
                        "commands": commands,
                        "elapsed": time.perf_counter() - start,
                        "occupied": list(lanes),
                    },
                )
            )
            response = connection.recv()
            if response["op"] in ("pause", "save"):
                for request, version in zip(requests, response.get("versions", [])):
                    lanes[request["lane"]].root_version = version
                snapshots = []
                for i, lane in lanes.items():
                    snapshot = lane.snapshot()
                    snapshots.append({"lane": i, **snapshot})
                connection.send(("saved", snapshots, commands))
                if response["op"] == "pause":
                    break
                response = connection.recv()
            outputs = unpack(slot, response["shapes"], copy=True)
            for request, output, version in zip(
                requests, outputs, response["versions"]
            ):
                output = search_values(output)
                answers[request["lane"]] = (output, version)
            for assignment in response.get("assignments", []):
                lanes[assignment["lane"]] = Lane(directory, config, assignment["game"])
    except BaseException:
        failure = traceback.format_exc()
        saved = []
        for i, lane in lanes.items():
            try:
                saved.append({"lane": i, **lane.snapshot()})
            except Exception:
                pass
        # 初始化失败时，仍保留尚未成功打开的原检查点局面。
        recovered = {lane["lane"] for lane in saved}
        for i, previous in enumerate(restored or []):
            if previous and i not in recovered:
                saved.append({"lane": i, **previous})
        atomic_json(
            Path(directory) / f"failed-worker-{index}.json",
            {"pid": os.getpid(), "error": failure, "lanes": saved},
        )
        connection.send(("error", failure))
        raise
    finally:
        for lane in lanes.values():
            lane.close()
        slot.close()
        connection.close()
