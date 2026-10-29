"""进程内 Rust 三档手工对手与跨时期模型池。"""

from .native import Environment


class Teacher:
    def __init__(self, difficulty="easy", environment=None):
        if difficulty not in ("easy", "medium", "hard"):
            raise ValueError("未知手工 AI 难度")
        self.difficulty = difficulty
        self.owned = environment is None
        self.environment = environment or Environment(1)

    def next(self, observation=None):
        args = {"difficulty": self.difficulty}
        if observation is not None:
            args["observation"] = observation
        return self.environment.call("teacher-next", **args)["command"]

    def snapshot(self):
        return self.environment.call("teacher-snapshot")

    def restore(self, snapshot):
        if snapshot is not None and not isinstance(snapshot, dict):
            raise ValueError("手工 AI 快照需要 Rust 对手格式")
        self.environment.call("teacher-restore", teacher=snapshot)

    def close(self):
        if self.owned:
            self.environment.close()


class Opponents:
    def __init__(self, entries=None):
        self.entries = entries or []

    def add(self, version, games):
        if not any(e["version"] == version for e in self.entries):
            self.entries.append({"version": version, "games": games, "wins": 0, "matches": 0})

    def choose(self, rng):
        if not self.entries:
            return None
        recent = self.entries[-8:]
        # 全历史按完成局数的对数区间留下锚点，早期模型不会被近期窗口挤走。
        anchors = {}
        for entry in self.entries:
            bucket = max(1, entry["games"]).bit_length()
            anchors.setdefault(bucket, entry)
        challenging = sorted(self.entries, key=lambda e: (e["wins"] + 1) / (e["matches"] + 2), reverse=True)[:8]
        bucket = int(rng.choice(3, p=[.4, .4, .2]))
        choices = [recent, list(anchors.values()), challenging][bucket]
        return choices[int(rng.integers(len(choices)))]["version"]

    def record(self, version, won):
        for entry in self.entries:
            if entry["version"] == version:
                entry["matches"] += 1
                entry["wins"] += int(won)
