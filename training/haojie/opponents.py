"""按近期、跨时期锚点与挑战表现抽取历史模型。"""


class Opponents:
    def __init__(self, entries=None):
        self.entries = entries or []

    def add(self, version, games):
        if not any(e["version"] == version for e in self.entries):
            self.entries.append(
                {"version": version, "games": games, "wins": 0, "matches": 0}
            )

    def choose(self, rng):
        if not self.entries:
            return None
        recent = self.entries[-8:]
        # 全历史按完成局数的对数区间留下锚点，早期模型不会被近期窗口挤走。
        anchors = {}
        for entry in self.entries:
            bucket = max(1, entry["games"]).bit_length()
            anchors.setdefault(bucket, entry)
        challenging = sorted(
            self.entries,
            key=lambda e: (e["wins"] + 1) / (e["matches"] + 2),
            reverse=True,
        )[:8]
        bucket = int(rng.choice(3, p=[0.4, 0.4, 0.2]))
        choices = [recent, list(anchors.values()), challenging][bucket]
        return choices[int(rng.integers(len(choices)))]["version"]

    def record(self, version, won):
        for entry in self.entries:
            if entry["version"] == version:
                entry["matches"] += 1
                entry["wins"] += int(won)
