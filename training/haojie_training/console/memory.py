"""有界终局样本：Pass保留少量名额，各组内均匀蓄水；超大样本拒收而非截断实体。"""

import random
import time
from collections import Counter, deque

import torch

from ..native.pipeline import example


def size_of(row):
    # 各张量可能共享同一协议帧；这里逐字段求和并加对象余量是保守上界。
    return (
        sum(t.numel() * t.element_size() for t in row.values() if isinstance(t, torch.Tensor))
        + 4096
    )


class Trajectory:
    def __init__(self, seed, capacity=128, sample_bytes=512 * 1024, context=None):
        self.rng = random.Random(seed)
        self.capacity, self.sample_bytes = capacity, sample_bytes
        self.choices, self.passes = [], []
        self.pass_limit = min(8, max(1, capacity // 16))
        self.eligible_groups = Counter()
        self.context = context or {}
        self.coverage = Counter()
        self.seen = self.eligible = self.oversize = self.used = 0

    @property
    def rows(self):
        return self.choices + self.passes

    def add(self, message):
        self.seen += 1
        key = "pass" if message.get("pass") else message.get("stage", "unknown")
        self.coverage[key] += 1
        size = message["bytes"] + message["candidates"] * 4 + 4096
        if size > self.sample_bytes:
            self.oversize += 1
            return
        self.eligible += 1
        is_pass = bool(message.get("pass"))
        self.eligible_groups[is_pass] += 1
        group = self.passes if is_pass else self.choices
        limit = self.pass_limit if is_pass else self.capacity - len(self.passes)
        index = self.rng.randrange(self.eligible_groups[is_pass])
        if len(group) == limit and index >= limit:
            return
        row = example(message["input"], message["selected"])
        row["_meta"] = {
            "game": self.context,
            "actor": message["actor"],
            "stage": key,
            "pass": message.get("pass", False),
            "decision": message.get("decision"),
            "entered": time.time(),
        }
        item = (row, message["actor"], message["step"], size_of(row))
        if len(group) < limit:
            if is_pass and len(self.choices) + len(self.passes) == self.capacity:
                # Pass出现时随机缩减普通组，仍是该组的均匀子样本；总额不增加。
                removed = self.choices.pop(self.rng.randrange(len(self.choices)))
                self.used -= removed[3]
            group.append(item)
            self.used += item[3]
        else:
            self.used += item[3] - group[index][3]
            group[index] = item

    def terminal(self, returns):
        for row, actor, step, _ in self.rows:
            row["value"] = torch.tensor(float(returns[str(actor)]))
            # MC Q 对每个实际分解选择监督；状态价值也按该前缀所属方学习。
            row["value_mask"] = torch.tensor(True)
            yield row


class SamplePool:
    def __init__(self, limit, reuse=2, max_age=4):
        self.limit, self.reuse = limit, reuse
        self.rows = deque()
        self.used = self.evicted = self.consumed = 0
        self.max_age, self.generation = max_age, 0
        self.age_evicted = self.capacity_evicted = self.exhausted = 0
        self.coverage = Counter()
        self.retained = Counter()

    def expire(self, generation):
        self.generation = generation
        kept = deque()
        for row, size, uses in self.rows:
            if generation - row.get("_meta", {}).get("generation", generation) > self.max_age:
                self.used -= size
                self.evicted += 1
                self.age_evicted += 1
            else:
                kept.append((row, size, uses))
        self.rows = kept

    @staticmethod
    def keys(row):
        meta = row.get("_meta", {})
        return [meta.get("stage", "unknown"), meta.get("game", {}).get("rules", "unknown")]

    def resize(self, limit):
        self.limit = limit
        while self.rows and self.used > self.limit:
            _, size, _ = self.rows.popleft()
            self.used -= size
            self.evicted += 1
            self.capacity_evicted += 1

    def budget(self, batch, maximum):
        remaining = sum(self.reuse - uses for _, _, uses in self.rows)
        return min(maximum, (remaining + batch - 1) // batch)

    def add(self, row):
        size = size_of(row)
        if size > self.limit:
            self.evicted += 1
            self.capacity_evicted += 1
            return
        while self.rows and self.used + size > self.limit:
            _, old, _ = self.rows.popleft()
            self.used -= old
            self.evicted += 1
            self.capacity_evicted += 1
        row.setdefault("_meta", {})["generation"] = self.generation
        self.rows.append((row, size, 0))
        self.used += size
        self.retained.update(self.keys(row))

    def batch(self, count, rng):
        # 每步随机抽样但不复制全池索引；次数到达上限即释放。
        result = []
        for _ in range(min(count, len(self.rows))):
            index = rng.randrange(len(self.rows))
            self.rows.rotate(-index)
            row, size, uses = self.rows.popleft()
            self.rows.rotate(index)
            result.append(row)
            self.consumed += 1
            self.coverage.update(self.keys(row))
            if uses + 1 < self.reuse:
                self.rows.append((row, size, uses + 1))
            else:
                self.used -= size
                self.exhausted += 1
        return result

    def snapshot(self):
        return {
            "bytes": self.used,
            "limit": self.limit,
            "samples": len(self.rows),
            "evicted": self.evicted,
            "consumed": self.consumed,
            "reuse": self.reuse,
            "exhausted": self.exhausted,
            "age_evicted": self.age_evicted,
            "capacity_evicted": self.capacity_evicted,
            "coverage": dict(self.coverage),
            "retained_coverage": dict(self.retained),
            "max_age": self.max_age,
        }

    def restore_totals(self, saved):
        # 只恢复小型累计计数；样本及其未消费次数不落盘，重启后池仍为空。
        for name in ("evicted", "consumed", "exhausted", "age_evicted", "capacity_evicted"):
            setattr(self, name, saved.get(name, 0))
        self.coverage.update(saved.get("coverage", {}))
        self.retained.update(saved.get("retained_coverage", {}))
