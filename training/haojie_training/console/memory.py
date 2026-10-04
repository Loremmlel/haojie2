"""有界终局样本：均匀蓄水池保留全局各阶段，超大样本明确拒收而非截断实体。"""

import random
from collections import deque

import torch

from ..native.pipeline import example


def size_of(row):
    # 各张量可能共享同一协议帧；这里逐字段求和并加对象余量是保守上界。
    return sum(t.numel() * t.element_size() for t in row.values()) + 4096


class Trajectory:
    def __init__(self, seed, capacity=64, sample_bytes=512 * 1024):
        self.rng = random.Random(seed)
        self.capacity, self.sample_bytes = capacity, sample_bytes
        self.rows = []
        self.seen = self.eligible = self.oversize = self.used = 0

    def add(self, message):
        self.seen += 1
        size = message["bytes"] + message["candidates"] * 4 + 4096
        if size > self.sample_bytes:
            self.oversize += 1
            return
        self.eligible += 1
        index = self.rng.randrange(self.eligible)
        if len(self.rows) == self.capacity and index >= self.capacity:
            return
        row = example(message["input"], message["selected"])
        item = (row, message["actor"], message["step"], size_of(row))
        if len(self.rows) < self.capacity:
            self.rows.append(item)
            self.used += item[3]
        else:
            self.used += item[3] - self.rows[index][3]
            self.rows[index] = item

    def terminal(self, returns):
        for row, actor, step, _ in self.rows:
            row["value"] = torch.tensor(float(returns[str(actor)]))
            # MC Q 对每个实际分解选择监督；状态价值也按该前缀所属方学习。
            row["value_mask"] = torch.tensor(True)
            yield row


class SamplePool:
    def __init__(self, limit, reuse=2):
        self.limit, self.reuse = limit, reuse
        self.rows = deque()
        self.used = self.evicted = self.consumed = 0

    def add(self, row):
        size = size_of(row)
        if size > self.limit:
            self.evicted += 1
            return
        while self.rows and self.used + size > self.limit:
            _, old, _ = self.rows.popleft()
            self.used -= old
            self.evicted += 1
        self.rows.append((row, size, 0))
        self.used += size

    def batch(self, count, rng):
        # 每步随机抽样但不复制全池索引；次数到达上限即释放。
        result = []
        for _ in range(min(count, len(self.rows))):
            index = rng.randrange(len(self.rows))
            self.rows.rotate(-index)
            row, size, uses = self.rows.popleft()
            self.rows.rotate(index)
            result.append(row)
            if uses + 1 < self.reuse:
                self.rows.append((row, size, uses + 1))
            else:
                self.used -= size
                self.consumed += 1
        return result

    def snapshot(self):
        return {
            "bytes": self.used,
            "limit": self.limit,
            "samples": len(self.rows),
            "evicted": self.evicted,
            "consumed": self.consumed,
            "reuse": self.reuse,
        }
