"""按学习席位连接真实随机选择；未获得后继的转换留在 pending。"""

from collections import deque


class Trajectories:
    def __init__(self, length=64):
        self.length = length
        self.pending = {}
        self.partial = {}
        self.ready = deque()
        self.next_id = 0
        self.acknowledged = -1

    def selected(self, game, node, selected, logp, version):
        if game in self.pending:
            raise RuntimeError("缺少同席位后继，不能覆盖转换")
        if not node["mask"][selected]:
            raise ValueError("选择不属于行为支持集")
        self.pending[game] = dict(
            node=node, selected=selected, logp=logp, version=version, reward=0.0, discount=1.0
        )

    def successor(self, game, node=None, reward=None):
        if game in self.pending:
            step = self.pending.pop(game)
            if reward is not None:
                step.update(reward=reward, discount=0.0)
            self.partial.setdefault(game, []).append(step)
        steps = self.partial.get(game, [])
        if steps and (len(steps) >= self.length or reward is not None):
            self.ready.append(dict(id=self.next_id, steps=steps, bootstrap=node))
            self.next_id += 1
            self.partial.pop(game)

    def commit(self, ids):
        """只在优化器成功提交之后确认，保存边界不能落在两者之间。"""
        for expected in ids:
            block = self.ready.popleft()
            if block["id"] != expected or expected <= self.acknowledged:
                raise RuntimeError("训练块确认顺序错误")
            self.acknowledged = expected

    def root_ids(self):
        ids = {s["node"]["root"] for s in self.pending.values()}
        for steps in self.partial.values():
            ids.update(s["node"]["root"] for s in steps)
        for block in self.ready:
            ids.update(s["node"]["root"] for s in block["steps"])
            if block["bootstrap"] is not None:
                ids.add(block["bootstrap"]["root"])
        return ids
