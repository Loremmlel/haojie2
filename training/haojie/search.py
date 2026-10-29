"""完整命令级随机 PUCT；生成器把叶子推理交给共享 GPU 服务。"""

from dataclasses import dataclass, field

import numpy as np


def probabilities(logits):
    values = np.exp(logits - np.max(logits))
    return values / values.sum()


@dataclass
class Node:
    info: dict
    states: np.ndarray
    actions: np.ndarray
    prior: np.ndarray | None = None
    value: float = 0.0
    visits: np.ndarray | None = None
    total: np.ndarray | None = None
    children: dict = field(default_factory=dict)
    rejected: set = field(default_factory=set)

    def install(self, answer):
        logits, self.value, _aux, version = answer
        self.prior = probabilities(logits)
        self.visits = np.zeros(len(logits), np.int64)
        self.total = np.zeros(len(logits), np.float64)
        return version


def search(environment, rng, *, simulations, candidates, depth, noise=True, include=None):
    seed = lambda: int(rng.integers(1, 2**32, dtype=np.uint64))
    info, states, actions = environment.prepare(0, seed(), candidates, include)
    root = Node(info, states, actions)
    if len(actions) == 1:
        return root, np.ones(1, np.float32), -1
    version = root.install((yield (states, actions)))
    if noise:
        root.prior = .75 * root.prior + .25 * rng.dirichlet(np.full(len(actions), .3))
    if simulations == 0:
        return root, root.prior.astype(np.float32), version
    for _ in range(simulations):
        node, path = root, []
        for level in range(depth):
            q = np.divide(node.total, node.visits, out=np.zeros_like(node.total), where=node.visits > 0)
            score = q + 1.5 * node.prior * np.sqrt(1 + node.visits.sum()) / (1 + node.visits)
            for rejected in node.rejected:
                score[rejected] = -np.inf
            action = int(np.argmax(score))
            result = environment.call("sample", id=node.info["id"], action=action, seed=seed())
            if result.get("rejected"):
                node.prior[action] = 0
                node.rejected.add(action)
                if node.prior.sum() == 0:
                    raise RuntimeError("搜索节点所有命令均被权威规则拒绝")
                node.prior /= node.prior.sum()
                continue
            path.append((node, action))
            actor = result["actor"]
            if result["winner"] is not None:
                value = 0.0 if result["winner"] == "draw" else (1.0 if result["winner"] == actor else -1.0)
                break
            key = action, result["key"]
            child = node.children.get(key)
            if child is None:
                leaf_info, leaf_states, leaf_actions = environment.prepare(result["id"], seed(), candidates)
                child = Node(leaf_info, leaf_states, leaf_actions)
                child.install((yield (leaf_states, leaf_actions)))
                node.children[key] = child
                value = child.value
                break
            node = child
        else:
            actor, value = node.info["actor"], node.value
        for parent, action in reversed(path):
            if parent.info["actor"] != actor:
                value = -value
            actor = parent.info["actor"]
            parent.visits[action] += 1
            parent.total[action] += value
    visits = root.visits.astype(np.float64)
    target = visits / visits.sum() if visits.sum() else root.prior
    return root, target.astype(np.float32), version
