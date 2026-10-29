"""完整命令级图搜索；自回归参数共享一次主干前向，前缀不消耗游戏深度。"""

from dataclasses import dataclass, field
import json
import math
import numpy as np


@dataclass
class Request:
    op: str
    context: str
    arrays: tuple


def probabilities(logits):
    values = np.exp(logits - np.max(logits))
    return values / values.sum()


@dataclass
class Prefix:
    front: dict
    options: np.ndarray
    prior: np.ndarray
    remaining: np.ndarray


@dataclass
class Edge:
    command: dict
    prior: float
    paths: list
    visits: int = 0
    total: float = 0.0


@dataclass
class Node:
    info: dict
    bundle: tuple
    value: float = 0.0
    version: int = -1
    prefixes: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)
    by_command: dict = field(default_factory=dict)
    visits: int = 0
    available: float = 1.0
    graph_nodes: int = 1
    transpositions: int = 0

    def initialize(self):
        answer, version = yield Request("encode", self.info["key"], self.bundle)
        self.value = float(answer[0])
        self.version = version

    def new_command(self, environment, rng, noise):
        prefix = 0
        trace = []
        logp = 0.0
        while True:
            if prefix not in self.prefixes:
                front, options = environment.decode(self.info["id"], prefix)
                if front["leaf"]:
                    command = front["command"]
                    available = 0.0
                    break
                if not len(options):
                    raise RuntimeError("合法解码前缀没有后继")
                if len(options) == 1:
                    prior = np.ones(1, np.float64)
                else:
                    logits, _ = yield Request("decode", self.info["key"], (options,))
                    prior = probabilities(logits).astype(np.float64)
                if noise and prefix == 0 and len(prior) > 1:
                    prior = 0.75 * prior + 0.25 * rng.dirichlet(
                        np.full(len(prior), 0.3)
                    )
                self.prefixes[prefix] = Prefix(
                    front, options, prior, np.ones(len(prior), np.float64)
                )
            branch = self.prefixes[prefix]
            weights = branch.prior * branch.remaining
            total = weights.sum()
            if total <= 0:
                return None
            choice = int(rng.choice(len(weights), p=weights / total))
            logp += math.log(max(float(branch.prior[choice]), 1e-300))
            trace.append((prefix, choice))
            prefix = branch.front["next"][choice]
        for parent, choice in reversed(trace):
            branch = self.prefixes[parent]
            branch.remaining[choice] = available
            available = float(np.dot(branch.prior, branch.remaining))
        self.available = available
        prior = math.exp(logp)
        key = json.dumps(command, sort_keys=True, separators=(",", ":"))
        if key in self.by_command:
            edge = self.by_command[key]
            edge.prior += prior
            edge.paths.append((prior, trace))
        else:
            edge = Edge(command, prior, [(prior, trace)])
            self.by_command[key] = edge
            self.edges.append(edge)
        return edge

    def supervision(self, policy):
        targets = {}
        for edge, mass in zip(self.edges, policy):
            for probability, trace in edge.paths:
                for prefix, choice in trace:
                    branch = self.prefixes[prefix]
                    if len(branch.options) == 1:
                        continue
                    targets.setdefault(
                        prefix, np.zeros(len(branch.options), np.float64)
                    )[choice] += float(mass) * probability / edge.prior
        return [
            (
                self.prefixes[p].options,
                (target / target.sum()).astype(np.float32),
                float(target.sum()),
            )
            for p, target in targets.items()
            if target.sum() > 0
        ]


def demonstration(environment, command):
    info, bundle = environment.encode()
    decisions = []
    for prefix, index in environment.call("trace", command=command):
        _, options = environment.decode(0, prefix)
        if len(options) > 1:
            target = np.zeros(len(options), np.float32)
            target[index] = 1
            decisions.append((options, target, 1.0))
    return info, bundle, decisions


def search(environment, rng, *, simulations, depth, noise=True):
    prefix = 0
    while True:
        front, options = environment.decode(0, prefix)
        if front["leaf"]:
            root = Node(
                environment.call("info"),
                (),
                edges=[Edge(front["command"], 1.0, [])],
                graph_nodes=0,
            )
            return root, np.ones(1, np.float32), -1
        if len(options) != 1:
            break
        prefix = front["next"][0]
    info, bundle = environment.encode()
    root = Node(info, bundle)
    yield from root.initialize()
    graph = {info["key"]: root}
    if simulations == 0:
        yield from root.new_command(environment, rng, noise)
        return root, np.ones(1, np.float32), root.version
    for _ in range(simulations):
        node = root
        path = []
        seen = {node.info["key"]}
        for _level in range(depth):
            # 渐进扩宽没有固定命令集合；新命令来自网络条件分布。
            limit = max(1, math.ceil(2 * math.sqrt(node.visits + 1)))
            edge = None
            if node.available > 0 and len(node.edges) < limit:
                edge = yield from node.new_command(
                    environment, rng, noise and node is root
                )
            if edge is None:
                if not node.edges:
                    raise RuntimeError("非终局没有合法完整命令")
                edge = max(
                    node.edges,
                    key=lambda e: (
                        e.total / max(1, e.visits)
                        + 1.5 * e.prior * math.sqrt(1 + node.visits) / (1 + e.visits)
                    ),
                )
            result = environment.call(
                "sample-command",
                id=node.info["id"],
                command=edge.command,
                seed=int(rng.integers(1, 2**32, dtype=np.uint64)),
            )
            if result.get("rejected"):
                raise RuntimeError(f"合法解码命令被拒绝: {result}")
            path.append((node, edge))
            actor = result["actor"]
            if result["winner"] is not None:
                value = (
                    0.0
                    if result["winner"] == "draw"
                    else (1.0 if result["winner"] == actor else -1.0)
                )
                break
            key = result["key"]
            if key not in graph:
                leaf_info, leaf_bundle = environment.encode(result["id"])
                child = Node(leaf_info, leaf_bundle)
                graph[key] = child
                yield from child.initialize()
                value = child.value
                break
            child = graph[key]
            root.transpositions += 1
            if key in seen:
                actor, value = child.info["actor"], child.value
                break
            seen.add(key)
            node = child
        else:
            actor, value = node.info["actor"], node.value
        for parent, edge in reversed(path):
            if parent.info["actor"] != actor:
                value = -value
            actor = parent.info["actor"]
            edge.visits += 1
            edge.total += value
            parent.visits += 1
    root.graph_nodes = len(graph)
    visits = np.asarray([e.visits for e in root.edges], np.float64)
    return root, (visits / visits.sum()).astype(np.float32), root.version
