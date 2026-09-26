"""对照试验的数据组合与采样；不改原分片，整族身份和软策略标签随样本保留。"""

import hashlib
from collections import defaultdict
from pathlib import Path

import torch
from haojie_training.data import collate_examples, load_dataset, sample_indices, select_batch
from haojie_training.evaluate import validate_split


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


class TrainingData:
    """只读多个二进制来源；可按自然样本量混合或按预先固定比例抽样。

    分层模式先均匀取种子族，再取命令/参数阶段与记录。
    仅族均衡保留族内自然频率，用于隔离按标签抽样改变命令先验的影响。
    这是改变训练分布的显式实验开关，验证保持自然分布；不按胜负挑选。
    """

    def __init__(
        self, paths, validation, config, mixture=None, balanced=False, family_balanced=False
    ):
        if balanced and family_balanced:
            raise ValueError("两种均衡方式互斥")
        self.config, self.balanced = config, balanced or family_balanced
        self.datasets = [load_dataset(path, config) for path in paths]
        self.paths = paths
        self.offsets, self.groups, self.buckets = [], set(), []
        offset = 0
        for data, meta in self.datasets:
            validate_split(meta, validation)
            if (
                meta.get("purpose") not in (None, "training")
                or meta.get("policy_source") == "value-only-behavior"
            ):
                raise ValueError("校准/留出集及纯行为价值流不能进入策略训练")
            self.offsets.append(offset)
            offset += len(meta["records"])
            self.groups.update(meta["groups"])
            grouped = defaultdict(lambda: defaultdict(list))
            for i, record in enumerate(meta["records"]):
                grouped[record["group"]][(record["command"], record["stage"])].append(i)
            self.buckets.append(
                [
                    [[i for bucket in v.values() for i in bucket]]
                    if family_balanced
                    else list(v.values())
                    for _, v in sorted(grouped.items())
                ]
            )
        self.count = offset
        self.weights = torch.tensor(
            mixture if mixture is not None else [len(m["records"]) for _, m in self.datasets],
            dtype=torch.float64,
        )
        if (
            len(self.weights) != len(paths)
            or not torch.isfinite(self.weights).all()
            or not (self.weights > 0).all()
        ):
            raise ValueError("混合权重必须与来源一一对应且为正")
        self.weights /= self.weights.sum()

    def sample(self, size, rng):
        choices = torch.multinomial(self.weights, size, replacement=True, generator=rng)
        examples, identities = [None] * size, [None] * size
        for source, (data, _) in enumerate(self.datasets):
            positions = (choices == source).nonzero().flatten().tolist()
            if not positions:
                continue
            if self.balanced:
                indices = []

                def pick(items):
                    return items[int(torch.randint(len(items), (1,), generator=rng))]

                for _ in positions:
                    indices.append(pick(pick(pick(self.buckets[source]))))
                ids = torch.tensor(indices)
            else:
                ids = sample_indices(data, len(positions), rng, 128)
            batch = select_batch(data, ids)
            for row, position in enumerate(positions):
                examples[position] = {key: value[row] for key, value in batch.items()}
                identities[position] = int(ids[row]) + self.offsets[source]
        return collate_examples(examples, self.config), torch.tensor(identities)
