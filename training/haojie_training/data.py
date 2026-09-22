"""预编码公开张量的格式边界及合成性能数据；这里不推导游戏数值或真实随机状态。"""

from pathlib import Path

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .model import ModelConfig

FORMAT = "haojie-training-tensors-v1"
FLOAT_KEYS = {"entities", "globals", "candidates", "policy", "value"}
MASK_KEYS = {"entity_mask", "candidate_mask", "value_mask"}
INDEX_KEYS = {"kinds", "sources", "targets"}
ENTITY_KEYS = {"entities", "kinds", "entity_mask"}
ACTION_KEYS = {"candidates", "sources", "targets", "candidate_mask", "policy"}
INPUT_KEYS = (FLOAT_KEYS | MASK_KEYS | INDEX_KEYS) - {"policy", "value", "value_mask"}


def collate_examples(examples: list[dict[str, Tensor]], config: ModelConfig) -> dict[str, Tensor]:
    """按本组最大长度填充，无截断；来源/目标保持样本内索引，填充指针使用全局token。"""
    if not examples:
        raise ValueError("没有可训练样本")
    batch = {}
    for key in FLOAT_KEYS | MASK_KEYS | INDEX_KEYS:
        values = [example[key] for example in examples]
        batch[key] = (
            pad_sequence(
                values, batch_first=True, padding_value=-1 if key in {"sources", "targets"} else 0
            )
            if key in ENTITY_KEYS | ACTION_KEYS
            else torch.stack(values)
        )
    validate_batch(batch, config)
    return batch


def select_batch(dataset: dict[str, Tensor], indices: Tensor) -> dict[str, Tensor]:
    """取样后仅移除整批无效的尾部填充，较大局面不会增加其他批次的注意力开销。"""
    batch = {key: value[indices] for key, value in dataset.items()}
    for mask, keys in [("entity_mask", ENTITY_KEYS), ("candidate_mask", ACTION_KEYS)]:
        used = batch[mask].any(dim=0).nonzero()
        length = int(used[-1, 0]) + 1 if used.numel() else 1
        for key in keys:
            batch[key] = batch[key][:, :length]
    return batch


def validate_inputs(batch: dict[str, Tensor], config: ModelConfig) -> None:
    """训练和推理共用CPU输入边界，拒绝无效指针、全空候选及隐藏状态字段。"""
    if set(batch) != INPUT_KEYS:
        raise ValueError("输入必须是固定张量白名单；不能传入Observation、Session或seed/rng")
    for key, tensor in batch.items():
        if not isinstance(tensor, Tensor) or tensor.device.type != "cpu":
            raise ValueError(f"{key}必须是CPU张量")
        expected = (
            torch.float32 if key in FLOAT_KEYS else torch.bool if key in MASK_KEYS else torch.long
        )
        if tensor.dtype != expected:
            raise ValueError(f"{key}需要{expected}")
        if key in FLOAT_KEYS and not torch.isfinite(tensor).all():
            raise ValueError(f"{key}含非有限数值")
    if batch["entities"].ndim != 3 or batch["candidates"].ndim != 3:
        raise ValueError("实体和候选必须是[B,N,F]三维张量")
    size, entities, _ = batch["entities"].shape
    actions = batch["candidates"].shape[1]
    if min(size, entities, actions) < 1:
        raise ValueError("批量、实体填充长度和候选长度至少为1；空棋盘用无效填充实体")
    shapes = {
        "entities": (size, entities, config.entity_features),
        "kinds": (size, entities),
        "entity_mask": (size, entities),
        "globals": (size, config.global_features),
        "candidates": (size, actions, config.action_features),
        "sources": (size, actions),
        "targets": (size, actions),
        "candidate_mask": (size, actions),
    }
    if any(tuple(batch[k].shape) != shape for k, shape in shapes.items()):
        raise ValueError("张量形状与模型输入约定不一致")
    if (batch["kinds"] < 0).any() or (batch["kinds"] >= config.kind_count).any():
        raise ValueError("实体类别超出词表，禁止静默截断或哈希替换")
    if not batch["candidate_mask"].any(dim=1).all():
        raise ValueError("每个样本至少要有一个可训练候选")
    valid_entities = torch.cat((torch.ones(size, 1, dtype=torch.bool), batch["entity_mask"]), 1)
    for key in ("sources", "targets"):
        indices = batch[key]
        if (indices < -1).any() or (indices >= entities).any():
            raise ValueError(f"{key}实体指针越界")
        if (~valid_entities.gather(1, indices + 1) & batch["candidate_mask"]).any():
            raise ValueError(f"{key}指向填充实体")


def validate_batch(batch: dict[str, Tensor], config: ModelConfig) -> None:
    """训练在相同输入约束上另外校验标签和终局遮罩。"""
    if set(batch) != FLOAT_KEYS | MASK_KEYS | INDEX_KEYS:
        raise ValueError("输入必须是固定张量白名单；不能传入Observation、Session或seed/rng")
    validate_inputs({key: batch[key] for key in INPUT_KEYS}, config)
    size, actions = batch["candidate_mask"].shape
    for key, shape, dtype in [
        ("policy", (size, actions), torch.float32),
        ("value", (size,), torch.float32),
        ("value_mask", (size,), torch.bool),
    ]:
        tensor = batch[key]
        if (
            not isinstance(tensor, Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != dtype
            or tuple(tensor.shape) != shape
        ):
            raise ValueError(f"{key}标签类型/形状不匹配")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{key}含非有限数值")
    policy = batch["policy"]
    if (policy < 0).any() or (policy[~batch["candidate_mask"]] != 0).any():
        raise ValueError("策略标签不能向无效候选分配概率")
    if not torch.allclose(policy.sum(dim=1), torch.ones(size), atol=1e-5, rtol=0):
        raise ValueError("策略目标必须归一化")
    if (batch["value"].abs() > 1).any():
        raise ValueError("价值目标必须在[-1,1]内，且采用观察所属方视角")


def synthetic_batch(
    config: ModelConfig, size: int = 8, entities: int = 64, actions: int = 64, seed: int = 7
) -> dict[str, Tensor]:
    """相同种子在CPU生成相同张量，仅用于速度/训练闭环，不是游戏训练样本。"""
    if min(size, entities, actions) < 1:
        raise ValueError("批量、实体和候选数必须为正")
    rng = torch.Generator().manual_seed(seed)
    entity_counts = torch.randint(max(1, entities // 2), entities + 1, (size,), generator=rng)
    action_counts = torch.randint(max(1, actions // 2), actions + 1, (size,), generator=rng)
    entity_mask = torch.arange(entities)[None, :] < entity_counts[:, None]
    candidate_mask = torch.arange(actions)[None, :] < action_counts[:, None]

    # 全局token作为基地/空参数的占位，指针只落到本样本的有效实体。
    def indices():
        return (torch.rand(size, actions, generator=rng) * (entity_counts[:, None] + 1)).long() - 1

    labels = (torch.rand(size, generator=rng) * action_counts).long()
    batch = {
        "entities": torch.randn(size, entities, config.entity_features, generator=rng) * 0.2,
        "kinds": torch.randint(1, config.kind_count, (size, entities), generator=rng),
        "entity_mask": entity_mask,
        "globals": torch.randn(size, config.global_features, generator=rng) * 0.2,
        "candidates": torch.randn(size, actions, config.action_features, generator=rng) * 0.2,
        "sources": indices(),
        "targets": indices(),
        "candidate_mask": candidate_mask,
        "policy": torch.nn.functional.one_hot(labels, actions).float(),
        "value": torch.randint(-1, 2, (size,), generator=rng).float(),
        "value_mask": torch.rand(size, generator=rng) > 0.25,
    }
    validate_batch(batch, config)
    return batch


def load_dataset(path: Path, config: ModelConfig) -> tuple[dict[str, Tensor], dict]:
    """只加载weights_only张量文件；元数据必须注明规则、编码与合成标记。"""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError("不支持的数据集格式；原始教师JSONL须先完成公开特征编码")
    metadata = payload.get("metadata", {})
    if (
        not isinstance(metadata, dict)
        or not isinstance(metadata.get("ruleset"), str)
        or not metadata["ruleset"]
        or not isinstance(metadata.get("encoding"), str)
        or not metadata["encoding"]
        or type(metadata.get("synthetic")) is not bool
    ):
        raise ValueError("数据集必须标明ruleset、encoding和synthetic")
    batch = payload["tensors"]
    validate_batch(batch, config)
    # ponytail: 首版预编码数据整体驻留CPU内存；大数据集再改为分片或mmap读取。
    return batch, metadata
