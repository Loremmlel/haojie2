"""公开v1张量的结构消融；额外语义属于独立研究网络，不改变引擎或旧编码。"""

import torch
from torch import nn
from torch.nn import functional as F

from scripts.training.search.recovery.spatial import SpatialPolicyNet

FEATURES = ("absolute", "relative", "relations", "fields", "candidate")
ARMS = ("spatial", *FEATURES, "combined", "capacity")


def geometry(batch):
    """只反解基地/单位/地标坐标；填充与非空间记录必须在指数反变换前遮罩。"""
    raw = batch["entities"]
    roles = (raw[..., 0] * 32).round().long()
    placed = batch["entity_mask"] & (roles >= 1) & (roles <= 3)
    xy = torch.where(placed[..., None], raw[..., 8:10], 0)
    xy = (xy.sign() * torch.expm1(xy.abs() * 8)).round().long()
    return roles, placed, xy


def relation_edges(batch):
    """按引用相等性生成有向附属/来源/目标边与同组边；编号大小不成为关系强度。"""
    raw = torch.where(batch["entity_mask"][..., None], batch["entities"], 0)
    refs = (raw[..., 2:7] * 256).round().long()
    identity = refs[..., 0]
    edges = []
    for column in (1, 2, 3):
        pointer = refs[..., column]
        forward = (pointer[:, :, None] == identity[:, None, :]) & (pointer[:, :, None] > 0)
        edges.extend((forward, forward.transpose(1, 2)))
    groups = refs[..., 4]
    edges.append((groups[:, :, None] == groups[:, None, :]) & (groups[:, :, None] > 0))
    _, placed, xy = geometry(batch)
    edges.append(
        (xy[:, :, None] == xy[:, None, :]).all(-1) & placed[:, :, None] & placed[:, None, :]
    )
    return torch.stack(edges, -1).float()


def expanded_fields(batch):
    """展开存在位、实体角色及模式类别；血量比例只由公开数值计算，不推导新规则。"""
    raw = torch.where(batch["entity_mask"][..., None], batch["entities"], 0)
    roles = (raw[..., 0] * 32).round().long().clamp(0, 32)
    packed = (raw[..., 60:64] * 8191).round().long()
    bits = torch.arange(13, device=raw.device)
    presence = ((packed[..., None] >> bits) & 1).flatten(-2).float()
    effect = torch.where(roles == 6, raw[..., 8], 0)
    effect = torch.expm1(effect * 8).round().long().clamp(0, 10)
    on_unit = (roles == 2) | (roles == 3)
    mode = torch.where(on_unit, raw[..., 17], 0)
    mode = torch.expm1(mode * 8).round().long().clamp(0, 14)
    hp_raw = torch.where(on_unit[..., None], raw[..., 10:12], 0)
    hp = hp_raw.sign() * torch.expm1(hp_raw.abs() * 8)
    ratio = hp[..., :1] / hp[..., 1:].abs().clamp_min(1)
    return torch.cat(
        (presence, F.one_hot(roles, 33), F.one_hot(effect, 11), F.one_hot(mode, 15), ratio), -1
    ).float()


class ImprovementNet(SpatialPolicyNet):
    """在已验证的空间小模型上逐项加入位置、关系、字段或候选注意力。

    所有额外输入由原有八项公开张量派生，不读取标签、教师分数或游戏随机数。
    默认spatial与上一轮空间网络相同。输入不可变，候选顺序等变，填充实体不参与聚合。
    关系采用Transformer加性图偏置，保留全局通信；不宣称另实现稀疏GAT。
    """

    def __init__(self, config, arm="combined", features=None):
        super().__init__(config)
        if arm not in ARMS:
            raise ValueError("未知消融臂")
        self.features = tuple(
            features
            if features is not None
            else FEATURES
            if arm == "combined"
            else (arm,)
            if arm in FEATURES
            else ()
        )
        if set(self.features) - set(FEATURES):
            raise ValueError("未知结构开关")
        self.arm = arm
        if "absolute" in self.features:
            self.x_embedding = nn.Embedding(11, config.width)
            self.y_embedding = nn.Embedding(15, config.width)
        if "relative" in self.features:
            self.dx = nn.Embedding(21, config.heads)
            self.dy = nn.Embedding(29, config.heads)
            nn.init.zeros_(self.dx.weight)
            nn.init.zeros_(self.dy.weight)
        if "relations" in self.features:
            self.edge_bias = nn.Linear(8, config.heads, bias=False)
            nn.init.zeros_(self.edge_bias.weight)
        if "fields" in self.features:
            self.field_projection = nn.Sequential(nn.Linear(112, config.width), nn.GELU())
        if "candidate" in self.features:
            self.query = nn.Linear(config.width * 2, config.width)
            self.key = nn.Linear(config.width, config.width)
            self.content = nn.Linear(config.width, config.width)
            self.readout = nn.Linear(config.width, config.width)

    def entity_context(self, batch, entities):
        if "relations" in self.features:
            # 关系臂只利用编号相等性，移除原投影中编号大小这一偶然信号。
            entities = entities - F.linear(
                batch["entities"][..., 2:7], self.entity_projection.weight[:, 2:7]
            )
        if "absolute" in self.features:
            _, placed, xy = geometry(batch)
            position = self.x_embedding(xy[..., 0].clamp(0, 10)) + self.y_embedding(
                xy[..., 1].clamp(0, 14)
            )
            entities = entities + position * placed[..., None]
        if "fields" in self.features:
            entities = entities + self.field_projection(expanded_fields(batch))
        return entities

    def attention_mask(self, batch, valid):
        if not {"relative", "relations"} & set(self.features):
            return super().attention_mask(batch, valid)
        _, placed, xy = geometry(batch)
        size = xy.shape[1]
        bias = batch["entities"].new_zeros(xy.shape[0], size, size, self.config.heads)
        if "relative" in self.features:
            delta = xy[:, :, None] - xy[:, None, :]
            spatial = self.dx(delta[..., 0].clamp(-10, 10) + 10) + self.dy(
                delta[..., 1].clamp(-14, 14) + 14
            )
            bias = bias + spatial * (placed[:, :, None] & placed[:, None, :])[..., None]
        if "relations" in self.features:
            bias = bias + self.edge_bias(relation_edges(batch))
        bias = F.pad(bias.permute(0, 3, 1, 2), (1, 0, 1, 0))
        return bias.masked_fill(~valid[:, None, None, :], -torch.inf)

    def target_context(self, batch, entities, targets):
        targets = super().target_context(batch, entities, targets)
        xy = (batch["candidates"][..., 36:38] * entities.new_tensor([9, 13])).round().long()
        placed = (xy > 0).all(-1) & (batch["targets"] == -1)
        if "absolute" in self.features:
            targets = (
                targets
                + (
                    self.x_embedding(xy[..., 0].clamp(0, 10))
                    + self.y_embedding(xy[..., 1].clamp(0, 14))
                )
                * placed[..., None]
            )
        if "candidate" in self.features:
            width, heads = self.config.width, self.config.heads
            sources = entities.gather(1, (batch["sources"] + 1)[..., None].expand(-1, -1, width))
            actions = self.action_projection(batch["candidates"].float())

            def split(value):
                return value.reshape(
                    value.shape[0], value.shape[1], heads, width // heads
                ).transpose(1, 2)

            query = split(self.query(torch.cat((actions, sources), -1)))
            key, value = split(self.key(entities)), split(self.content(entities))
            valid = torch.cat(
                (torch.ones_like(batch["entity_mask"][:, :1]), batch["entity_mask"]), 1
            )
            read = F.scaled_dot_product_attention(
                query, key, value, attn_mask=valid[:, None, None, :], dropout_p=0
            )
            targets = targets + self.readout(
                read.transpose(1, 2).reshape(query.shape[0], -1, width)
            )
        return targets
