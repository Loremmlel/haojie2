"""实体Transformer骨架；只接收已编码公开张量，不负责模拟规则或生成合法命令。"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

NETWORK_VERSION = "entity-transformer-v3"
HISTORY_LAYERS = 2
RECENT_JOURNAL = 16


def active_tokens(batch: dict[str, Tensor]) -> Tensor:
    """历史先完整交互两层；近期日志及有效候选引用继续接受每层更新。

    亡者角色使用编码第零列的精确值，日志使用独立类别；字段和原实体索引保持不变。
    全局始终活跃，遮罩实体和候选不提升；集合取自完整输入，不能随 MC 已选候选改变。
    """
    global_mask = torch.ones_like(batch["entity_mask"][:, :1])
    valid = torch.cat((global_mask, batch["entity_mask"]), dim=1)
    journal = batch["kinds"] == 250
    # 日志按发生顺序追加；只数有效条目，保留最近搜索分支的完整深层交互。
    older_journal = (journal & batch["entity_mask"]).flip(1).cumsum(1).flip(1) > RECENT_JOURNAL
    history = (journal & older_journal) | (~journal & (batch["entities"][..., 0] == 5 / 32))
    active = torch.cat((global_mask, batch["entity_mask"] & ~history), dim=1)
    for key in ("sources", "targets"):
        pointers = torch.where(batch["candidate_mask"], batch[key] + 1, 0)
        active = active.scatter(1, pointers, True)
    return active & valid


def history_queries(batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    """按原位置排列活跃查询，各行补齐索引仍唯一，避免重复 scatter 放大梯度。"""
    active = active_tokens(batch)
    positions = torch.arange(active.shape[1], device=active.device)
    order = (positions + (~active).long() * active.shape[1]).argsort(dim=1)
    indices = order[:, positions < active.sum(dim=1).max()]
    return indices, active.gather(1, indices)


@dataclass(frozen=True)
class ModelConfig:
    width: int = 384
    layers: int = 6
    heads: int = 6
    ffn: int = 1536
    entity_features: int = 64
    global_features: int = 32
    action_features: int = 64
    kind_count: int = 256

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in vars(self).values()):
            raise ValueError("模型维度必须是正整数")
        if self.width % self.heads:
            raise ValueError("width必须能被heads整除")

    @classmethod
    def tiny(cls):
        return cls(width=64, layers=2, heads=4, ffn=256)


class EncoderBlock(nn.Module):
    """Pre-LN自注意力；True表示可被关注，不让填充实体成为其他实体的上下文。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.attention_norm = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.output = nn.Linear(config.width, config.width)
        self.ffn_norm = nn.LayerNorm(config.width)
        self.ffn = nn.Sequential(
            nn.Linear(config.width, config.ffn),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.ffn, config.width),
        )

    def forward(self, x: Tensor, allowed: Tensor, readout: Tensor | None = None) -> Tensor:
        batch, tokens, width = x.shape
        qkv = self.qkv(self.attention_norm(x))
        q, k, v = qkv.reshape(batch, tokens, 3, self.heads, width // self.heads).permute(
            2, 0, 3, 1, 4
        )
        if readout is not None:
            # 仅计算所需查询；中层更新活跃实体，末层读取候选，所有实体仍参与 K/V。
            q = q.gather(
                2, readout[:, None, :, None].expand(-1, self.heads, -1, width // self.heads)
            )
            x = x.gather(1, readout[..., None].expand(-1, -1, width))
            tokens = readout.shape[1]
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
        x = x + self.output(attended.transpose(1, 2).reshape(batch, tokens, width))
        return x + self.ffn(self.ffn_norm(x))


class PolicyValueNet(nn.Module):
    """
    共享实体主干，给外部提供的候选评分，并预测观察所属方的终局收益。
    输入由data.validate_batch在CPU边界校验；前向不修改输入、不复制游戏规则。
    候选src/target是实体索引，-1指全局token；游戏编码来自共享TypeScript适配层。
    不设实体数量上限、不截断实体，不把候选评分误称为完整动作生成器。
    """

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        self.entity_projection = nn.Linear(config.entity_features, config.width)
        self.kind_embedding = nn.Embedding(config.kind_count, config.width, padding_idx=0)
        self.global_projection = nn.Linear(config.global_features, config.width)
        self.blocks = nn.ModuleList(EncoderBlock(config) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.width)
        self.action_projection = nn.Sequential(
            nn.Linear(config.action_features, config.width),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(config.width),
        )
        self.policy = nn.Sequential(
            nn.Linear(config.width * 4, config.width),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.width, config.width),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.width, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(config.width, config.width),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.width, 1),
            nn.Tanh(),
        )

    def encode(self, batch: dict[str, Tensor], readout: Tensor | None = None) -> Tensor:
        """前两层完整交互，中间层仅更新活跃查询；完整历史继续作为每层 K/V。

        历史表示只在本次前向内复用，不 detach，深层读取仍训练其编码。末层沿用
        完整或稀疏读出；研究子类的关系遮罩和实体扩展保留原有完整路径。
        """
        entities = self.entity_projection(batch["entities"]) + self.kind_embedding(batch["kinds"])
        entities = self.entity_context(batch, entities)
        global_token = self.global_projection(batch["globals"]).unsqueeze(1)
        x = torch.cat((global_token, entities), dim=1)
        valid = torch.cat(
            (torch.ones_like(batch["entity_mask"][:, :1]), batch["entity_mask"]), dim=1
        )
        allowed = self.attention_mask(batch, valid)
        queries = (
            history_queries(batch)
            if type(self) is PolicyValueNet and len(self.blocks) > HISTORY_LAYERS + 1
            else None
        )
        for index, block in enumerate(self.blocks):
            last = index == len(self.blocks) - 1
            if queries is not None and HISTORY_LAYERS <= index and not last:
                positions, query_valid = queries
                gather = positions[..., None].expand(-1, -1, x.shape[-1])
                updated = block(x, allowed, positions)
                updated = torch.where(query_valid[..., None], updated, x.gather(1, gather))
                x = x.scatter(1, gather, updated)
            else:
                x = block(x, allowed, readout if last else None)
        return self.norm(x)

    def score_policy(self, batch: dict[str, Tensor], x: Tensor) -> Tensor:
        action_count = batch["candidates"].shape[1]
        context = x[:, :1].expand(-1, action_count, -1)
        sources = x.gather(1, (batch["sources"] + 1).unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        targets = x.gather(1, (batch["targets"] + 1).unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        targets = self.target_context(batch, x, targets)
        # 候选投影与评分矩阵共用所选 AMP；仅标量输出、遮罩和收益回归保留 FP32。
        actions = self.action_projection(batch["candidates"])
        action_context = torch.cat((context, sources, targets, actions), dim=-1)
        logits = self.policy(action_context).squeeze(-1).float()
        logits = logits.masked_fill(~batch["candidate_mask"], -1e9)
        return logits

    def policy_logits(self, batch: dict[str, Tensor]) -> Tensor:
        """原生采样只消费策略，不计算或传回价值头。"""
        return self.score_policy(batch, self.encode(batch))

    def policy_readout(self, batch, positions, sources, targets):
        """原网络的稀疏末层读取；位置表由 CPU 执行器生成，不缓存局面或模型权重。

        此优化仅适用于原骨架。研究子类可能依赖其他实体的末层输出，仍走完整前向。
        候选顺序、有效遮罩和公开输入保持不变；只重映射末层表示的读取下标。
        """
        if type(self) is not PolicyValueNet:
            return self.policy_logits(batch)
        x = self.encode(batch, positions)
        return self.score_policy({**batch, "sources": sources - 1, "targets": targets - 1}, x)

    def selected_value(self, batch, selected):
        """MC-Q 仅监督已选候选；一次主干得到全局、来源和目标，跳过无梯度的其他评分。

        各层 K/V 保留所有实体，活跃集合取自完整候选。重复指针的梯度经 gather 自动相加；损失权重、
        价值头和检查点参数完全共用原实现。研究模型保留其完整上下文扩展。
        """
        if type(self) is not PolicyValueNet:
            logits, value = self(batch)
            return logits.gather(1, selected[:, None]).squeeze(1), value
        sources = batch["sources"].gather(1, selected[:, None]) + 1
        targets = batch["targets"].gather(1, selected[:, None]) + 1
        positions = torch.cat((torch.zeros_like(sources), sources, targets), dim=1)
        x = self.encode(batch, positions)
        index = selected[:, None, None].expand(-1, 1, self.config.action_features)
        chosen = {
            **batch,
            "candidates": batch["candidates"].gather(1, index),
            "sources": torch.zeros_like(sources),
            "targets": torch.ones_like(targets),
            "candidate_mask": batch["candidate_mask"].gather(1, selected[:, None]),
        }
        return self.score_policy(chosen, x).squeeze(1), self.value(x[:, 0]).squeeze(-1).float()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        x = self.encode(batch)
        return self.score_policy(batch, x), self.value(x[:, 0]).squeeze(-1).float()

    def entity_context(self, batch: dict[str, Tensor], entities: Tensor) -> Tensor:
        """默认保持原实体投影；研究模型可展开已有公开字段，不更改旧检查点参数。"""
        return entities

    def attention_mask(self, batch: dict[str, Tensor], valid: Tensor) -> Tensor:
        """默认仅遮罩填充键；研究模型的关系偏置仍须保留同一填充边界。"""
        return valid[:, None, None, :]

    def target_context(self, batch: dict[str, Tensor], entities: Tensor, targets: Tensor) -> Tensor:
        """默认使用显式目标实体；实验可在此补充候选位置上下文，不复制整个前向实现。"""
        return targets


def policy_value_loss(
    output: tuple[Tensor, Tensor], batch: dict[str, Tensor], value_weight: float = 1.0
) -> Tensor:
    """策略支持软访问分布/one-hot教师标签；截断样本的价值mask为False，不充当平局。"""
    logits, value = output
    policy_loss = -(batch["policy"] * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    known = batch["value_mask"].float()
    value_loss = ((value - batch["value"]).square() * known).sum() / known.sum().clamp_min(1)
    return policy_loss + value_weight * value_loss
