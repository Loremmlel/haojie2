"""实体Transformer骨架；只接收已编码公开张量，不负责模拟规则或生成合法命令。"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

NETWORK_VERSION = "entity-transformer-v2"


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

    def forward(self, x: Tensor, allowed: Tensor) -> Tensor:
        batch, tokens, width = x.shape
        qkv = self.qkv(self.attention_norm(x))
        q, k, v = qkv.reshape(batch, tokens, 3, self.heads, width // self.heads).permute(
            2, 0, 3, 1, 4
        )
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

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        entities = self.entity_projection(batch["entities"]) + self.kind_embedding(batch["kinds"])
        global_token = self.global_projection(batch["globals"]).unsqueeze(1)
        x = torch.cat((global_token, entities), dim=1)
        valid = torch.cat(
            (torch.ones_like(batch["entity_mask"][:, :1]), batch["entity_mask"]), dim=1
        )
        allowed = valid[:, None, None, :]
        for block in self.blocks:
            x = block(x, allowed)
        x = self.norm(x)
        action_count = batch["candidates"].shape[1]
        context = x[:, :1].expand(-1, action_count, -1)
        sources = x.gather(1, (batch["sources"] + 1).unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        targets = x.gather(1, (batch["targets"] + 1).unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        # 候选先独立提取非线性特征，避免坐标差异被共享上下文淹没；评分保持FP32，
        # 防止BF16在较大logit附近把相邻落点量化成相同分数。实体主干仍使用所选AMP。
        with torch.autocast(x.device.type, enabled=False):
            actions = self.action_projection(batch["candidates"].float())
            action_context = torch.cat(
                (context.float(), sources.float(), targets.float(), actions), dim=-1
            )
            logits = self.policy(action_context).squeeze(-1)
        logits = logits.masked_fill(~batch["candidate_mask"], -1e9)
        return logits, self.value(x[:, 0]).squeeze(-1).float()


def policy_value_loss(
    output: tuple[Tensor, Tensor], batch: dict[str, Tensor], value_weight: float = 1.0
) -> Tensor:
    """策略支持软访问分布/one-hot教师标签；截断样本的价值mask为False，不充当平局。"""
    logits, value = output
    policy_loss = -(batch["policy"] * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    known = batch["value_mask"].float()
    value_loss = ((value - batch["value"]).square() * known).sum() / known.sum().clamp_min(1)
    return policy_loss + value_weight * value_loss
