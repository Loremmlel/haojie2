"""实体注意力和完整命令评分；同一次状态前向服务全部候选。"""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

STATE_WIDTH = 128
ACTION_WIDTH = 576


def device_for_bf16(name):
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用；CPU 功能验收可显式选择 --device cpu")
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("训练与推理要求硬件原生 BF16")
    return device


@dataclass
class ModelConfig:
    width: int = 256
    layers: int = 6
    heads: int = 8
    expansion: int = 4


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.width
        self.heads = config.heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, d * 3)
        self.output = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d * config.expansion), nn.GELU(),
                                 nn.Linear(d * config.expansion, d))

    def forward(self, x, mask):
        b, n, d = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        x = x + self.output(attention.transpose(1, 2).reshape(b, n, d))
        return x + self.ffn(self.norm2(x))


class Network(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        d = self.config.width
        self.state = nn.Linear(STATE_WIDTH, d)
        self.roles = nn.Embedding(32, d)
        self.kinds = nn.Embedding(128, d)
        self.blocks = nn.ModuleList(Block(self.config) for _ in range(self.config.layers))
        self.norm = nn.LayerNorm(d)
        self.action = nn.Sequential(nn.Linear(ACTION_WIDTH, d), nn.GELU(), nn.Linear(d, d))
        self.policy = nn.Sequential(nn.Linear(d * 4, d), nn.GELU(), nn.Linear(d, 1))
        self.value = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 1), nn.Tanh())
        self.auxiliary = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 4))
        self.bfloat16()

    def forward(self, states, state_mask, actions, action_mask, state_ids, action_ids):
        # 类别和引用保持整数语义，数值投影使用独立的缩放副本。
        features = states.clone()
        features[..., :10] /= 256
        x = self.state(features) + self.roles(state_ids[..., 0]) + self.kinds(state_ids[..., 1])
        for block in self.blocks:
            x = block(x, state_mask)
        x = self.norm(x)
        global_state = x[:, 0]
        scaled = actions.clone()
        scaled[..., :16] /= 256
        a = self.action(scaled)
        source = action_ids[..., 0]
        target = action_ids[..., 1]
        source_x = x.gather(1, source[..., None].expand(-1, -1, x.shape[-1]))
        target_x = x.gather(1, target[..., None].expand(-1, -1, x.shape[-1]))
        logits = self.policy(torch.cat((a, source_x, target_x,
                                       global_state[:, None].expand_as(a)), -1)).squeeze(-1)
        logits = logits.masked_fill(~action_mask, -torch.inf)
        return logits, self.value(global_state).squeeze(-1), self.auxiliary(global_state)

    def description(self):
        return {"architecture": asdict(self.config),
                "parameters": sum(p.numel() for p in self.parameters()),
                "state_width": STATE_WIDTH, "action_width": ACTION_WIDTH, "dtype": "bfloat16"}


def collate(items, device, pinned=False):
    b = len(items)
    n = max(len(s) for s, _ in items)
    k = max(len(a) for _, a in items)
    states = torch.zeros((b, n, STATE_WIDTH), dtype=torch.bfloat16, pin_memory=pinned)
    actions = torch.zeros((b, k, ACTION_WIDTH), dtype=torch.bfloat16, pin_memory=pinned)
    state_ids = torch.zeros((b, n, 2), dtype=torch.int64, pin_memory=pinned)
    action_ids = torch.zeros((b, k, 2), dtype=torch.int64, pin_memory=pinned)
    smask = torch.zeros((b, n), dtype=torch.bool, pin_memory=pinned)
    amask = torch.zeros((b, k), dtype=torch.bool, pin_memory=pinned)
    for i, (s, a) in enumerate(items):
        states[i, :len(s)].copy_(torch.from_numpy(s))
        actions[i, :len(a)].copy_(torch.from_numpy(a))
        # 必须在浮点特征转 BF16 前提取引用，避免第 257 行被舍入到第 256 行。
        state_ids[i, :len(s)].copy_(torch.from_numpy(s[:, :2]))
        action_ids[i, :len(a)].copy_(torch.from_numpy(a[:, 1:3]))
        smask[i, :len(s)] = True
        amask[i, :len(a)] = True
    return tuple(t.to(device, non_blocking=pinned) for t in (states, smask, actions, amask, state_ids, action_ids))
