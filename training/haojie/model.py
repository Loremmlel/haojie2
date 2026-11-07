"""实体主干只前向一次；局部记录池、几何注意力和条件参数头共享状态表示。"""

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

STATE_WIDTH = 128
ACTION_WIDTH = 704
RELATION_WIDTH = 8


def device_for_bf16(name):
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用；CPU 验收可选择 --device cpu")
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("训练与推理要求硬件原生 BF16")
    return device


@dataclass
class ModelConfig:
    width: int = 384
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
        self.ffn = nn.Sequential(
            nn.Linear(d, d * config.expansion),
            nn.GELU(),
            nn.Linear(d * config.expansion, d),
        )

    def forward(self, x, bias):
        b, n, d = x.shape
        q, k, v = (
            self.qkv(self.norm1(x))
            .view(b, n, 3, self.heads, d // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.output(attention.transpose(1, 2).reshape(b, n, d))
        return x + self.ffn(self.norm2(x))


class Network(nn.Module):
    @classmethod
    def from_weights(cls, weights, device):
        # v3 恢复点与独立历史权重都保留状态投影；先按实际宽度建网，再严格加载。
        model = cls(ModelConfig(width=weights["state.weight"].shape[0])).to(device)
        model.load_state_dict(weights)
        return model

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        d = self.config.width
        self.state = nn.Linear(STATE_WIDTH, d)
        self.roles = nn.Embedding(32, d)
        self.kinds = nn.Embedding(128, d)
        self.records = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.relations = nn.Linear(RELATION_WIDTH, self.config.heads, bias=False)
        self.blocks = nn.ModuleList(
            Block(self.config) for _ in range(self.config.layers)
        )
        self.norm = nn.LayerNorm(d)
        self.action = nn.Sequential(
            nn.Linear(ACTION_WIDTH, d), nn.GELU(), nn.Linear(d, d)
        )
        self.policy = nn.Sequential(nn.Linear(d * 4, d), nn.GELU(), nn.Linear(d, 1))
        self.value = nn.Sequential(
            nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 1), nn.Tanh()
        )
        self.auxiliary = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 4))
        self.bfloat16()

    def embed(self, features, ids):
        scaled = features.clone()
        scaled[..., :11].div_(256)
        return self.state(scaled) + self.roles(ids[..., 0]) + self.kinds(ids[..., 1])

    def encode(
        self,
        states,
        mask,
        records,
        record_mask,
        owners,
        state_ids,
        record_ids,
        relations,
    ):
        x = self.embed(states, state_ids)
        if records.shape[1]:
            local = (
                self.records(self.embed(records, record_ids)) * record_mask[..., None]
            )
            pooled = torch.zeros_like(x)
            pooled.scatter_add_(1, owners[..., None].expand_as(local), local)
            counts = torch.zeros(x.shape[:2], dtype=x.dtype, device=x.device)
            counts.scatter_add_(1, owners, record_mask.to(x.dtype))
            x = x + pooled / counts.clamp_min(1).sqrt()[..., None]
        bias = (
            self.relations(relations)
            .permute(0, 3, 1, 2)
            .masked_fill(~mask[:, None, None, :], -torch.inf)
        )
        for block in self.blocks:
            x = block(x, bias)
        x = self.norm(x)
        return x, self.value(x[:, 0]).squeeze(-1), self.auxiliary(x[:, 0])

    def decode(self, context, actions, mask, indices):
        scaled = actions.clone()
        scaled[..., :16].div_(256)
        # 附带的死亡对象描述和状态行采用相同数值尺度。
        scaled[..., 576:587].div_(256)
        scaled[..., 42] = 0
        focus = context.gather(
            1, indices[..., 2, None].expand(-1, -1, context.shape[-1])
        )
        a = self.action(scaled) + focus + self.kinds(actions[..., 12].long())
        source = context.gather(
            1, indices[..., 0, None].expand(-1, -1, context.shape[-1])
        )
        target = context.gather(
            1, indices[..., 1, None].expand(-1, -1, context.shape[-1])
        )
        logits = self.policy(
            torch.cat((a, source, target, context[:, 0, None].expand_as(a)), -1)
        ).squeeze(-1)
        return logits.masked_fill(~mask, -torch.inf)

    def description(self):
        return {
            "architecture": asdict(self.config),
            "parameters": sum(p.numel() for p in self.parameters()),
            "state_width": STATE_WIDTH,
            "action_width": ACTION_WIDTH,
            "dtype": "bfloat16",
            "format": "entity-autoregressive-v2",
        }


def _host_tensor(array, dtype, pinned):
    source = torch.from_numpy(array)
    if not pinned:
        return source.to(dtype=dtype)
    return torch.empty(array.shape, dtype=dtype, pin_memory=True).copy_(source)


def collate_states(
    items, device, pinned=False, profile=None, scope="states", cuda=False
):
    with profile.stage(scope + ".collate_cpu") if profile else nullcontext():
        b = len(items)
        ns = np.array([len(v[0]) for v in items])
        ms = np.array([len(v[1]) for v in items])
        n, m = ns.max(), ms.max()
        states = np.zeros((b, n, STATE_WIDTH), np.float32)
        records = np.zeros((b, m, STATE_WIDTH), np.float32)
        owners = np.zeros((b, m), np.int64)
        relations = np.zeros((b, n, n, RELATION_WIDTH), np.float32)
        for i, (s, r, o, g) in enumerate(items):
            states[i, : len(s)] = s
            records[i, : len(r)] = r
            owners[i, : len(o)] = o
            relations[i, : len(s), : len(s)] = g
        # 先按行拼原始数组，再整批转换；索引从未舍入的输入读取。
        batch = (
            _host_tensor(states, torch.bfloat16, pinned),
            _host_tensor(np.arange(n) < ns[:, None], torch.bool, pinned),
            _host_tensor(records, torch.bfloat16, pinned),
            _host_tensor(np.arange(m) < ms[:, None], torch.bool, pinned),
            _host_tensor(owners, torch.int64, pinned),
            _host_tensor(states[..., :2], torch.int64, pinned),
            _host_tensor(records[..., :2], torch.int64, pinned),
            _host_tensor(relations, torch.bfloat16, pinned),
        )
    with profile.stage(scope + ".h2d_submit", cuda=cuda) if profile else nullcontext():
        return tuple(t.to(device, non_blocking=pinned) for t in batch)


def collate_options(
    items, device, pinned=False, profile=None, scope="options", cuda=False
):
    with profile.stage(scope + ".collate_cpu") if profile else nullcontext():
        sizes = np.array([len(a) for a in items])
        k = sizes.max()
        actions = torch.zeros(
            (len(items), k, ACTION_WIDTH), dtype=torch.bfloat16, pin_memory=pinned
        )
        # 动作数相差很大；只转换有效行，用原始 BF16 位填入已经补零的目标。
        packed = (
            torch.from_numpy(np.concatenate(items))
            .bfloat16()
            .view(torch.uint16)
            .numpy()
        )
        output = actions.view(torch.uint16).numpy()
        raw_ids = np.zeros((len(items), k, 3), np.int64)
        offset = 0
        for i, a in enumerate(items):
            output[i, : len(a)] = packed[offset : offset + len(a)]
            raw_ids[i, : len(a), :2] = a[:, 1:3]
            raw_ids[i, : len(a), 2] = a[:, 42]
            offset += len(a)
        mask = _host_tensor(np.arange(k) < sizes[:, None], torch.bool, pinned)
        ids = _host_tensor(raw_ids, torch.int64, pinned)
    with profile.stage(scope + ".h2d_submit", cuda=cuda) if profile else nullcontext():
        return tuple(t.to(device, non_blocking=pinned) for t in (actions, mask, ids))
