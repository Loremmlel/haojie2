"""公开根编码与根内自回归解码；模型浮点张量只有 BF16。"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

DTYPE = torch.bfloat16
FORMAT = "haojie-root-vtrace-v2"


def validate_weights(weights):
    if any(v.is_floating_point() and v.dtype != DTYPE for v in weights.values()):
        raise ValueError("模型浮点参数必须全部采用 BF16")
    if any(not bool(torch.isfinite(v).all()) for v in weights.values()):
        raise ValueError("模型包含非有限参数")


@dataclass(frozen=True)
class Config:
    width: int = 256
    layers: int = 3
    heads: int = 4
    ffn: int = 512
    history: int = 128


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.norm = nn.LayerNorm(config.width, dtype=DTYPE)
        self.qkv = nn.Linear(config.width, 3 * config.width, dtype=DTYPE)
        self.output = nn.Linear(config.width, config.width, dtype=DTYPE)
        self.ffn_norm = nn.LayerNorm(config.width, dtype=DTYPE)
        self.ffn = nn.Sequential(
            nn.Linear(config.width, config.ffn, dtype=DTYPE),
            nn.GELU(),
            nn.Linear(config.ffn, config.width, dtype=DTYPE),
        )

    def forward(self, x, mask):
        batch, count, width = x.shape
        q, k, v = (
            self.qkv(self.norm(x))
            .reshape(batch, count, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None])
        x = x + self.output(attended.transpose(1, 2).reshape(batch, count, width))
        return x + self.ffn(self.ffn_norm(x))


class Network(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        self.entities = nn.Linear(64, config.width, dtype=DTYPE)
        self.kinds = nn.Embedding(256, config.width, dtype=DTYPE)
        self.globals = nn.Linear(32, config.width, dtype=DTYPE)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.width, dtype=DTYPE)
        self.prefix = nn.Linear(64, config.width, dtype=DTYPE)
        self.history = nn.GRU(68, config.history, batch_first=True, dtype=DTYPE)
        context = config.width * 2 + config.history
        self.value = nn.Sequential(
            nn.Linear(context, config.width, dtype=DTYPE),
            nn.GELU(),
            nn.Linear(config.width, 1, dtype=DTYPE),
        )
        self.candidate = nn.Linear(64, config.width, dtype=DTYPE)
        self.policy = nn.Sequential(
            nn.Linear(context + config.width * 3, config.width, dtype=DTYPE),
            nn.GELU(),
            nn.Linear(config.width, 1, dtype=DTYPE),
        )

    def flatten_history(self):
        """固定 Torch 2.14 的 BF16 GRU 权重布局；不改变门控公式或计算精度。

        该版本 flatten_parameters 的 cuDNN 白名单漏掉 BF16，导致每次调用重新打包。
        使用其同一原生布局操作，参数和优化器继续引用原 Parameter。
        """
        parameter = next(self.history.parameters())
        if parameter.device.type != "cuda":
            return
        from torch.backends.cudnn.rnn import get_cudnn_mode

        with torch.cuda.device(parameter.device), torch.no_grad():
            torch._cudnn_rnn_flatten_weight(
                self.history._flat_weights,
                4,
                68,
                get_cudnn_mode("GRU"),
                self.config.history,
                0,
                1,
                True,
                False,
            )

    def encode(self, entities, kinds, globals_, mask):
        """签名刻意不接受候选、前缀或尝试历史。"""
        x = torch.cat(
            (self.globals(globals_)[:, None], self.entities(entities) + self.kinds(kinds)), 1
        )
        valid = torch.cat((torch.ones_like(mask[:, :1]), mask), 1)
        for block in self.blocks:
            x = block(x, valid)
        return self.norm(x)

    def histories(self, events, lengths, initial=None):
        if events.shape[1] == 0:
            return (
                initial
                if initial is not None
                else events.new_zeros((events.shape[0], self.config.history))
            )
        sequence, _ = self.history(events, None if initial is None else initial[None])
        last = sequence[
            torch.arange(len(lengths), device=lengths.device), (lengths - 1).clamp_min(0)
        ]
        fallback = 0 if initial is None else initial
        return torch.where((lengths > 0)[:, None], last, fallback)

    def node_histories(self, nodes, device, staging, chunk_size=256):
        """每个根的真实历史只重建一次，在各选择发生前取 GRU 状态。"""
        grouped = {}
        for i, node in enumerate(nodes):
            grouped.setdefault(node["root"], []).append(i)
        groups = list(grouped.values())
        longest = [max(group, key=lambda i: nodes[i]["history_end"]) for group in groups]
        ends = [nodes[i]["history_end"] for i in longest]
        end = max(ends)
        hidden = torch.zeros((len(groups), self.config.history), device=device, dtype=DTYPE)
        outputs = [hidden[0] for _ in nodes]

        def sequence(events, initial):
            values, last = self.history(events, initial[None])
            return values, last[0]

        for start in range(0, end, chunk_size):
            size = min(chunk_size, end - start)
            events = torch.zeros((len(groups), size, 68), dtype=DTYPE)
            for i, index in enumerate(longest):
                count = min(size, max(0, ends[i] - start))
                events[i, :count] = node_events(nodes[index])[start : start + count]
            events = staging.transfer("history-events", events)
            if torch.is_grad_enabled() and end > chunk_size:
                values, hidden = checkpoint(
                    sequence,
                    events,
                    hidden,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                values, hidden = sequence(events, hidden)
            for i, group in enumerate(groups):
                for index in group:
                    length = nodes[index]["history_end"]
                    if start < length <= start + size:
                        outputs[index] = values[i, length - start - 1]
        return torch.stack(outputs)

    def decode(self, encoded, prefix, prefix_mask, history, candidates, sources, targets, mask):
        prefix_context = (self.prefix(prefix) * prefix_mask[..., None]).sum(1)
        prefix_context = prefix_context / prefix_mask.sum(1).clamp_min(1)[:, None]
        context = torch.cat((encoded[:, 0], prefix_context, history), -1)
        width = encoded.shape[-1]
        source = encoded.gather(1, (sources + 1)[..., None].expand(-1, -1, width))
        target = encoded.gather(1, (targets + 1)[..., None].expand(-1, -1, width))
        logits = (
            self.policy(
                torch.cat(
                    (
                        context[:, None].expand(-1, candidates.shape[1], -1),
                        source,
                        target,
                        self.candidate(candidates),
                    ),
                    -1,
                )
            )
            .squeeze(-1)
            .masked_fill(~mask, -torch.inf)
        )
        return logits, self.value(context).squeeze(-1)


def root_batch(roots, device, staging=None):
    count = max(1, max(len(root["entities"]) for root in roots))
    entities = torch.zeros((len(roots), count, 64), dtype=DTYPE)
    kinds = torch.zeros((len(roots), count), dtype=torch.long)
    mask = torch.zeros((len(roots), count), dtype=torch.bool)
    for i, root in enumerate(roots):
        n = len(root["entities"])
        entities[i, :n] = root["entities"]
        kinds[i, :n] = root["kinds"]
        mask[i, :n] = True
    tensors = (entities, kinds, torch.stack([r["globals"] for r in roots]), mask)
    return tuple(
        staging.transfer(f"root-{i}", t) if staging else t.to(device) for i, t in enumerate(tensors)
    )


def node_events(node):
    return node["history"]["events"][: node["history_end"]]


def work_batches(items, limit, dimensions, work_limit=262144):
    """按填充后的实际工作量切块；独立的大样本单独处理，不能裁剪支持集。"""
    chunk, maxima = [], ()
    for item in items:
        shape = dimensions(item)
        proposed = tuple(max(a, b) for a, b in zip(maxima or shape, shape))
        if chunk and (len(chunk) >= limit or (len(chunk) + 1) * sum(proposed) > work_limit):
            yield chunk
            chunk, maxima = [], ()
        chunk.append(item)
        maxima = tuple(max(a, b) for a, b in zip(maxima or shape, shape))
    if chunk:
        yield chunk


def node_batch(nodes, device, staging=None, *, include_events=True):
    def rows(key, width):
        values = [node_events(n) if key == "events" else n[key] for n in nodes]
        count = max(1, max(len(v) for v in values))
        result = torch.zeros((len(nodes), count, width), dtype=DTYPE)
        mask = torch.zeros((len(nodes), count), dtype=torch.bool)
        for i, value in enumerate(values):
            n = len(value)
            result[i, :n] = value
            mask[i, :n] = True
        return result, mask

    prefix, prefix_mask = rows("prefix", 64)
    events = (
        rows("events", 68)[0] if include_events else torch.empty(len(nodes), 0, 68, dtype=DTYPE)
    )
    candidates, mask = rows("candidates", 64)
    sources = torch.full(mask.shape, -1, dtype=torch.long)
    targets = sources.clone()
    for i, node in enumerate(nodes):
        count = len(node["candidates"])
        sources[i, :count] = node["sources"]
        targets[i, :count] = node["targets"]
        mask[i, :count] = node["mask"]
    lengths = torch.tensor([n["history_end"] for n in nodes])
    tensors = (prefix, prefix_mask, events, lengths, candidates, sources, targets, mask)
    return tuple(
        staging.transfer(f"node-{i}", t) if staging else t.to(device) for i, t in enumerate(tensors)
    )
