"""原网络的紧凑推理；沿用采样精度，每条请求保留完整双向注意力边界。"""

from itertools import accumulate
from math import prod

import torch

from ..data import _collate
from ..model import ModelConfig, PolicyValueNet


def supports_packed(model, device, precision):
    """私有算子只在当前固定依赖及已验证的原网络配置启用，其他入口沿用普通推理。"""
    if not (
        device.type == "cuda"
        and precision in {"fp32", "bf16"}
        and type(model) is PolicyValueNet
        and model.config == ModelConfig()
        and next(model.parameters()).dtype == torch.float32
        and torch.__version__.split("+")[0] == "2.14.0"
        and torch.version.cuda is not None
        and torch.backends.cuda.is_built()
        and torch.backends.cuda.mem_efficient_sdp_enabled()
    ):
        return False
    # 开关开启不代表当前构建/GPU有该内核；按正式头宽和实际AMP计算类型检查。
    width, heads = model.config.width, model.config.heads
    query = (
        model.entity_projection.weight.detach().view(-1)[:width].view(1, heads, 1, width // heads)
    )
    if precision == "bf16":
        query = query.to(dtype=torch.bfloat16)
    with torch.cuda.device(device):
        return torch.backends.cuda.can_use_efficient_attention(
            torch.backends.cuda.SDPAParams(query, query, query, None, 0.0, False, False)
        )


def _attention(q, k, v, batch, layout, last):
    # 与 Torch 2.14 jagged SDPA 相同的 1MHK 布局；累积长度是 CUDA int32。
    # 官方对应：torch/nested/_internal/sdpa.py 的 memory efficient 分支。
    return torch.ops.aten._efficient_attention_forward(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        None,
        batch["cu_q"] if last else batch["cu_k"],
        batch["cu_k"],
        layout["max_q"] if last else layout["max_k"],
        layout["max_k"],
        0.0,
        0,
        False,
    )[0].squeeze(0)


class PackedPolicy:
    """只持有原模块和主机容量；每次前向直接读取最新参数，不缓存实体表示。"""

    def __init__(self, model, buffers, pinned):
        self.model, self.buffers, self.pinned = model, buffers, pinned

    def batch(self, examples):
        """压紧有效实体并重映射末层指针；原始特征、候选顺序和输入帧保持不变。"""

        def empty(key, shape, dtype):
            key = "packed_" + key
            count = prod(shape)
            old = self.buffers.get(key)
            if old is None or old.numel() < count:
                with torch.inference_mode(False):
                    old = self.buffers[key] = torch.empty(
                        count, dtype=dtype, pin_memory=self.pinned
                    )
            return old[:count].view(shape)

        records, lengths, queries = [], [], []
        for row in examples:
            # 原生请求通常没有填充，直接复制切片，省去索引表及一次实体复制。
            if row["entity_mask"].all():
                valid = None
                length = len(row["entities"])
                source, target = row["sources"] + 1, row["targets"] + 1
            else:
                valid = row["entity_mask"].nonzero().flatten()
                length = len(valid)
                remap = torch.zeros(len(row["entities"]) + 1, dtype=torch.long)
                remap[valid + 1] = torch.arange(1, length + 1)
                # 已校验的有效候选不引用遮罩实体；无效候选的指针可映射到全局占位。
                source, target = remap[row["sources"] + 1], remap[row["targets"] + 1]
            positions, inverse = torch.unique(
                torch.cat((torch.zeros(1, dtype=torch.long), source, target)),
                sorted=True,
                return_inverse=True,
            )
            records.append((valid, positions, inverse))
            lengths.append(length + 1)
            queries.append(len(positions))
        starts = list(accumulate(lengths, initial=0))
        read_starts = list(accumulate(queries, initial=0))
        size, actions, reads = len(examples), max(len(r["sources"]) for r in examples), max(queries)
        config = self.model.config
        batch = {
            "entities": empty("entities", (starts[-1], config.entity_features), torch.float32),
            "kinds": empty("kinds", (starts[-1],), torch.long),
            "globals": empty("globals", (size, config.global_features), torch.float32),
            "global_indices": empty("global_indices", (size,), torch.long),
            "cu_k": empty("cu_k", (size + 1,), torch.int32),
            "cu_q": empty("cu_q", (size + 1,), torch.int32),
            "readout": empty("readout", (read_starts[-1],), torch.long),
            "head_indices": empty("head_indices", (size, reads), torch.long),
            "sources": empty("sources", (size, actions), torch.long),
            "targets": empty("targets", (size, actions), torch.long),
        }
        batch["sources"].fill_(-1)
        batch["targets"].fill_(-1)
        batch["cu_k"].copy_(torch.tensor(starts, dtype=torch.int32))
        batch["cu_q"].copy_(torch.tensor(read_starts, dtype=torch.int32))
        batch["global_indices"].copy_(torch.tensor(starts[:-1], dtype=torch.long))
        for i, (row, (valid, positions, inverse)) in enumerate(zip(examples, records)):
            start, end = starts[i : i + 2]
            qstart, qend = read_starts[i : i + 2]
            count = len(row["sources"])
            batch["entities"][start].zero_()
            batch["kinds"][start] = 0
            for key in ("entities", "kinds"):
                values = row[key] if valid is None else row[key].index_select(0, valid)
                batch[key][start + 1 : end].copy_(values)
            batch["globals"][i].copy_(row["globals"])
            batch["readout"][qstart:qend].copy_(positions + start)
            batch["head_indices"][i].fill_(qstart)
            batch["head_indices"][i, : len(positions)].copy_(torch.arange(qstart, qend))
            batch["sources"][i, :count].copy_(inverse[1 : 1 + count] - 1)
            batch["targets"][i, :count].copy_(inverse[1 + count :] - 1)
        batch.update(
            _collate(
                examples,
                {"candidates", "candidate_mask"},
                buffers=self.buffers,
                pin_memory=self.pinned,
            )
        )
        return batch, {
            "max_k": max(lengths),
            "max_q": max(queries),
            "size": size,
            "entities": starts[-1] - size,
            "candidates": sum(len(row["sources"]) for row in examples),
            "candidate_slots": size * actions,
        }

    def forward(self, batch, layout):
        """前层完整 Q/K/V，末层只读取所需 Q；候选头复用原模型方法。"""
        model = self.model
        x = model.entity_projection(batch["entities"]) + model.kind_embedding(batch["kinds"])
        # 与普通encode的cat类型提升一致：AMP全局投影并入FP32残差流。
        global_tokens = model.global_projection(batch["globals"]).to(dtype=x.dtype)
        x.index_copy_(0, batch["global_indices"], global_tokens)
        width = model.config.width
        for index, block in enumerate(model.blocks):
            qkv = block.qkv(block.attention_norm(x)).view(-1, 3, block.heads, width // block.heads)
            q, k, v = qkv.unbind(1)
            last = index == len(model.blocks) - 1
            if last:
                q = q.index_select(0, batch["readout"])
                x = x.index_select(0, batch["readout"])
            attended = _attention(q, k, v, batch, layout, last).reshape(-1, width)
            x = x + block.output(attended)
            x = x + block.ffn(block.ffn_norm(x))
        x = model.norm(x)
        head = x.index_select(0, batch["head_indices"].flatten()).view(
            layout["size"], batch["head_indices"].shape[1], width
        )
        return model.score_policy(batch, head)
