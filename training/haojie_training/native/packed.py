"""原网络的紧凑推理；沿用采样精度，每条请求保留完整双向注意力边界。"""

from itertools import accumulate
from math import prod

import numpy as np
import torch

from ..model import HISTORY_LAYERS, RECENT_JOURNAL, ModelConfig, PolicyValueNet

ENCODER_KEYS = (
    "entities",
    "kinds",
    "globals",
    "global_indices",
    "cu_k",
    "cu_q",
    "cu_active",
    "readout",
    "active",
)
POLICY_KEYS = (
    "head_indices",
    "sources",
    "targets",
    "candidates",
    "candidate_mask",
    "candidate_positions",
)


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


def _attention(q, k, v, batch, layout, query):
    # 与 Torch 2.14 jagged SDPA 相同的 1MHK 布局；累积长度是 CUDA int32。
    # 官方对应：torch/nested/_internal/sdpa.py 的 memory efficient 分支。
    return torch.ops.aten._efficient_attention_forward(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        None,
        batch[f"cu_{query}"],
        batch["cu_k"],
        layout[f"max_{query}"],
        layout["max_k"],
        0.0,
        0,
        False,
    )[0].squeeze(0)


def prepare_row(row):
    """在 CPU 输入进程准备索引；返回值不持有原帧，不改变实体或候选顺序。"""
    if row["entity_mask"].all():
        valid = None
        length = len(row["entities"])
        source, target = row["sources"] + 1, row["targets"] + 1
    else:
        valid = np.flatnonzero(row["entity_mask"])
        length = len(valid)
        remap = np.zeros(len(row["entities"]) + 1, dtype=np.int64)
        remap[valid + 1] = np.arange(1, length + 1)
        source, target = remap[row["sources"] + 1], remap[row["targets"] + 1]
    # 末层只需有效候选的查询；完整 K/V 与中层活跃集合继续保留。
    selected = row["candidate_mask"]
    positions = np.unique(
        np.concatenate((np.zeros(1, dtype=np.int64), source[selected], target[selected]))
    )
    readout_map = np.zeros(length + 1, dtype=np.int64)
    readout_map[positions] = np.arange(len(positions))
    inverse = np.concatenate(
        (np.zeros(1, dtype=np.int64), readout_map[source], readout_map[target])
    )
    # 与 model.active_tokens 同一集合；有效近期日志、非历史实体及有效候选引用。
    journal = row["kinds"] == 250
    older = np.cumsum((journal & row["entity_mask"])[::-1])[::-1] > RECENT_JOURNAL
    history = (journal & older) | (~journal & (row["entities"][:, 0] == 5 / 32))
    active = np.concatenate((np.ones(1, dtype=bool), row["entity_mask"] & ~history))
    for key in ("sources", "targets"):
        active[np.where(row["candidate_mask"], row[key] + 1, 0)] = True
    active[1:] &= row["entity_mask"]
    if valid is not None:
        active = np.concatenate((active[:1], active[valid + 1]))
    return valid, positions, inverse, np.flatnonzero(active), length


class PackedPolicy:
    """只持有原模块和主机容量；每次前向直接读取最新参数，不缓存实体表示。"""

    def __init__(self, model, buffers, pinned, precision="fp32"):
        self.model, self.buffers, self.pinned = model, buffers, pinned
        from .aten import operator, parameters

        self.operator = operator() if pinned else None
        self.flash = (
            self.operator is not None
            and torch.backends.cuda.flash_sdp_enabled()
            and torch.backends.cuda.is_flash_attention_available()
        )
        self.factored = True
        self.fused = True
        self.residual_norm, self.norm_error = None, None
        if self.operator is not None and precision == "bf16":
            from .kernels import residual_norm

            self.residual_norm, self.norm_error = residual_norm(
                model.entity_projection.weight.device.index
            )
        self.modules = tuple(model.modules()) if self.operator else ()
        self.weights, self.epsilon = parameters(model) if self.operator else (None, None)
        self.policy_weights = (
            [
                parameter
                for layer in (
                    model.action_projection[0],
                    model.action_projection[2],
                    model.policy[0],
                    model.policy[2],
                    model.policy[4],
                )
                for parameter in (layer.weight, layer.bias)
            ]
            if self.operator
            else None
        )
        self.cpp_buffers = []

    def batch(self, examples):
        """压紧有效实体并重映射末层指针；原始特征、候选顺序和输入帧保持不变。"""
        native = self.native_dispatch
        if native and all(hasattr(row, "address") for row in examples):
            self.cpp_buffers, tensors, values = torch.ops.haojie.pack(
                [row.address for row in examples],
                [row.entity_count for row in examples],
                [row.candidate_count for row in examples],
                self.cpp_buffers,
                self.pinned,
                RECENT_JOURNAL,
            )
            for row in examples:
                row.validated = True
            layout = dict(
                zip(
                    (
                        "max_k",
                        "max_q",
                        "max_active",
                        "size",
                        "entities",
                        "candidates",
                        "candidate_slots",
                    ),
                    values,
                )
            )
            layout["scored_candidates"] = tensors[-1].numel()
            return dict(zip(ENCODER_KEYS + POLICY_KEYS, tensors)), layout

        for row in examples:
            if hasattr(row, "ensure_validated"):
                row.ensure_validated()

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

        records, lengths, queries, active_lengths = [], [], [], []
        for row in examples:
            prepared = getattr(row, "packing", None)
            # 主机准备统一走 NumPy 原生数组运算，避免每条请求反复进入小型 Torch 算子。
            row = {
                key: value if isinstance(value, np.ndarray) else value.numpy()
                for key, value in row.items()
            }
            valid, positions, inverse, active, length = prepared or prepare_row(row)
            records.append((row, valid, positions, inverse, active))
            lengths.append(length + 1)
            queries.append(len(positions))
            active_lengths.append(len(active))
        starts = list(accumulate(lengths, initial=0))
        read_starts = list(accumulate(queries, initial=0))
        active_starts = list(accumulate(active_lengths, initial=0))
        size, actions, reads = len(examples), max(len(r["sources"]) for r in examples), max(queries)
        scored = sum(np.count_nonzero(row["candidate_mask"]) for row, *_ in records)
        config = self.model.config
        batch = {
            "entities": empty("entities", (starts[-1], config.entity_features), torch.float32),
            "kinds": empty("kinds", (starts[-1],), torch.long),
            "globals": empty("globals", (size, config.global_features), torch.float32),
            "global_indices": empty("global_indices", (size,), torch.long),
            "cu_k": empty("cu_k", (size + 1,), torch.int32),
            "cu_q": empty("cu_q", (size + 1,), torch.int32),
            "cu_active": empty("cu_active", (size + 1,), torch.int32),
            "readout": empty("readout", (read_starts[-1],), torch.long),
            "active": empty("active", (active_starts[-1],), torch.long),
            "head_indices": empty(
                "head_indices", (scored,) if native else (size, reads), torch.long
            ),
            "sources": empty("sources", (scored,) if native else (size, actions), torch.long),
            "targets": empty("targets", (scored,) if native else (size, actions), torch.long),
            "candidates": empty(
                "candidates",
                (scored, config.action_features)
                if native
                else (size, actions, config.action_features),
                torch.float32,
            ),
            "candidate_mask": empty("candidate_mask", (size, actions), torch.bool),
        }
        if native:
            batch["candidate_positions"] = empty("candidate_positions", (scored,), torch.long)
        # NumPy 视图直接填充原页锁定容量；交付后的 CUDA 票据继续独占这组缓冲。
        arrays = {key: value.numpy() for key, value in batch.items()}
        if not native:
            arrays["sources"].fill(-1)
            arrays["targets"].fill(-1)
            arrays["candidates"].fill(0)
        arrays["candidate_mask"].fill(False)
        arrays["cu_k"][:] = starts
        arrays["cu_q"][:] = read_starts
        arrays["cu_active"][:] = active_starts
        arrays["global_indices"][:] = starts[:-1]
        candidate_start = 0
        for i, (row, valid, positions, inverse, active) in enumerate(records):
            start, end = starts[i : i + 2]
            qstart, qend = read_starts[i : i + 2]
            astart, aend = active_starts[i : i + 2]
            count = len(row["sources"])
            arrays["entities"][start].fill(0)
            arrays["kinds"][start] = 0
            for key in ("entities", "kinds"):
                arrays[key][start + 1 : end] = row[key] if valid is None else row[key][valid]
            arrays["globals"][i] = row["globals"]
            arrays["readout"][qstart:qend] = positions + start
            arrays["active"][astart:aend] = active + start
            arrays["candidate_mask"][i, :count] = row["candidate_mask"]
            if native:
                # 读出位置在 CPU 已知，不再上传补齐特征或在 GPU 重复展开指针。
                positions = np.flatnonzero(row["candidate_mask"])
                selected = slice(candidate_start, candidate_start + len(positions))
                arrays["head_indices"][selected] = qstart
                arrays["sources"][selected] = qstart + inverse[1 + positions]
                arrays["targets"][selected] = qstart + inverse[1 + count + positions]
                arrays["candidates"][selected] = row["candidates"][positions]
                arrays["candidate_positions"][selected] = positions + i * actions
                candidate_start = selected.stop
            else:
                arrays["head_indices"][i].fill(qstart)
                arrays["head_indices"][i, : len(positions)] = np.arange(qstart, qend)
                arrays["sources"][i, :count] = inverse[1 : 1 + count] - 1
                arrays["targets"][i, :count] = inverse[1 + count :] - 1
                arrays["candidates"][i, :count] = row["candidates"]
        layout = {
            "max_k": max(lengths),
            "max_q": max(queries),
            "max_active": max(active_lengths),
            "size": size,
            "entities": starts[-1] - size,
            "candidates": sum(len(row["sources"]) for row in examples),
            "candidate_slots": size * actions,
        }
        if native:
            layout["scored_candidates"] = batch["candidate_positions"].numel()
        return batch, layout

    @property
    def native_dispatch(self):
        return self.operator is not None and not any(
            m._forward_hooks or m._forward_pre_hooks for m in self.modules
        )

    def forward(self, batch, layout):
        """浅层全量交互，中层只更新活跃查询，末层读出；每层保留全部 K/V。"""
        model = self.model
        if self.native_dispatch:
            norm = self.residual_norm if self.fused else None
            x = self.operator(
                [batch[key] for key in ENCODER_KEYS],
                self.weights,
                model.config.heads,
                HISTORY_LAYERS,
                [layout["max_k"], layout["max_q"], layout["max_active"]],
                self.epsilon,
                self.flash,
                self.fused,
                norm[0] if norm else 0,
                norm[1] if norm else 0,
                torch.cuda.current_stream(model.entity_projection.weight.device).cuda_stream
                if norm
                else 0,
            )
            return torch.ops.haojie.packed_score(
                x,
                [batch[key] for key in POLICY_KEYS],
                self.policy_weights,
                model.action_projection[2].eps,
                self.factored,
                self.fused,
            )
        x = model.entity_projection(batch["entities"]) + model.kind_embedding(batch["kinds"])
        # 与普通encode的cat类型提升一致：AMP全局投影并入FP32残差流。
        global_tokens = model.global_projection(batch["globals"]).to(dtype=x.dtype)
        x.index_copy_(0, batch["global_indices"], global_tokens)
        width = model.config.width
        for index, block in enumerate(model.blocks):
            qkv = block.qkv(block.attention_norm(x)).view(-1, 3, block.heads, width // block.heads)
            q, k, v = qkv.unbind(1)
            last = index == len(model.blocks) - 1
            query = "q" if last else "active" if index >= HISTORY_LAYERS else "k"
            positions = batch["readout" if last else "active"] if query != "k" else None
            residual = x
            if positions is not None:
                q = q.index_select(0, positions)
                residual = x.index_select(0, positions)
            attended = _attention(q, k, v, batch, layout, query).reshape(-1, width)
            updated = residual + block.output(attended)
            updated = updated + block.ffn(block.ffn_norm(updated))
            # 中层仅回写活跃项，历史表示仍连接计算图，并继续作为下一层的 K/V。
            x = x.index_copy(0, positions, updated) if query == "active" else updated
        x = model.norm(x)
        head = x.index_select(0, batch["head_indices"].flatten()).view(
            layout["size"], batch["head_indices"].shape[1], width
        )
        return model.score_policy(batch, head)
