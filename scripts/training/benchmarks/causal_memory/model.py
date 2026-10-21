"""隔离的 prefix-LM 实验：固定局面双向、日志因果、当前查询读取完整记忆。

只复用正式模型参数，不改变生产网络。输入为已校验的单条公开 CPU 张量，固定区
长度由采集器提供；无缓存参考可微，增量路径只供权重不变期间的推理计量。
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from haojie_training.model import PolicyValueNet, active_tokens


@dataclass
class _Layout:
    row: dict[str, Tensor]
    base_entities: int
    base: Tensor
    journal: Tensor
    dynamic: Tensor
    stable_globals: Tensor
    readout: Tensor
    sources: Tensor
    targets: Tensor

    @property
    def base_tokens(self):
        return 1 + len(self.base)

    @property
    def memory_tokens(self):
        return self.base_tokens + len(self.journal)


class CausalMemoryPolicy(nn.Module):
    """单流缓存包装器；多个流可共享同一组模型参数，但各自拥有缓存。

    active 模式复制全部正式活跃实体作为动态查询；readout 模式仅复制动作前缀、
    近期日志和有效候选引用。两者始终保留全部固定实体与日志作为记忆，不丢字段。
    根、固定输入、旧日志、权重版本或执行精度变化会重建缓存；只追加的日志增量计算。
    """

    def __init__(self, model: PolicyValueNet, query_scope="active"):
        super().__init__()
        if type(model) is not PolicyValueNet or query_scope not in {
            "active",
            "readout",
        }:
            raise ValueError("实验只接受正式模型与 active/readout 查询模式")
        self.model = model
        self.query_scope = query_scope
        self.stats = {
            "requests": 0,
            "cache_hits": 0,
            "root_rebuilds": 0,
            "reused_memory_tokens": 0,
            "appended_journal_tokens": 0,
            "prefill_tokens": 0,
            "dynamic_query_tokens": 0,
            "readout_query_tokens": 0,
        }
        self.last_layout = {}
        self.last_event = {}
        self.reset_cache()

    def reset_cache(self):
        """清空单流记忆，累计统计保留，便于调用者排除预热。"""
        self._cache = None
        self._kv = []

    def _layout(self, row, base_entities):
        if row["entities"].ndim != 2 or any(value.device.type != "cpu" for value in row.values()):
            raise ValueError("实验入口需要单条已校验的公开 CPU 张量")
        if not 0 <= base_entities <= len(row["entities"]):
            raise ValueError("固定实体前缀长度越界")
        journal_mask = row["kinds"] == 250
        if journal_mask[:base_entities].any():
            raise ValueError("固定局面前缀不能包含 MC 日志")
        valid = row["entity_mask"]
        base = valid[:base_entities].nonzero().flatten()
        journal = (journal_mask & valid).nonzero().flatten()
        dynamic_mask = active_tokens({key: value.unsqueeze(0) for key, value in row.items()})[0, 1:]
        if self.query_scope == "readout":
            dynamic_mask[:base_entities] = False
        # 所有节点附加实体保留；已校验的生产输入中它们是前缀、参数或路径。
        dynamic_mask[base_entities:] |= valid[base_entities:] & ~journal_mask[base_entities:]
        for key in ("sources", "targets"):
            pointers = row[key][row["candidate_mask"]]
            dynamic_mask[pointers[pointers >= 0]] = True
        dynamic = (dynamic_mask & valid).nonzero().flatten()
        remap = torch.zeros(len(valid) + 1, dtype=torch.long)
        remap[dynamic + 1] = torch.arange(1, len(dynamic) + 1)
        pointers = [
            remap[torch.where(row["candidate_mask"], row[key] + 1, 0)] - 1
            for key in ("sources", "targets")
        ]
        readout = torch.cat(
            (torch.zeros(1, dtype=torch.long), *(pointer + 1 for pointer in pointers))
        ).unique()
        remap = torch.zeros(len(dynamic) + 1, dtype=torch.long)
        remap[readout] = torch.arange(len(readout))
        pointers = [remap[pointer + 1] - 1 for pointer in pointers]
        stable_globals = row["globals"].clone()
        stable_globals[31] = 0
        return _Layout(
            row,
            base_entities,
            base,
            journal,
            dynamic,
            stable_globals,
            readout,
            *pointers,
        )

    @property
    def device(self):
        return self.model.entity_projection.weight.device

    def _entities(self, layout, positions):
        entities = layout.row["entities"].index_select(0, positions).to(self.device)
        kinds = layout.row["kinds"].index_select(0, positions).to(self.device)
        return self.model.entity_projection(entities) + self.model.kind_embedding(kinds)

    def _memory_input(self, layout):
        global_token = self.model.global_projection(layout.stable_globals.to(self.device))[None]
        positions = torch.cat((layout.base, layout.journal))
        return torch.cat((global_token, self._entities(layout, positions))).unsqueeze(0)

    def _dynamic_input(self, layout):
        global_token = self.model.global_projection(layout.row["globals"].to(self.device))[None]
        return torch.cat((global_token, self._entities(layout, layout.dynamic))).unsqueeze(0)

    def _allowed(self, base, memory, total):
        positions = torch.arange(total, device=self.device)
        query, key = positions[:, None], positions[None, :]
        # 固定区内部双向；日志只看固定区及此前日志；动态区读取全部记忆和动态查询。
        return (
            ((query < base) & (key < base))
            | ((query >= base) & (query < memory) & (key <= query))
            | (query >= memory)
        )[None, None]

    def _score(self, layout, dynamic, policy_only):
        x = self.model.norm(dynamic)
        batch = {
            key: layout.row[key].unsqueeze(0).to(self.device)
            for key in ("candidates", "candidate_mask")
        }
        batch["sources"] = layout.sources.unsqueeze(0).to(self.device)
        batch["targets"] = layout.targets.unsqueeze(0).to(self.device)
        return (
            self.model.score_policy(batch, x)[0],
            None if policy_only else self.model.value(x[:, 0]).reshape(()).float(),
        )

    def forward_reference(self, row, base_entities, *, policy_only=False):
        """可微的完整重算：同一序列与同一注意力关系，不读取或改写增量缓存。"""
        layout = self._layout(row, base_entities)
        x = torch.cat((self._memory_input(layout), self._dynamic_input(layout)), dim=1)
        allowed = self._allowed(layout.base_tokens, layout.memory_tokens, x.shape[1])
        for index, block in enumerate(self.model.blocks):
            if index == len(self.model.blocks) - 1:
                # 末层只更新评分实际读取的查询；记忆及全部动态实体仍提供 K/V。
                readout = (layout.readout.to(self.device) + layout.memory_tokens)[None]
                x = block(x, None, readout)
            else:
                x = block(x, allowed)
        return self._score(layout, x, policy_only)

    def _qkv(self, block, x):
        batch, tokens, width = x.shape
        qkv = block.qkv(block.attention_norm(x))
        return qkv.reshape(batch, tokens, 3, block.heads, width // block.heads).permute(
            2, 0, 3, 1, 4
        )

    def _finish(self, block, x, q, k, v, allowed=None):
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
        x = x + block.output(attended.transpose(1, 2).reshape_as(x))
        return x + block.ffn(block.ffn_norm(x))

    def _signature(self):
        device = self.device
        return (
            device,
            self.model.entity_projection.weight.dtype,
            torch.is_autocast_enabled(device.type),
            torch.get_autocast_dtype(device.type),
            tuple((id(parameter), parameter._version) for parameter in self.model.parameters()),
        )

    def _reusable(self, layout, root_key, signature):
        cached = self._cache
        if (
            cached is None
            or cached["root_key"] != root_key
            or cached["signature"] != signature
            or cached["base_entities"] != layout.base_entities
            or len(layout.journal) < len(cached["journal"])
        ):
            return False
        for key in ("entities", "kinds", "entity_mask"):
            if not torch.equal(cached[key], layout.row[key][: layout.base_entities]):
                return False
        return torch.equal(cached["globals"], layout.stable_globals) and torch.equal(
            cached["journal"],
            layout.row["entities"].index_select(0, layout.journal[: len(cached["journal"])]),
        )

    def _prefill(self, layout):
        x = self._memory_input(layout)
        allowed = self._allowed(layout.base_tokens, layout.memory_tokens, layout.memory_tokens)
        self._kv = []
        for index, block in enumerate(self.model.blocks):
            q, k, v = self._qkv(block, x)
            self._kv.append((k, v))
            if index < len(self.model.blocks) - 1:
                x = self._finish(block, x, q, k, v, allowed)

    @torch.inference_mode()
    def forward_cached(self, row, base_entities, root_key, *, policy_only=False):
        """仅推理；输入缓冲可被调用者复用，缓存快照独占其 CPU 数据和每层 K/V。"""
        if self.model.training:
            raise ValueError("增量缓存仅用于 eval 模型；可微计算请使用 forward_reference")
        layout = self._layout(row, base_entities)
        signature = self._signature()
        hit = self._reusable(layout, root_key, signature)
        previous = len(self._cache["journal"]) if hit else 0
        added = len(layout.journal) - previous
        self.stats["requests"] += 1
        self.stats["dynamic_query_tokens"] += len(layout.dynamic) + 1
        self.stats["readout_query_tokens"] += len(layout.readout)
        if hit:
            self.stats["cache_hits"] += 1
            self.stats["reused_memory_tokens"] += layout.base_tokens + previous
            self.stats["appended_journal_tokens"] += added
        else:
            self.stats["root_rebuilds"] += 1
            self.stats["prefill_tokens"] += layout.memory_tokens
            self._prefill(layout)
            self._cache = {
                "root_key": root_key,
                "signature": signature,
                "base_entities": base_entities,
                "globals": layout.stable_globals.clone(),
                **{
                    key: row[key][:base_entities].clone()
                    for key in ("entities", "kinds", "entity_mask")
                },
            }
        self._cache["journal"] = row["entities"].index_select(0, layout.journal).clone()
        self.last_layout = {
            "base_tokens": layout.base_tokens,
            "journal_tokens": len(layout.journal),
            "memory_tokens": layout.memory_tokens,
            "dynamic_query_tokens": len(layout.dynamic) + 1,
            "readout_query_tokens": len(layout.readout),
            "cache_hit": hit,
            "prefill_tokens": 0 if hit else layout.memory_tokens,
            "reused_memory_tokens": layout.base_tokens + previous if hit else 0,
            "appended_journal_tokens": added if hit else 0,
        }
        self.last_event = self.last_layout
        x = self._dynamic_input(layout)
        appended = added if hit else 0
        allowed = None
        if appended:
            # 新日志与动态查询逐层一起计算；只有日志查询禁止读取更晚日志和动态区。
            journal = self._entities(layout, layout.journal[previous:]).unsqueeze(0)
            x = torch.cat((journal, x), dim=1)
            old_tokens = layout.memory_tokens - appended
            query = torch.arange(x.shape[1], device=self.device)[:, None]
            key = torch.arange(old_tokens + x.shape[1], device=self.device)[None, :]
            allowed = ((query >= appended) | (key <= old_tokens + query))[None, None]
        for index, (block, (memory_k, memory_v)) in enumerate(zip(self.model.blocks, self._kv)):
            q, k, v = self._qkv(block, x)
            if appended:
                memory_k = torch.cat((memory_k, k[:, :, :appended]), dim=2)
                memory_v = torch.cat((memory_v, v[:, :, :appended]), dim=2)
                self._kv[index] = (memory_k, memory_v)
                k, v = k[:, :, appended:], v[:, :, appended:]
            last = index == len(self.model.blocks) - 1
            if last:
                positions = layout.readout.to(self.device) + appended
                q, x = q.index_select(2, positions), x.index_select(1, positions)
            x = self._finish(
                block,
                x,
                q,
                torch.cat((memory_k, k), dim=2),
                torch.cat((memory_v, v), dim=2),
                None if last else allowed,
            )
        return self._score(layout, x, policy_only)
