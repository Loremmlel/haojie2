"""已校验原生请求的同步策略执行器；主机缓冲只在上一批回传完成后复用。"""

import time

import torch

from ..data import INPUT_KEYS, _collate
from ..runtime import autocast


class PolicyInference:
    def __init__(self, model, device, precision="fp32", *, pinned=True):
        self.model, self.device, self.precision = model, torch.device(device), precision
        self.pinned = pinned and self.device.type == "cuda"
        self.buffers = {}
        self.collation_seconds = 0.0
        self.execution_seconds = 0.0
        self.calls = self.requests = self.entities = self.candidates = 0
        self.entity_slots = self.candidate_slots = 0

    def batch(self, examples):
        """输入来自一次完整协议校验，容量只增长；不缓存局面计算或改写输入帧。"""
        size = len(examples)
        entities = max(len(e["entities"]) for e in examples)
        actions = max(len(e["candidates"]) for e in examples)
        result = _collate(examples, INPUT_KEYS, buffers=self.buffers, pin_memory=self.pinned)
        self.entities += sum(len(e["entities"]) for e in examples)
        self.candidates += sum(len(e["candidates"]) for e in examples)
        self.entity_slots += size * entities
        self.candidate_slots += size * actions
        return result

    def __call__(self, examples):
        before = time.perf_counter()
        batch = self.batch(examples)
        self.collation_seconds += time.perf_counter() - before
        before = time.perf_counter()
        with torch.inference_mode(), autocast(self.device, self.precision):
            batch = {k: v.to(self.device, non_blocking=self.pinned) for k, v in batch.items()}
            # cpu() 是每批唯一结果同步；返回独立结果后才可重用主机输入。
            logits = self.model.policy_logits(batch).cpu()
        self.execution_seconds += time.perf_counter() - before
        if not torch.isfinite(logits).all():
            raise ValueError("模型输出非有限")
        self.calls += 1
        self.requests += len(examples)
        return logits
