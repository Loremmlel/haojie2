"""已校验原生请求的同步策略执行器；主机缓冲只在上一批回传完成后复用。"""

import time
from collections import Counter

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
        self.batch_sizes = Counter()
        self.cuda_timing = {"batches": 0, "upload_ms": 0.0, "model_ms": 0.0, "download_ms": 0.0}

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
        # 每256批取一组设备事件，只作分账诊断；计时开销仍包含在总墙钟内。
        events = (
            [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            if (self.device.type == "cuda" and self.calls % 256 == 0)
            else None
        )
        with torch.inference_mode(), autocast(self.device, self.precision):
            if events:
                events[0].record()
            batch = {k: v.to(self.device, non_blocking=self.pinned) for k, v in batch.items()}
            if events:
                events[1].record()
            # cpu() 是每批唯一结果同步；返回独立结果后才可重用主机输入。
            output = self.model.policy_logits(batch)
            if events:
                events[2].record()
            logits = output.cpu()
            if events:
                events[3].record()
                events[3].synchronize()
                self.cuda_timing["batches"] += 1
                for i, key in enumerate(("upload_ms", "model_ms", "download_ms")):
                    self.cuda_timing[key] += events[i].elapsed_time(events[i + 1])
        self.execution_seconds += time.perf_counter() - before
        if not torch.isfinite(logits).all():
            raise ValueError("模型输出非有限")
        self.calls += 1
        self.requests += len(examples)
        self.batch_sizes[len(examples)] += 1
        return logits
