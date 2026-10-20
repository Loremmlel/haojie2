"""已校验原生请求的同步策略执行器；主机缓冲只在上一批回传完成后复用。"""

import time
from collections import Counter
from dataclasses import dataclass, field

import torch

from ..data import INPUT_KEYS, _collate
from ..runtime import autocast


@dataclass
class InferenceStats:
    """同一工作池共享累计数字，不持有模型或张量；采样线程独占写入。"""

    collation_seconds: float = 0.0
    execution_seconds: float = 0.0
    calls: int = 0
    requests: int = 0
    entities: int = 0
    candidates: int = 0
    entity_slots: int = 0
    candidate_slots: int = 0
    batch_sizes: Counter = field(default_factory=Counter)
    cuda_timing: dict = field(
        default_factory=lambda: {
            "batches": 0,
            "upload_ms": 0.0,
            "model_ms": 0.0,
            "download_ms": 0.0,
        }
    )

    def snapshot(self):
        return {
            "forwards": self.calls,
            "inference_requests": self.requests,
            "inference_seconds": self.execution_seconds,
            "collation_seconds": self.collation_seconds,
            "mean_batch": self.requests / max(1, self.calls),
            "batch_sizes": dict(self.batch_sizes),
            "entity_fill": self.entities / max(1, self.entity_slots),
            "candidate_fill": self.candidates / max(1, self.candidate_slots),
            "cuda_timing_sample": dict(self.cuda_timing),
        }


class PolicyInference:
    def __init__(self, model, device, precision="fp32", *, pinned=True, stats=None):
        self.model, self.device, self.precision = model, torch.device(device), precision
        if precision == "bf16" and self.device.type == "cuda":
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("本 CUDA 设备不支持 BF16，请使用 FP32 采样")
        self.pinned = pinned and self.device.type == "cuda"
        self.buffers = {}
        self.stats = stats if stats is not None else InferenceStats()

    def batch(self, examples):
        """输入来自一次完整协议校验，容量只增长；不缓存局面计算或改写输入帧。"""
        size = len(examples)
        entities = max(len(e["entities"]) for e in examples)
        actions = max(len(e["candidates"]) for e in examples)
        result = _collate(examples, INPUT_KEYS, buffers=self.buffers, pin_memory=self.pinned)
        self.stats.entities += sum(len(e["entities"]) for e in examples)
        self.stats.candidates += sum(len(e["candidates"]) for e in examples)
        self.stats.entity_slots += size * entities
        self.stats.candidate_slots += size * actions
        return result

    def __call__(self, examples):
        before = time.perf_counter()
        batch = self.batch(examples)
        self.stats.collation_seconds += time.perf_counter() - before
        before = time.perf_counter()
        # 每256批取一组设备事件，只作分账诊断；计时开销仍包含在总墙钟内。
        events = (
            [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            if (self.device.type == "cuda" and self.stats.calls % 256 == 0)
            else None
        )
        with torch.inference_mode(), autocast(self.device, self.precision):
            if events:
                events[0].record(torch.cuda.current_stream(self.device))
            batch = {k: v.to(self.device, non_blocking=self.pinned) for k, v in batch.items()}
            if events:
                events[1].record(torch.cuda.current_stream(self.device))
            # cpu() 是每批唯一结果同步；返回独立结果后才可重用主机输入。
            output = self.model.policy_logits(batch)
            if events:
                events[2].record(torch.cuda.current_stream(self.device))
            logits = output.cpu()
            if events:
                events[3].record(torch.cuda.current_stream(self.device))
                events[3].synchronize()
                self.stats.cuda_timing["batches"] += 1
                for i, key in enumerate(("upload_ms", "model_ms", "download_ms")):
                    self.stats.cuda_timing[key] += events[i].elapsed_time(events[i + 1])
        self.stats.execution_seconds += time.perf_counter() - before
        if not torch.isfinite(logits).all():
            raise ValueError("模型输出非有限")
        self.stats.calls += 1
        self.stats.requests += len(examples)
        self.stats.batch_sizes[len(examples)] += 1
        return logits
