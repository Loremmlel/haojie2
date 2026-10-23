"""已校验原生请求的策略执行器；异步票据独占输入缓冲和输出，回收后才能复用。"""

import queue
import threading
import time
from collections import Counter
from concurrent.futures import Future
from dataclasses import dataclass, field

import torch

from ..data import INPUT_KEYS, _collate, bf16_tensors
from ..model import PolicyValueNet
from .packed import PackedPolicy, supports_packed


@dataclass
class InferenceStats:
    """同一工作池共享累计数字，不持有模型或张量；采样线程独占写入。"""

    collation_seconds: float = 0.0
    execution_seconds: float = 0.0
    calls: int = 0
    packed_calls: int = 0
    async_calls: int = 0
    async_overlaps: int = 0
    async_discarded: int = 0
    inflight: int = 0
    submit_seconds: float = 0.0
    wait_seconds: float = 0.0
    background_calls: int = 0
    background_launch_seconds: float = 0.0
    background_wait_seconds: float = 0.0
    requests: int = 0
    entities: int = 0
    candidates: int = 0
    scored_candidates: int = 0
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
            "packed_forwards": self.packed_calls,
            "async_forwards": self.async_calls,
            "async_overlaps": self.async_overlaps,
            "async_discarded": self.async_discarded,
            "async_submit_seconds": self.submit_seconds,
            "async_wait_seconds": self.wait_seconds,
            "background_forwards": self.background_calls,
            "background_launch_seconds": self.background_launch_seconds,
            "background_wait_seconds": self.background_wait_seconds,
            "inference_requests": self.requests,
            "inference_seconds": self.execution_seconds,
            # 异步只加提交和回收等待；设备事件与组批可重叠，不能再简单相加。
            "inference_timing": "host-submit-plus-background-launch-plus-device-wait"
            if self.background_calls
            else "synchronous-wall-or-async-submit-plus-wait",
            "collation_seconds": self.collation_seconds,
            "mean_batch": self.requests / max(1, self.calls),
            "mean_entities": self.entities / max(1, self.requests),
            "mean_candidates": self.candidates / max(1, self.requests),
            "mean_scored_candidates": self.scored_candidates / max(1, self.requests),
            "batch_sizes": dict(self.batch_sizes),
            "entity_fill": self.entities / max(1, self.entity_slots),
            "candidate_fill": self.candidates / max(1, self.candidate_slots),
            "candidates": self.candidates,
            "scored_candidates": self.scored_candidates,
            "cuda_timing_sample": dict(self.cuda_timing),
        }


@dataclass(eq=False)
class InferenceTicket:
    """调用者持有独立 CPU 结果；owners 保证 DMA 结束前输入和设备输出仍存活。"""

    buffer: int
    slots: tuple
    event: object
    logits: torch.Tensor
    owners: tuple
    timing: object = None
    launch: object = None

    def ready(self):
        if self.launch is not None:
            if not self.launch.done():
                return False
            self.launch.result()
        return self.event.query()


class ForwardWorkers:
    """常驻双流发射；窗口先排空，关闭工作池时才停止线程。"""

    def __init__(self, device, count=2):
        self.device, self.next = device, 0
        self.queues, self.threads = [], []
        self.closed = False
        try:
            for index in range(count):
                incoming, started = queue.SimpleQueue(), Future()
                stream = torch.cuda.Stream(device=device)
                thread = threading.Thread(
                    target=self.run, args=(incoming, stream, started), name=f"haojie-launch-{index}"
                )
                self.queues.append(incoming)
                self.threads.append(thread)
                thread.start()
                started.result(timeout=30)
        except BaseException:
            self.close()
            raise

    def run(self, incoming, stream, started):
        try:
            with torch.cuda.device(self.device), torch.cuda.stream(stream):
                started.set_result(None)
                while item := incoming.get():
                    future, function, args = item
                    if future.set_running_or_notify_cancel():
                        try:
                            future.set_result(function(*args))
                        except BaseException as error:
                            future.set_exception(error)
        except BaseException as error:
            if not started.done():
                started.set_exception(error)
            else:
                raise

    def submit(self, function, *args):
        if self.closed:
            raise RuntimeError("GPU 发射线程已关闭")
        future = Future()
        self.queues[self.next].put((future, function, args))
        self.next = (self.next + 1) % len(self.queues)
        return future

    def close(self):
        if self.closed:
            return
        self.closed = True
        for incoming in self.queues:
            incoming.put(None)
        for thread in self.threads:
            if thread.ident is not None:
                thread.join()


class PolicyInference:
    def __init__(self, model, device, *, pinned=True, stats=None):
        self.model, self.device, self.precision = model, torch.device(device), "bf16"
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported(including_emulation=False):
                    raise ValueError("本 CUDA 设备不支持 BF16")
        self.pinned = pinned and self.device.type == "cuda"
        self.buffers = {}
        self.stats = stats if stats is not None else InferenceStats()
        self.packed = (
            PackedPolicy(model, self.buffers, self.pinned)
            if supports_packed(model, self.device)
            else None
        )
        self._alternate_packed = None
        self._tickets = [None, None]
        self._slots = set()
        self.launcher, self.weights_ready = None, None

    def set_launcher(self, launcher):
        if any(ticket is not None for ticket in self._tickets):
            raise RuntimeError("推理未排空时不能切换发射器")
        self.launcher, self.weights_ready = launcher, None

    @property
    def supports_packed_batches(self):
        return self.packed is not None

    @property
    def supports_async(self):
        # 紧凑CUDA后端共用双缓冲；CPU/XPU及研究模型沿用同步路径。
        return self.packed is not None

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

    def readout(self, examples, batch):
        """CPU 上收集末层实际使用的实体；完整前层和 K/V 不裁剪，研究模型不改路径。"""
        if type(self.model) is not PolicyValueNet or batch["entities"].shape[1] < 32:
            return None
        positions, remaps = [], []
        for row in examples:
            count = len(row["sources"])
            pointers = torch.cat(
                (torch.zeros(1, dtype=torch.long), row["sources"] + 1, row["targets"] + 1)
            )
            unique, inverse = torch.unique(pointers, sorted=True, return_inverse=True)
            positions.append(unique)
            remaps.append((inverse[1 : 1 + count], inverse[1 + count :]))
        # 查询几乎全宽时沿用原前向，避免索引开销抵消收益。
        width = max(len(p) for p in positions)
        if width * 4 >= (batch["entities"].shape[1] + 1) * 3:
            return None
        indices = torch.zeros((len(examples), width), dtype=torch.long, pin_memory=self.pinned)
        sources = torch.zeros(batch["sources"].shape, dtype=torch.long, pin_memory=self.pinned)
        targets = torch.zeros_like(sources, pin_memory=self.pinned)
        for i, (position, (source, target)) in enumerate(zip(positions, remaps)):
            indices[i, : len(position)] = position
            sources[i, : len(source)] = source
            targets[i, : len(target)] = target
        return indices, sources, targets

    def _prepare(self, examples, packed):
        before = time.perf_counter()
        if packed is None or getattr(packed, "operator", None) is None:
            for row in examples:
                if hasattr(row, "ensure_validated"):
                    row.ensure_validated()
        if packed is not None:
            batch, layout = packed.batch(examples)
            readout = None
            self.stats.entities += layout["entities"]
            self.stats.entity_slots += layout["entities"]
            self.stats.candidates += layout["candidates"]
            self.stats.scored_candidates += (
                layout["scored_candidates"] if packed.native_dispatch else layout["candidate_slots"]
            )
            self.stats.candidate_slots += layout["candidate_slots"]
        else:
            examples = [
                bf16_tensors(
                    {
                        key: value if isinstance(value, torch.Tensor) else torch.from_numpy(value)
                        for key, value in row.items()
                    }
                )
                for row in examples
            ]
            layout = None
            batch = self.batch(examples)
            readout = self.readout(examples, batch)
        self.stats.collation_seconds += time.perf_counter() - before
        return batch, layout, readout

    def _events(self):
        # 每256批取一组设备事件，只作分账诊断；计时开销仍包含在总墙钟内。
        return (
            [torch.cuda.Event(enable_timing=True, blocking=i == 3) for i in range(4)]
            if (self.device.type == "cuda" and self.stats.calls % 256 == 0)
            else None
        )

    def _forward(self, batch, layout, readout, packed, events):
        with torch.no_grad():
            if events:
                events[0].record(torch.cuda.current_stream(self.device))
            # 原生评分只读取遮罩的输出形状，有效位置已在 CPU 排好，无需上传遮罩。
            batch = {
                k: v
                if k == "candidate_mask" and packed is not None and packed.native_dispatch
                else v.to(self.device, non_blocking=self.pinned)
                for k, v in batch.items()
            }
            if readout is not None:
                readout = tuple(v.to(self.device, non_blocking=self.pinned) for v in readout)
            if events:
                events[1].record(torch.cuda.current_stream(self.device))
            if packed is not None:
                output = packed.forward(batch, layout)
            else:
                output = (
                    self.model.policy_logits(batch)
                    if readout is None
                    else self.model.policy_readout(batch, *readout)
                )
            if events:
                events[2].record(torch.cuda.current_stream(self.device))
        return output, batch, readout

    def _record_timing(self, events):
        if events:
            self.stats.cuda_timing["batches"] += 1
            for i, key in enumerate(("upload_ms", "model_ms", "download_ms")):
                self.stats.cuda_timing[key] += events[i].elapsed_time(events[i + 1])

    def _count(self, examples):
        self.stats.calls += 1
        self.stats.packed_calls += self.packed is not None
        self.stats.requests += len(examples)
        self.stats.batch_sizes[len(examples)] += 1

    def __call__(self, examples):
        if any(ticket is not None for ticket in self._tickets):
            raise RuntimeError("同步前向前必须回收或关闭异步推理")
        batch, layout, readout = self._prepare(examples, self.packed)
        before = time.perf_counter()
        events = self._events()
        output, _, _ = self._forward(batch, layout, readout, self.packed, events)
        logits = output.cpu()
        if events:
            events[3].record(torch.cuda.current_stream(self.device))
            events[3].synchronize()
            self._record_timing(events)
        self.stats.execution_seconds += time.perf_counter() - before
        if not torch.isfinite(logits).all():
            raise ValueError("模型输出非有限")
        self._count(examples)
        return logits

    def submit(self, examples, *, slots):
        """最多两组在途；槽位身份必须包含进程代次，不能用请求 ID 绕过独占。"""
        if not self.supports_async:
            raise ValueError("此模型或设备仅支持同步推理")
        slots = tuple(slots)
        if not examples or len(slots) != len(examples) or len(set(slots)) != len(slots):
            raise ValueError("每个请求必须对应独立槽位")
        if self._slots.intersection(slots):
            raise ValueError("同一槽位已有在途推理")
        try:
            index = self._tickets.index(None)
        except ValueError:
            raise RuntimeError("两组推理均在途，须先回收一组") from None
        if index and self._alternate_packed is None:
            self._alternate_packed = PackedPolicy(self.model, {}, self.pinned)
        packed = self._alternate_packed if index else self.packed
        batch, layout, readout = self._prepare(examples, packed)
        before = time.perf_counter()
        stream = torch.cuda.current_stream(self.device)
        events = self._events()
        launcher = getattr(self, "launcher", None)
        if launcher is not None and self.weights_ready is not None and packed.native_dispatch:
            logits = torch.empty(
                batch["candidate_mask"].shape, dtype=torch.bfloat16, device="cpu", pin_memory=True
            )
            event = events[3] if events else torch.cuda.Event(blocking=True)
            launch = launcher.submit(
                self._launch,
                batch,
                layout,
                readout,
                packed,
                events,
                logits,
                event,
                self.weights_ready,
            )
            owners = (batch,)
            self.stats.background_calls += 1
        else:
            launch = None
            try:
                output, device_batch, device_readout = self._forward(
                    batch, layout, readout, packed, events
                )
                with torch.inference_mode():
                    logits = torch.empty(
                        output.shape, dtype=output.dtype, device="cpu", pin_memory=True
                    )
                    logits.copy_(output, non_blocking=True)
                event = events[3] if events else torch.cuda.Event(blocking=True)
                event.record(stream)
            except BaseException:
                # 提交中断也必须等已入队拷贝结束，才能再次写主机缓冲。
                stream.synchronize()
                raise
            owners = (batch, device_batch, device_readout, output)
            if launcher is not None and packed.native_dispatch:
                # 首批确认当前权重已就绪；其他流继续等待同一发布边界。
                self.weights_ready = event
        ticket = InferenceTicket(index, slots, event, logits, owners, events, launch)
        # 当前与历史模型共用统计，跨执行器的在途工作同样计入重叠。
        self.stats.async_overlaps += self.stats.inflight > 0
        self.stats.inflight += 1
        self._tickets[index] = ticket
        self._slots.update(slots)
        elapsed = time.perf_counter() - before
        self.stats.submit_seconds += elapsed
        self.stats.execution_seconds += elapsed
        self.stats.async_calls += 1
        self._count(examples)
        return ticket

    def _launch(self, batch, layout, readout, packed, events, logits, event, weights_ready):
        began = time.perf_counter()
        stream = torch.cuda.current_stream(self.device)
        try:
            stream.wait_event(weights_ready)
            output, device_batch, device_readout = self._forward(
                batch, layout, readout, packed, events
            )
            with torch.inference_mode():
                logits.copy_(output, non_blocking=True)
            event.record(stream)
            launched = time.perf_counter() - began
            before_wait = time.perf_counter()
            event.synchronize()
            return (
                (device_batch, device_readout, output),
                launched,
                time.perf_counter() - before_wait,
            )
        except BaseException:
            stream.synchronize()
            raise

    def _release(self, ticket):
        if self._tickets[ticket.buffer] is not ticket:
            raise ValueError("推理票据过期、重复回收或属于其他执行器")
        try:
            if ticket.launch is not None:
                before = time.perf_counter()
                owners, elapsed, device_wait = ticket.launch.result()
                ticket.owners += owners
                self.stats.background_wait_seconds += time.perf_counter() - before
                self.stats.background_launch_seconds += elapsed
                self.stats.wait_seconds += device_wait
                self.stats.execution_seconds += elapsed + device_wait
            before = time.perf_counter()
            ticket.event.synchronize()
            elapsed = time.perf_counter() - before
            self.stats.wait_seconds += elapsed
            self.stats.execution_seconds += elapsed
            self._record_timing(ticket.timing)
        finally:
            self._tickets[ticket.buffer] = None
            self._slots.difference_update(ticket.slots)
            self.stats.inflight -= 1
            ticket.owners, ticket.launch = (), None

    def finish(self, ticket):
        self._release(ticket)
        if not torch.isfinite(ticket.logits).all():
            raise ValueError("模型输出非有限")
        return ticket.logits

    def close(self):
        """结束/失败/换权重前排空；取消的输出不再进入动作或训练样本。"""
        error = None
        for ticket in tuple(self._tickets):
            if ticket is not None:
                self.stats.async_discarded += 1
                try:
                    self._release(ticket)
                except BaseException as caught:
                    error = error or caught
        self.launcher, self.weights_ready = None, None
        if error is not None:
            raise error
