"""有界 CUDA event 计时与批量分布；状态查看不等待仍在执行的 GPU 工作。"""

import time
from collections import deque

import torch


class Measurements:
    def __init__(self, device, metrics):
        self.device, self.metrics = device, metrics
        self.pending = deque()
        self.cpu_started = time.process_time()
        self.cpu_elapsed = metrics.get("cpu_seconds", 0.0)

    def begin(self, phase, count):
        self.collect()
        buckets = self.metrics.setdefault(f"{phase}_batch_histogram", {})
        bucket = str(1 << (max(1, count) - 1).bit_length())
        buckets[bucket] = buckets.get(bucket, 0) + 1
        if self.device.type != "cuda":
            return None
        # 不让未回收的事件变成另一条无界队列。
        if len(self.pending) >= 32:
            self.pending[0][2].synchronize()
            self.collect()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        return phase, start, end

    def end(self, handle):
        if handle is not None:
            handle[2].record()
            self.pending.append(handle)

    def collect(self, wait=False):
        if wait and self.pending:
            self.pending[-1][2].synchronize()
        while self.pending and self.pending[0][2].query():
            phase, start, end = self.pending.popleft()
            key = f"{phase}_gpu_seconds"
            self.metrics[key] = self.metrics.get(key, 0.0) + start.elapsed_time(end) / 1000
        self.metrics["cpu_seconds"] = self.cpu_elapsed + time.process_time() - self.cpu_started


def resident_bytes():
    """当前进程驻留字节，覆盖 Python 容器及分配器保留页。"""
    import os

    if os.name != "nt":
        from pathlib import Path

        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "peak_working",
                "working",
                "peak_paged",
                "paged",
                "peak_nonpaged",
                "nonpaged",
                "pagefile",
                "peak_pagefile",
            )
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.working
