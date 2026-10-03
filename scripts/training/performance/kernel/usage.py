"""只测量本轮 Python 与 Client 子进程；不调整模型、组批或调度策略。"""

import os
import threading
import time

from haojie_training.native.client import Client

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    class Memory(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "peak", "rss", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile"
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    memory = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    memory.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD]

    def read(process=None):
        handle = int(process._handle) if process else kernel.GetCurrentProcess()
        times = [wintypes.FILETIME() for _ in range(4)]
        cpu = 0.0
        if kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            cpu = sum((t.dwHighDateTime << 32) | t.dwLowDateTime for t in times[2:]) / 1e7
        info = Memory()
        info.cb = ctypes.sizeof(info)
        if not memory.GetProcessMemoryInfo(handle, ctypes.byref(info), info.cb):
            return cpu, 0, 0
        return cpu, info.rss, info.peak
else:
    import resource
    from pathlib import Path

    def read(process=None):
        try:
            fields = dict(line.split(":", 1) for line in Path(f"/proc/{process.pid if process else os.getpid()}/status").read_text().splitlines())
            return 0.0, int(fields.get("VmRSS", "0 kB").split()[0]) * 1024, int(fields.get("VmHWM", "0 kB").split()[0]) * 1024
        except (FileNotFoundError, ProcessLookupError):
            return 0.0, 0, 0


class Usage:
    def __init__(self):
        self.processes = {}
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.peak = self.single = 0
        self.child_cpu = 0.0
        self.result = None
        self.original_init = Client.__init__
        self.original_close = Client.close

    def start(self):
        self.started_cpu = time.process_time()
        if os.name != "nt":
            self.children_before = resource.getrusage(resource.RUSAGE_CHILDREN)

        def initialize(client, *args, **kwargs):
            self.original_init(client, *args, **kwargs)
            with self.lock:
                self.processes[client.process.pid] = client.process

        def close(client):
            self.original_close(client)
            with self.lock:
                process = self.processes.pop(client.process.pid, None)
                if process:
                    cpu, _, peak = read(process)
                    self.child_cpu += cpu
                    self.single = max(self.single, peak)

        Client.__init__, Client.close = initialize, close

        def monitor():
            while not self.stopped.is_set():
                with self.lock:
                    rows = [read(), *(read(p) for p in self.processes.values())]
                    self.peak = max(self.peak, sum(row[1] for row in rows))
                    self.single = max(self.single, *(row[2] for row in rows))
                self.stopped.wait(0.05)

        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        if self.result is not None:
            return self.result
        self.stopped.set()
        self.thread.join()
        Client.__init__, Client.close = self.original_init, self.original_close
        if os.name != "nt":
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            self.child_cpu = after.ru_utime + after.ru_stime - self.children_before.ru_utime - self.children_before.ru_stime
        own = time.process_time() - self.started_cpu
        self.result = {"cpu_seconds": own + self.child_cpu, "python_cpu_seconds": own,
                "engine_cpu_seconds": self.child_cpu, "sampled_peak_rss_bytes": self.peak,
                "peak_single_process_bytes": self.single, "rss_interval_ms": 50,
                "note": "自有进程总 RSS 每50ms采样；单进程高水位可能包含较早轮次，不等于精确并发峰值"}
        return self.result
