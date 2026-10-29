"""可关闭的分段诊断；CUDA 流区间与主机时间分别汇总，不添加同步。"""

from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import time

import torch


class Profile:
    """每个实例只由所属线程使用；嵌套主机段同时记录包含和排除子段的时间。"""

    def __init__(self):
        self.enabled = os.environ.get("HAOJIE_PROFILE") == "1"
        self.annotate = False
        self.host_stages = {}
        self.stream_stages = {}
        self.sampling = {}
        self.pending = []
        self.stack = []

    def stage(self, name, cuda=False):
        if not self.enabled and not self.annotate:
            return nullcontext()
        return Stage(self, name, self.enabled and cuda)

    def select(self, kind, every, items=1):
        if not self.enabled:
            return False
        value = self.sampling.setdefault(
            kind,
            {
                "every": every,
                "total_calls": 0,
                "sample_calls": 0,
                "total_items": 0,
                "sample_items": 0,
            },
        )
        selected = value["total_calls"] % every == 0
        value["total_calls"] += 1
        value["total_items"] += items
        if selected:
            value["sample_calls"] += 1
            value["sample_items"] += items
        return selected

    def collect(self):
        if not self.pending:
            return
        waiting = []
        for name, start, end in self.pending:
            if end.query():
                value = self.stream_stages.setdefault(
                    name, {"count": 0, "stream_elapsed_seconds": 0.0}
                )
                value["count"] += 1
                value["stream_elapsed_seconds"] += start.elapsed_time(end) / 1000
            else:
                waiting.append((name, start, end))
        self.pending = waiting

    def report(self):
        self.collect()
        return {
            "host_stages": copy.deepcopy(self.host_stages),
            "stream_stages": copy.deepcopy(self.stream_stages),
            "sampling": copy.deepcopy(self.sampling),
            "pending_events": len(self.pending),
            "stream_time_scope": "sampled current-stream intervals, including host launch gaps; not kernel active time",
        }


class Stage:
    def __init__(self, profile, name, cuda):
        self.profile, self.name, self.cuda = profile, name, cuda
        self.child_wall = self.child_cpu = 0.0
        self.annotation = None

    def __enter__(self):
        if self.profile.annotate:
            self.annotation = torch.profiler.record_function("haojie." + self.name)
            self.annotation.__enter__()
        if self.profile.enabled:
            self.wall, self.cpu = time.perf_counter(), time.thread_time()
            self.profile.stack.append(self)
        if self.cuda:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
        return self

    def __exit__(self, *error):
        if self.cuda:
            self.end.record()
            self.profile.pending.append((self.name, self.start, self.end))
        if self.profile.enabled:
            wall, cpu = time.perf_counter() - self.wall, time.thread_time() - self.cpu
            self.profile.stack.pop()
            value = self.profile.host_stages.setdefault(
                self.name,
                {
                    "count": 0,
                    "inclusive_wall_seconds": 0.0,
                    "exclusive_wall_seconds": 0.0,
                    "inclusive_thread_cpu_seconds": 0.0,
                    "exclusive_thread_cpu_seconds": 0.0,
                },
            )
            value["count"] += 1
            value["inclusive_wall_seconds"] += wall
            value["exclusive_wall_seconds"] += max(0, wall - self.child_wall)
            value["inclusive_thread_cpu_seconds"] += cpu
            value["exclusive_thread_cpu_seconds"] += max(0, cpu - self.child_cpu)
            if self.profile.stack:
                self.profile.stack[-1].child_wall += wall
                self.profile.stack[-1].child_cpu += cpu
        if self.annotation is not None:
            self.annotation.__exit__(*error)


class TraceWindow:
    """独立短 trace：循环顶部第 64 次 step 开始，覆盖 32 个完整循环。"""

    def __init__(self, profile, device):
        self.profile = profile
        target = os.environ.get("HAOJIE_TRACE")
        self.path = Path(target) if target else None
        self.device = torch.device(device)
        self.calls = 0
        self.profiler = None
        self.started = None
        self.last_stats = None
        self.completed_loops = 0
        self.last_completed_iteration = None
        self.finished = False

    def step(self, stats):
        if self.path is None or self.finished:
            return
        self.calls += 1
        self.last_stats = copy.deepcopy(stats)
        if self.calls == 64:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.profiler = torch.profiler.profile(
                activities=activities,
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            self.profiler.start()
            self.profile.annotate = True
            self.started = {
                "iteration": self.calls,
                "wall_time": time.time(),
                "stats": self.last_stats,
            }
        elif self.calls == 96:
            self.close()

    def complete(self, stats):
        if self.profiler is not None:
            self.completed_loops += 1
            self.last_completed_iteration = self.calls
            self.last_stats = copy.deepcopy(stats)

    def close(self, stats=None):
        if self.profiler is None:
            return
        if stats is not None:
            self.last_stats = copy.deepcopy(stats)
        ended_at = time.time()
        self.profile.annotate = False
        self.profiler.stop()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.profiler.export_chrome_trace(str(self.path))
        metadata = {
            "start": self.started,
            "end": {
                "iteration": self.calls,
                "wall_time": ended_at,
                "stats": self.last_stats,
            },
            "completed_loops": self.completed_loops,
            "last_completed_iteration": self.last_completed_iteration,
            "incomplete_iteration": self.calls
            if self.completed_loops < 32 and self.last_completed_iteration != self.calls
            else None,
            "partial": self.completed_loops < 32,
            "activities": ["cpu", "cuda"] if self.device.type == "cuda" else ["cpu"],
            "scope": "independent trace; profiler stop includes its own flush",
        }
        self.path.with_suffix(".meta.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.profiler = None
        self.finished = True
