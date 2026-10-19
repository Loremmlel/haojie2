"""读取进程实际 CPU/RAM 配额，并对本次训练进程树采样；不预留或更改系统资源。"""

import math
import os
import time
from functools import lru_cache
from pathlib import Path

import psutil
import torch


def _read(path):
    try:
        return path.read_text().strip()
    except OSError:
        return ""


@lru_cache(maxsize=2)
def _cgroups(controller):
    """覆盖可见的当前组和父组；兼容容器将组本身挂到 cgroup 根目录的布局。"""
    root = Path("/sys/fs/cgroup")
    paths = []
    for line in _read(Path("/proc/self/cgroup")).splitlines():
        _, controllers, relative = line.split(":", 2)
        if controllers and controller not in controllers.split(","):
            continue
        bases = [root] if not controllers else [root / controllers, root / controller]
        for base in bases:
            if not base.is_dir():
                continue
            current = (base / relative.lstrip("/")).resolve()
            if not current.is_relative_to(base) or not current.is_dir():
                current = base
            while True:
                paths.append(current)
                if current == base:
                    break
                current = current.parent
    return tuple(dict.fromkeys(paths))


def cpu_capacity():
    """返回 CPU 时间预算；逻辑 CPU、调度亲和性和各层带宽配额取最小值。"""
    capacity = float(os.cpu_count() or 1)
    if hasattr(os, "sched_getaffinity"):
        capacity = min(capacity, len(os.sched_getaffinity(0)))
    for path in _cgroups("cpu"):
        quota = _read(path / "cpu.max").split()
        if not quota:
            quota = [_read(path / "cpu.cfs_quota_us"), _read(path / "cpu.cfs_period_us")]
        if len(quota) == 2 and all(value.lstrip("-").isdigit() for value in quota):
            budget, period = map(int, quota)
            if budget > 0 and period > 0:
                capacity = min(capacity, budget / period)
    return capacity


def _memory():
    memory = psutil.virtual_memory()
    total, available = memory.total, memory.available
    for path in _cgroups("memory"):
        limit = _read(path / "memory.max") or _read(path / "memory.limit_in_bytes")
        used = _read(path / "memory.current") or _read(path / "memory.usage_in_bytes")
        if limit.isdigit() and used.isdigit():
            limit, used = int(limit), int(used)
            stats = dict(line.split() for line in _read(path / "memory.stat").splitlines())
            # 冷文件缓存可回收；扣除脏页/回写量，避免将待写数据视作即时余量。
            inactive = int(stats.get("total_inactive_file", stats.get("inactive_file", 0)))
            dirty = int(stats.get("total_dirty", stats.get("file_dirty", 0)))
            writeback = int(stats.get("total_writeback", stats.get("file_writeback", 0)))
            reclaimable = max(0, min(used, inactive) - dirty - writeback)
            total = min(total, limit)
            available = min(available, limit, max(0, limit - used + reclaimable))
    return total, available


def available_memory():
    return _memory()[1]


def hardware():
    total, available = _memory()
    return {
        "logical_cpus": os.cpu_count(),
        "cpu_capacity": cpu_capacity(),
        "ram_total": total,
        "ram_available": available,
        "gpus": [
            {"index": i, "name": p.name, "bytes": p.total_memory}
            for i in range(torch.cuda.device_count())
            for p in [torch.cuda.get_device_properties(i)]
        ],
    }


def server_defaults(device="auto"):
    """服务器起步配置按 CPU/RAM 选槽；推理和更新批量仍可单独调节。"""
    from . import defaults

    config = defaults()
    cuda = str(device).startswith("cuda") or device == "auto" and torch.cuda.is_available()
    memory = available_memory()
    workers = min(32, max(1, math.ceil(cpu_capacity() * (2 if cuda else 1))))
    workers = min(workers, max(1, memory // (256 * 2**20)))
    config.update(
        environments=workers,
        games_per_round=workers * 2,
        inference_batch_size=min(32, workers),
        pool_mib=min(32768, max(64, memory // (4 * 2**20))),
        update_limit=max(256, workers * 64),
        eval_games=max(64, workers * 4),
    )
    return config


def _processes(process):
    result = {}
    if process is None:
        return result
    try:
        members = [process, *process.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return result
    for member in members:
        try:
            cpu = member.cpu_times()
            result[(member.pid, member.create_time())] = {
                "pid": member.pid,
                "name": member.name(),
                "cpu": cpu.user + cpu.system,
                "reaped_cpu": getattr(cpu, "children_user", 0) + getattr(cpu, "children_system", 0),
                "rss": member.memory_info().rss,
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return result


def monitor(controller):
    """CPU 按进程树时间差除以配额；系统均值单列，不调用训练器或复制张量。"""
    try:
        process = psutil.Process()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        process = None
    previous, began = _processes(process), time.monotonic()
    psutil.cpu_percent()
    nvml = None
    if torch.cuda.is_available():
        try:
            import pynvml

            pynvml.nvmlInit()
            nvml = pynvml
        except Exception:
            pass
    try:
        while not controller.shutdown.wait(2):
            now = time.monotonic()
            current = _processes(process)
            workers = []
            for key, member in current.items():
                delta = member["cpu"] - previous.get(key, {"cpu": 0})["cpu"]
                workers.append({
                    "pid": member["pid"], "name": member["name"],
                    "cpu_cores": max(0, delta) / (now - began), "rss": member["rss"],
                })
            # Linux 父进程累积已回收子进程时间；合计求差，避免短局退出丢账或重复记账。
            total = sum(row["cpu"] + row["reaped_cpu"] for row in current.values())
            prior = sum(row["cpu"] + row["reaped_cpu"] for row in previous.values())
            cores = max(0, total - prior) / (now - began)
            capacity = cpu_capacity()
            device = controller.device
            cuda = device.type == "cuda" and torch.cuda.is_available()
            telemetry = {
                "process_tree_available": bool(current),
                "rss": sum(row["rss"] for row in workers) if current else None,
                "cpu_cores": cores if current else None,
                "cpu_capacity": capacity,
                "cpu_percent": cores / capacity * 100 if current else None,
                "system_cpu_percent": psutil.cpu_percent(),
                "available_ram": available_memory(),
                "workers": max(0, len(workers) - 1) if current else None,
                "processes": workers,
                "gpu_allocated": torch.cuda.memory_allocated(device) if cuda else None,
                "gpu_reserved": torch.cuda.memory_reserved(device) if cuda else None,
                "gpu_utilization": None,
            }
            if cuda and nvml:
                try:
                    properties = torch.cuda.get_device_properties(device)
                    identity = getattr(properties, "uuid", None)
                    handle = (
                        nvml.nvmlDeviceGetHandleByUUID(str(identity)) if identity
                        else nvml.nvmlDeviceGetHandleByIndex(device.index or 0)
                    )
                    telemetry["gpu_utilization"] = nvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    telemetry["gpu_total"] = nvml.nvmlDeviceGetMemoryInfo(handle).total
                except Exception:
                    pass
            controller.telemetry = telemetry
            previous, began = current, now
    finally:
        if nvml:
            nvml.nvmlShutdown()
