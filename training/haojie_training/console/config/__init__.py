"""控制台设置与服务器资源配置；HTTP 修改和磁盘恢复共用同一份输入边界。"""

import math

from ...runtime import default_precision, resolve_device
from .resources import available_memory, cpu_capacity

METHOD = "decomposed-mc-q-v2"

# 每项依次为默认值、最小值、最大值；资源相关初值由 defaults 统一按设备调整。
_INTEGERS = {
    "environments": (4, 1, 128),
    "games_per_round": (4, 1, 4096),
    "updates_per_round": (0, 0, 16384),
    "update_limit": (256, 1, 16384),
    "batch_size": (8, 1, 256),
    "inference_batch_size": (32, 1, 256),
    "classic_percent": (50, 0, 100),
    "history_percent": (50, 0, 100),
    "current_first_percent": (50, 0, 100),
    "pool_mib": (4096, 16, 131072),
    "save_seconds": (600, 10, 86400),
    "save_games": (0, 0, 100000),
    "eval_games": (16, 1, 100000),
    "eval_pairs": (1, 1, 20),
    "eval_seconds": (180, 2, 7200),
    "confirmation_pairs": (5, 5, 20),
    "max_commands": (20000, 1, 20000),
    "max_plies": (1000, 1, 1000),
    "history_every": (200, 1, 100000),
    "history_size": (4, 4, 8),
    "seed": (20261004, 1, 999999999),
}


def defaults(device="auto"):
    """GUI、cloud 与新建实验共用硬件初配；调用方最后合并用户显式设置。"""
    device = resolve_device(str(device))
    memory = available_memory()
    pool_mib = min(32768, max(64, memory // (4 * 2**20)))
    # 样本池与进程共享余量；另外留出模型、优化器和临时张量所需的起步空间。
    worker_memory = max(0, memory - pool_mib * 2**20 - 512 * 2**20)
    workers = min(128, max(1, math.ceil(cpu_capacity() * (1 if device.type == "cpu" else 2))))
    workers = min(workers, max(1, worker_memory // (256 * 2**20)))
    config = {key: default for key, (default, _, _) in _INTEGERS.items()}
    config.update(
        environments=workers,
        games_per_round=workers * 2,
        inference_batch_size=min(32, workers),
        pool_mib=pool_mib,
        update_limit=max(256, workers * 64),
        eval_games=max(64, workers * 4),
    )
    return {
        **config,
        "method": METHOD,
        "temperature": 0.25,
        "batch_wait_ms": 0.0,
        "sampling_precision": default_precision(device),
    }


def upgrade(config):
    """旧恢复点只补齐新增执行选项；原学习参数逐项保留并继续校验。"""
    result = {
        "inference_batch_size": 32,
        "batch_wait_ms": 0.0,
        "sampling_precision": "fp32",
        **config,
    }
    validate(result)
    return result


def validate(config):
    if set(config) != _INTEGERS.keys() | {
        "method",
        "temperature",
        "batch_wait_ms",
        "sampling_precision",
    }:
        raise ValueError("设置字段不匹配")
    for key, (_, low, high) in _INTEGERS.items():
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError(f"{key} 必须为 {low} 至 {high} 的整数")
    if config["method"] not in (METHOD, "sampled-action-imitation"):
        raise ValueError("未知训练方法")
    for key, low, high in (("temperature", 0.05, 2), ("batch_wait_ms", 0, 10)):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} 必须为 {low} 至 {high} 的有限数值")
    if config["sampling_precision"] not in {"fp32", "bf16"}:
        raise ValueError("采样精度必须为 fp32 或 bf16")
