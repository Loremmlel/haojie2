"""控制台设置与服务器资源配置；HTTP 修改和磁盘恢复共用同一份输入边界。"""

import math

from .resources import available_memory

METHOD = "decomposed-mc-q-v2"

# 每项依次为默认值、最小值、最大值；样本池默认容量再按启动时可用 RAM 调整。
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


def defaults():
    config = {key: default for key, (default, _, _) in _INTEGERS.items()}
    config["pool_mib"] = min(4096, max(64, available_memory() // (4 * 2**20)))
    return {
        **config,
        "method": METHOD,
        "temperature": 0.25,
        "batch_wait_ms": 0.0,
        "sampling_precision": "fp32",
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
        "method", "temperature", "batch_wait_ms", "sampling_precision",
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
