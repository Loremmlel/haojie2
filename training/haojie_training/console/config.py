"""控制台设置的默认值与输入边界；HTTP 修改和磁盘恢复共用同一份定义。"""

import psutil

METHOD = "decomposed-mc-q-v2"

# 每项依次为默认值、最小值、最大值；样本池默认容量再按启动时可用 RAM 调整。
_INTEGERS = {
    "environments": (4, 1, 16),
    "games_per_round": (4, 1, 128),
    "updates_per_round": (0, 0, 1000),
    "update_limit": (256, 1, 1000),
    "batch_size": (8, 1, 32),
    "classic_percent": (50, 0, 100),
    "history_percent": (50, 0, 100),
    "current_first_percent": (50, 0, 100),
    "pool_mib": (4096, 16, 4096),
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
    config["pool_mib"] = min(4096, max(64, int(psutil.virtual_memory().available / 4 / 2**20)))
    return {**config, "method": METHOD, "temperature": 0.25}


def validate(config):
    if set(config) != _INTEGERS.keys() | {"method", "temperature"}:
        raise ValueError("设置字段不匹配")
    for key, (_, low, high) in _INTEGERS.items():
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError(f"{key} 必须为 {low} 至 {high} 的整数")
    if config["method"] not in (METHOD, "sampled-action-imitation"):
        raise ValueError("未知训练方法")
    if (
        not isinstance(config["temperature"], (int, float))
        or not 0.05 <= config["temperature"] <= 2
    ):
        raise ValueError("温度必须为0.05至2")
