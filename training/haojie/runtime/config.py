"""单机资源与普通局训练配置。"""

import os
from dataclasses import dataclass
from pathlib import Path


def cpu_limit():
    affinity = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    # 容器配额优先于宿主逻辑核心数。
    path = Path("/sys/fs/cgroup/cpu.max")
    if path.exists():
        quota, period = path.read_text().split()[:2]
        if quota != "max":
            affinity = min(
                affinity, max(1, (int(quota) + int(period) - 1) // int(period))
            )
    return affinity


@dataclass
class Config:
    workers: int = 8
    games_per_worker: int = 8
    simulations: int = 64
    fast_simulations: int = 16
    full_fraction: float = 0.25
    depth: int = 16
    batch: int = 16
    inference_batch: int = 32
    sample_reuse: float = 4
    publish_updates: int = 16
    history_games: int = 64
    replay_games: int = 128
    replay_mib: int = 4096
    chunk_commands: int = 64
    max_commands: int = 20000
    max_ply: int = 50
    full_game_fraction: float = 0.25
    warmup_games: int = 8
    warmup_updates: int = 512
    calibration_games: int = 16
    bootstrap_after_games: int = 128
    bootstrap_weight: float = 0.25
    teacher_fraction: float = 0.1
    history_fraction: float = 0.2
    learning_rate: float = 0.0003
    device: str = "cuda"
    seed: int = 12345
    seconds: float = 0
    games: int = 0
    updates: int = 0
    save_seconds: float = 1800
    temperature: float = 1

    def validate(self):
        for key in (
            "workers",
            "games_per_worker",
            "depth",
            "batch",
            "inference_batch",
            "publish_updates",
            "history_games",
            "replay_games",
            "replay_mib",
            "chunk_commands",
            "max_commands",
        ):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} 必须为正")
        if min(self.simulations, self.fast_simulations) < 0:
            raise ValueError("搜索预算无效")
        if self.max_ply < 0 or not 0 <= self.full_game_fraction <= 1:
            raise ValueError("回合上限和完整局比例无效")
        if (
            self.calibration_games < 5
            or self.bootstrap_after_games < self.calibration_games
            or not 0 < self.bootstrap_weight <= 1
        ):
            raise ValueError("截断软目标配置无效")
        if (
            not 0 <= self.full_fraction <= 1
            or not 0 <= self.teacher_fraction + self.history_fraction <= 1
        ):
            raise ValueError("采样比例无效")
        if min(self.teacher_fraction, self.history_fraction, self.sample_reuse) < 0:
            raise ValueError("比例不能为负")
        if (
            min(
                self.seconds,
                self.games,
                self.updates,
                self.save_seconds,
                self.warmup_games,
                self.warmup_updates,
                self.temperature,
            )
            < 0
        ):
            raise ValueError("停止条件、预热局数和温度不能为负")
        if self.learning_rate <= 0:
            raise ValueError("学习率必须为正")
