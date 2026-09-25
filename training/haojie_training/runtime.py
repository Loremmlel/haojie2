"""共用训练步与检查点；FP32参数/Adam状态、可选AMP计算，异常不冒充成功更新。"""

import math
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch

from .model import NETWORK_VERSION, ModelConfig, PolicyValueNet, policy_value_loss


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("XPU不可用；检查XPU版PyTorch和Intel驱动，不静默改测CPU")
    if device.type not in {"cpu", "xpu", "cuda"}:
        raise ValueError("设备仅支持cpu、xpu、cuda")
    return device


def synchronize(device: torch.device) -> None:
    if device.type != "cpu":
        getattr(torch, device.type).synchronize(device)


def autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if precision not in {"bf16", "fp16"}:
        raise ValueError("precision必须是fp32、bf16或fp16")
    return torch.autocast(
        device.type, dtype=torch.bfloat16 if precision == "bf16" else torch.float16
    )


class Trainer:
    """
    执行一个策略/价值优化步；输入须先通过CPU边界校验。
    基准和普通训练复用同一实现，含清梯度、前向、反向、裁剪、AdamW及缩放更新。
    FP16使用动态GradScaler；记录真正的optimizer更新数，溢出跳步不算有效吞吐。
    同步与数值健康检查由调用方在计时边界执行，避免逐张量GPU同步影响测量。
    """

    def __init__(
        self,
        model: PolicyValueNet,
        device: torch.device,
        precision="fp32",
        lr=3e-4,
        value_weight=1.0,
    ):
        if precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("未知训练精度")
        if not math.isfinite(value_weight) or value_weight < 0:
            raise ValueError("价值权重必须是有限非负数")
        self.value_weight = value_weight
        self.device, self.precision = device, precision
        self.model = model.to(device).train()
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.01, foreach=True
        )
        self.scaler = torch.amp.GradScaler(
            device.type, enabled=precision == "fp16", init_scale=1024
        )
        self.steps = 0
        self.updates = 0
        self.optimizer.register_step_post_hook(lambda *_: self._updated())

    def _updated(self):
        self.updates += 1

    def step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device, self.precision):
            loss = policy_value_loss(self.model(batch), batch, self.value_weight)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, foreach=True)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.steps += 1
        return loss.detach()

    def health(self) -> dict:
        parameters = list(self.model.parameters())
        gradients = [p.grad for p in parameters if p.grad is not None]
        return {
            "finite_parameters": bool(
                torch.stack([torch.isfinite(p).all() for p in parameters]).all()
            ),
            "finite_gradients": bool(gradients)
            and bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()),
            "scale": self.scaler.get_scale(),
        }

    def save(self, path: Path, metadata: dict) -> None:
        """CPU可加载的张量检查点；保存训练随机状态，不接受或保存游戏正式PRNG。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "haojie-checkpoint-v1",
            "network": NETWORK_VERSION,
            "config": asdict(self.model.config),
            "metadata": metadata,
            "precision": self.precision,
            "value_weight": self.value_weight,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "steps": self.steps,
            "updates": self.updates,
            "rng_cpu": torch.get_rng_state(),
            "rng_device": None
            if self.device.type == "cpu"
            else getattr(torch, self.device.type).get_rng_state(self.device),
            "device_type": self.device.type,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def restore(self, path: Path, metadata: dict) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("format") != "haojie-checkpoint-v1"
            or payload.get("network") != NETWORK_VERSION
            or payload["config"] != asdict(self.model.config)
            or payload["metadata"] != metadata
            or payload["precision"] != self.precision
            or payload.get("value_weight", 1.0) != self.value_weight
        ):
            raise ValueError("检查点的模型、数据来源、精度或损失权重与当前训练不一致")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scaler.load_state_dict(payload["scaler"])
        self.steps, self.updates = payload["steps"], payload["updates"]
        torch.set_rng_state(payload["rng_cpu"])
        if self.device.type != "cpu" and payload["device_type"] == self.device.type:
            getattr(torch, self.device.type).set_rng_state(payload["rng_device"], self.device)


def checkpoint_config(path: Path) -> ModelConfig:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "haojie-checkpoint-v1" or payload.get("network") != NETWORK_VERSION:
        raise ValueError("检查点网络版本不兼容；旧骨架检查点需用原版本代码读取")
    return ModelConfig(**payload["config"])
