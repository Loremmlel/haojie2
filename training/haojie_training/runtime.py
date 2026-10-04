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
        optimizer="auto",
        objective="sampled-action-imitation",
    ):
        if precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("未知训练精度")
        if not math.isfinite(value_weight) or value_weight < 0:
            raise ValueError("价值权重必须是有限非负数")
        self.value_weight = value_weight
        if objective not in {"sampled-action-imitation", "decomposed-mc-q-v1"}:
            raise ValueError("未知学习目标")
        self.objective = objective
        self.last_losses = {}
        self.device, self.precision = device, precision
        self.model = model.to(device).train()
        if optimizer not in {"auto", "foreach", "fused"}:
            raise ValueError("optimizer必须是auto、foreach或fused")
        self.requested_optimizer = optimizer
        # FP16沿用GradScaler可准确跳步的foreach路径，避免融合step钩子把溢出算成更新。
        fused = optimizer == "fused" or (
            optimizer == "auto" and device.type == "cuda" and precision != "fp16"
        )
        if fused and (device.type != "cuda" or precision == "fp16"):
            raise ValueError("融合AdamW仅用于CUDA FP32/BF16；FP16请使用foreach")
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.01, foreach=not fused, fused=fused
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
            output = self.model(batch)
            logits, value = output
            known = batch["value_mask"].float()
            value_loss = ((value - batch["value"]).square() * known).sum() / known.sum().clamp_min(
                1
            )
            if self.objective == "decomposed-mc-q-v1":
                # 分解节点已执行候选的 MC 收益回归；不是策略梯度，不需要回溯概率。
                chosen = logits.gather(1, batch["policy"].argmax(-1, keepdim=True)).squeeze(1)
                policy_loss = (
                    (chosen.tanh() - batch["value"]).square() * known
                ).sum() / known.sum().clamp_min(1)
                loss = policy_loss + self.value_weight * value_loss
            else:
                loss = policy_value_loss(output, batch, self.value_weight)
                policy_loss = loss - self.value_weight * value_loss
            self.last_losses = {
                "policy": float(policy_loss.detach()),
                "value": float(value_loss.detach()),
            }
        if not torch.isfinite(loss):
            raise FloatingPointError("非有限损失，未执行更新")
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
            "finite_optimizer": all(
                bool(torch.isfinite(value).all())
                for state in self.optimizer.state.values()
                for value in state.values()
                if isinstance(value, torch.Tensor)
            ),
            "scale": self.scaler.get_scale(),
        }

    def save(self, path: Path, metadata: dict) -> None:
        """CPU可加载的张量检查点；保存训练随机状态，不接受或保存游戏正式PRNG。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.checkpoint(metadata), temporary)
        temporary.replace(path)

    def checkpoint(self, metadata):
        """供统一配额出口使用；调用者必须在更新边界消费，不能并发修改参数。"""
        return {
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
            "execution": {
                "deterministic": torch.are_deterministic_algorithms_enabled(),
                "matmul_precision": torch.get_float32_matmul_precision(),
            },
        }

    def restore(self, path: Path, metadata: dict) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.restore_payload(payload, metadata)

    def restore_payload(self, payload, metadata):
        if (
            payload.get("format") != "haojie-checkpoint-v1"
            or payload.get("network") != NETWORK_VERSION
            or payload["config"] != asdict(self.model.config)
            or payload["metadata"] != metadata
            or payload["precision"] != self.precision
            or payload.get("value_weight", 1.0) != self.value_weight
        ):
            raise ValueError("检查点的模型、数据来源、精度或损失权重与当前训练不一致")
        saved_fused = payload["optimizer"]["param_groups"][0].get("fused", False)
        execution = {
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "matmul_precision": torch.get_float32_matmul_precision(),
        }
        if payload.get("execution", execution) != execution:
            raise ValueError("续训确定性/矩阵精度执行设置不匹配")
        saved_mode = "fused" if saved_fused else "foreach"
        if self.requested_optimizer != "auto" and self.requested_optimizer != saved_mode:
            raise ValueError("续训优化器执行方式不匹配；使用auto恢复检查点模式")
        if saved_fused and (self.device.type != "cuda" or self.precision == "fp16"):
            raise ValueError("此融合优化器检查点须使用原CUDA执行模式恢复")
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
