"""共用训练步与检查点；FP32参数/Adam状态、可选AMP计算，异常不冒充成功更新。"""

import math
from collections import defaultdict
from contextlib import nullcontext
from copy import copy
from dataclasses import asdict
from pathlib import Path

import torch

from .model import NETWORK_VERSION, ModelConfig, PolicyValueNet, policy_value_loss


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用；请在新建训练中选择CPU或安装CUDA版PyTorch")
    if device.type == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("XPU不可用；检查XPU版PyTorch和Intel驱动，不静默改测CPU")
    if device.type not in {"cpu", "xpu", "cuda"}:
        raise ValueError("设备仅支持cpu、xpu、cuda")
    return device


def synchronize(device: torch.device) -> None:
    if device.type != "cpu":
        getattr(torch, device.type).synchronize(device)


def cpu_state(value):
    """在调用者独占学习状态的边界复制到 CPU；整棵状态只同步一次设备。

    模型及 Adam 状态的许多小张量先排队传回，全部完成后才交给哈希或序列化。
    CPU 张量保持原引用，调用者仍须在下一次更新前消费此状态。
    """
    devices = set()

    def transfer(item):
        if isinstance(item, torch.Tensor):
            if item.device.type != "cpu":
                devices.add(item.device)
            return item.detach().to("cpu", non_blocking=True)
        if isinstance(item, dict):
            result = copy(item)
            result.update((key, transfer(child)) for key, child in item.items())
            return result
        if isinstance(item, list):
            return [transfer(child) for child in item]
        if isinstance(item, tuple):
            return tuple(transfer(child) for child in item)
        return item

    result = transfer(value)
    for device in devices:
        synchronize(device)
    return result


def autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if precision not in {"bf16", "fp16"}:
        raise ValueError("precision必须是fp32、bf16或fp16")
    return torch.autocast(
        device.type, dtype=torch.bfloat16 if precision == "bf16" else torch.float16
    )


def _finite(tensors):
    """每个设备只取回一次检查结果；CPU 步数不与 GPU 状态混叠。"""
    checks = defaultdict(list)
    for tensor in tensors:
        checks[tensor.device].append(torch.isfinite(tensor).all())
    return all(bool(torch.stack(values).all()) for values in checks.values())


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
        if objective not in {
            "sampled-action-imitation",
            "decomposed-mc-q-v1",
            "decomposed-mc-q-v2",
        }:
            raise ValueError("未知学习目标")
        self.objective = objective
        self._last_losses = ()
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

    @property
    def last_losses(self):
        """只在报告边界同步日志标量，避免每步分别等待两次 GPU。"""
        if not self._last_losses:
            return {}
        values = torch.stack(self._last_losses).cpu().tolist()
        return dict(zip(("policy", "value"), values))

    def _loss(self, batch, denominator=None):
        mc = self.objective.startswith("decomposed-mc-q-")
        if mc:
            chosen, value = self.model.selected_value(batch, batch["policy"].argmax(-1))
        else:
            output = self.model(batch)
            _, value = output
        known = batch["value_mask"].float()
        if denominator is None:
            denominator = known.sum().clamp_min(1)
        value_loss = ((value - batch["value"]).square() * known).sum() / denominator
        if mc:
            policy_loss = ((chosen.tanh() - batch["value"]).square() * known).sum() / denominator
            loss = policy_loss + self.value_weight * value_loss
        else:
            loss = policy_value_loss(output, batch, self.value_weight)
            policy_loss = loss - self.value_weight * value_loss
        return loss, policy_loss, value_loss

    def step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.step_batches([batch])

    def step_batches(self, batches):
        """同一步的变长子批：整步归一化、逐组反向、只裁剪和更新一次。

        样本由控制器预先抽定；分组不消耗随机数。每组释放计算图后才处理下一组，
        保留 FP16 溢出跳步语义。同形一步不额外计算整组归一化分母。
        """
        if not batches:
            raise ValueError("没有可训练子批")
        denominator = None
        if len(batches) > 1:
            if not self.objective.startswith("decomposed-mc-q-"):
                raise ValueError("变长子批当前仅用于 MC-Q 终局回归")
            denominator = (
                torch.stack([b["value_mask"].float().sum() for b in batches]).sum().clamp_min(1)
            )
        self.optimizer.zero_grad(set_to_none=True)
        totals = None
        for batch in batches:
            with autocast(self.device, self.precision):
                losses = self._loss(batch, denominator)
                loss = losses[0]
            if not torch.isfinite(loss):
                raise FloatingPointError("非有限损失，未执行更新")
            self.scaler.scale(loss).backward()
            detached = tuple(value.detach() for value in losses)
            totals = detached if totals is None else tuple(a + b for a, b in zip(totals, detached))
            # 仅累计日志标量，不把上一子批计算图保留到下一次前向。
            del losses, loss
        self._last_losses = totals[1:]
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, foreach=True)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.steps += 1
        return totals[0]

    def health(self) -> dict:
        parameters = list(self.model.parameters())
        gradients = [p.grad for p in parameters if p.grad is not None]
        return {
            "finite_parameters": _finite(parameters),
            "finite_gradients": bool(gradients) and _finite(gradients),
            "finite_optimizer": _finite(
                value
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
        return cpu_state(
            {
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
        )

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


def read_checkpoint(path: Path) -> tuple[ModelConfig, dict]:
    """一次读取并校验正式检查点；配置和权重始终来自同一个磁盘快照。"""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and payload.get("format") == "haojie-stream-recovery-v2":
        payload = payload.get("trainer")
    if (
        not isinstance(payload, dict)
        or payload.get("format") not in {"haojie-checkpoint-v1", "haojie-stream-weights-v1"}
        or payload.get("network") != NETWORK_VERSION
        or any(not isinstance(payload.get(key), dict) for key in ("config", "metadata", "model"))
    ):
        raise ValueError("检查点网络版本不兼容；旧骨架检查点需用原版本代码读取")
    return ModelConfig(**payload["config"]), payload


def checkpoint_config(path: Path) -> ModelConfig:
    return read_checkpoint(path)[0]
