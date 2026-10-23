"""BF16 训练步与检查点；参数、梯度、矩和舍入余量使用同一数值体系。"""

import math
from collections import defaultdict
from copy import copy
from dataclasses import asdict
from pathlib import Path

import torch

from .model import NETWORK_VERSION, ModelConfig, PolicyValueNet, policy_value_loss, stable_tanh


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用；请在新建训练中选择CPU或安装CUDA版PyTorch")
    if device.type == "cuda":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise RuntimeError("本 CUDA 设备没有原生 BF16 支持")
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


def _finite(tensors):
    """分组最大绝对值捕获 NaN/Inf；避免为每个参数及 Adam 状态发射数个小内核。"""
    checks = defaultdict(list)
    for tensor in tensors:
        if tensor.numel():
            checks[tensor.device].append(tensor)
    with torch.no_grad():
        return all(
            bool(torch.isfinite(torch.stack(torch._foreach_norm(values, math.inf))).all())
            for values in checks.values()
        )


class BF16AdamW(torch.optim.Optimizer):
    """参数、矩和舍入余量全为 BF16，步数使用整数。

    EMA 用增量形式补偿舍入，避免 beta2=0.999 在 BF16 中变成不衰减。
    参数同样保留舍入余量，让小于参数 ULP 的更新在后续步骤生效。
    """

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if not math.isfinite(lr) or lr < 0 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("学习率须有限非负，epsilon 须有限且为正")
        if not all(math.isfinite(beta) and 0 <= beta < 1 for beta in betas):
            raise ValueError("Adam beta 须在 [0,1) 内")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("权重衰减须有限非负")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        if any(p.dtype != torch.bfloat16 for group in self.param_groups for p in group["params"]):
            raise ValueError("BF16 AdamW 只接受 BF16 参数")

    @staticmethod
    def _add(values, residuals, changes=None):
        if changes is not None:
            torch._foreach_add_(residuals, changes)
        before = torch._foreach_sub(values, 0)
        torch._foreach_add_(values, residuals)
        torch._foreach_sub_(before, values)
        torch._foreach_add_(residuals, before)

    def load_state_dict(self, state_dict):
        # 旧检查点仅在数据边界转换，不修改调用方或磁盘上的原始状态。
        converted = {
            **state_dict,
            "param_groups": [
                {key: group[key] for key in ("params", "lr", "betas", "eps", "weight_decay")}
                for group in state_dict["param_groups"]
            ],
            "state": {
                key: {**state, "step": int(state["step"])}
                for key, state in state_dict["state"].items()
            },
        }
        super().load_state_dict(converted)
        for parameter, state in self.state.items():
            for key in ("compensation", "avg_compensation", "sq_compensation"):
                state.setdefault(key, torch.zeros_like(parameter))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            by_step = defaultdict(list)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("BF16 AdamW 不支持稀疏梯度")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    for key in (
                        "exp_avg",
                        "exp_avg_sq",
                        "compensation",
                        "avg_compensation",
                        "sq_compensation",
                    ):
                        state[key] = torch.zeros_like(parameter)
                state["step"] += 1
                by_step[state["step"]].append(parameter)
            beta1, beta2 = group["betas"]
            for step, parameters in by_step.items():
                gradients = [p.grad for p in parameters]
                states = [self.state[p] for p in parameters]
                average = [s["exp_avg"] for s in states]
                square = [s["exp_avg_sq"] for s in states]
                delta = torch._foreach_sub(gradients, average)
                torch._foreach_mul_(delta, 1 - beta1)
                self._add(average, [s["avg_compensation"] for s in states], delta)
                delta = torch._foreach_mul(gradients, gradients)
                torch._foreach_sub_(delta, square)
                torch._foreach_mul_(delta, 1 - beta2)
                self._add(square, [s["sq_compensation"] for s in states], delta)
                denominator = torch._foreach_sqrt(square)
                torch._foreach_div_(denominator, math.sqrt(1 - beta2**step))
                torch._foreach_add_(denominator, group["eps"])
                residual = [s["compensation"] for s in states]
                torch._foreach_add_(
                    residual, parameters, alpha=-group["lr"] * group["weight_decay"]
                )
                torch._foreach_addcdiv_(
                    residual, average, denominator, value=-group["lr"] / (1 - beta1**step)
                )
                self._add(parameters, residual)
        return loss


class Trainer:
    """
    执行一个策略/价值优化步；输入须先通过CPU边界校验。
    基准和普通训练复用同一实现，含清梯度、前向、反向、裁剪、AdamW及缩放更新。
    同步与数值健康检查由调用方在计时边界执行，避免逐张量GPU同步影响测量。
    """

    def __init__(
        self,
        model: PolicyValueNet,
        device: torch.device,
        lr=3e-4,
        value_weight=1.0,
        objective="sampled-action-imitation",
    ):
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
        self.device, self.precision = device, "bf16"
        self.model = model.to(device=device, dtype=torch.bfloat16).train()
        self.optimizer = BF16AdamW(model.parameters(), lr=lr)
        self.steps = 0
        self.updates = 0

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
        known = batch["value_mask"].to(torch.bfloat16)
        if denominator is None:
            denominator = known.sum().clamp_min(1)
        value_loss = ((value - batch["value"]).square() * known).sum() / denominator
        if mc:
            policy_loss = (
                (stable_tanh(chosen) - batch["value"]).square() * known
            ).sum() / denominator
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
        同形一步不额外计算整组归一化分母。
        """
        if not batches:
            raise ValueError("没有可训练子批")
        denominator = None
        if len(batches) > 1:
            if not self.objective.startswith("decomposed-mc-q-"):
                raise ValueError("变长子批当前仅用于 MC-Q 终局回归")
            denominator = (
                torch.stack([b["value_mask"].sum() for b in batches])
                .sum()
                .clamp_min(1)
                .to(torch.bfloat16)
            )
        self.optimizer.zero_grad(set_to_none=True)
        totals = None
        for batch in batches:
            losses = self._loss(batch, denominator)
            loss = losses[0]
            if not torch.isfinite(loss):
                raise FloatingPointError("非有限损失，未执行更新")
            loss.backward()
            detached = tuple(value.detach() for value in losses)
            totals = detached if totals is None else tuple(a + b for a, b in zip(totals, detached))
            # 仅累计日志标量，不把上一子批计算图保留到下一次前向。
            del losses, loss
        self._last_losses = totals[1:]
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, foreach=True)
        self.optimizer.step()
        self.steps += 1
        self.updates += 1
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
                "numerics": "bf16-compensated-adamw-v1",
                "value_weight": self.value_weight,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps": self.steps,
                "updates": self.updates,
                "rng_cpu": torch.get_rng_state(),
                "rng_device": None
                if self.device.type == "cpu"
                else getattr(torch, self.device.type).get_rng_state(self.device),
                "device_type": self.device.type,
                "execution": {
                    "deterministic": torch.are_deterministic_algorithms_enabled(),
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
            or payload.get("value_weight", 1.0) != self.value_weight
        ):
            raise ValueError("检查点的模型、数据来源或损失权重与当前训练不一致")
        execution = {
            "deterministic": torch.are_deterministic_algorithms_enabled(),
        }
        if (
            payload.get("execution", execution).get("deterministic", False)
            != execution["deterministic"]
        ):
            raise ValueError("续训确定性执行设置不匹配")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
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
