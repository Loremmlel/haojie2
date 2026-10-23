"""补偿 BF16 AdamW；提交之前聚合检查梯度和预计算更新的有限性。"""

import math

import torch


def finite(values):
    values = list(values)
    return not values or bool(torch.stack([torch.isfinite(v).all() for v in values]).all())


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if not (math.isfinite(lr) and lr > 0 and math.isfinite(eps) and eps > 0):
            raise ValueError("学习率和 epsilon 必须为有限正数")
        if not all(0 <= b < 1 for b in betas) or not 0 <= weight_decay < math.inf:
            raise ValueError("非法 AdamW 参数")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        if any(p.dtype != torch.bfloat16 for g in self.param_groups for p in g["params"]):
            raise ValueError("参数必须为 BF16")

    @staticmethod
    def add(value, residual, change):
        residual = residual + change
        result = value + residual
        return result, residual + (value - result)

    @torch.no_grad()
    def step(self):
        updates = []
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                old = self.state.get(p)
                if old is None:
                    old = {key: torch.zeros_like(p) for key in ("m", "v", "mr", "vr", "pr")}
                    old["step"] = 0
                step = old["step"] + 1
                m, mr = self.add(old["m"], old["mr"], (p.grad - old["m"]) * (1 - b1))
                v, vr = self.add(old["v"], old["vr"], (p.grad.square() - old["v"]) * (1 - b2))
                denominator = (v / (1 - b2**step)).sqrt() + group["eps"]
                change = -group["lr"] * (
                    m / (1 - b1**step) / denominator + group["weight_decay"] * p
                )
                value, pr = self.add(p, old["pr"], change)
                updates.append((p, value, dict(m=m, v=v, mr=mr, vr=vr, pr=pr, step=step)))
        tensors = [
            v
            for p, value, state in updates
            for v in (p.grad, value, *[x for x in state.values() if isinstance(x, torch.Tensor)])
        ]
        if not finite(tensors):
            raise FloatingPointError("BF16 AdamW 提交前发现非有限值；参数和优化器未提交")
        for p, value, state in updates:
            p.copy_(value)
            self.state[p] = state

    def load_state_dict(self, state_dict):
        for state in state_dict["state"].values():
            if set(state) != {"m", "v", "mr", "vr", "pr", "step"}:
                raise ValueError("不支持的优化器状态")
            if any(
                isinstance(v, torch.Tensor) and v.dtype != torch.bfloat16 for v in state.values()
            ):
                raise ValueError("优化器浮点状态必须为 BF16")
            if type(state["step"]) is not int or state["step"] < 0:
                raise ValueError("优化器步数必须为非负整数")
            if not finite(v for v in state.values() if isinstance(v, torch.Tensor)):
                raise ValueError("优化器状态包含非有限值")
        super().load_state_dict(state_dict)
