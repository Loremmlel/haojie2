"""连续同席位 V-trace；根去重一次，微批即时反传并统一归一化。"""

from contextlib import nullcontext

import torch

from .model import DTYPE, node_batch, root_batch, work_batches
from .optimizer import AdamW, finite
from .staging import Staging


@torch.no_grad()
def vtrace(rewards, discounts, values, bootstrap, logp, behavior):
    ratio = (logp - behavior).exp().clamp_max(1)
    following = torch.cat((values[1:], bootstrap[None]))
    delta = ratio * (rewards + discounts * following - values)
    correction = torch.zeros_like(bootstrap)
    targets = torch.empty_like(values)
    for t in range(len(values) - 1, -1, -1):
        correction = delta[t] + discounts[t] * ratio[t] * correction
        targets[t] = values[t] + correction
    following_targets = torch.cat((targets[1:], bootstrap[None]))
    return targets, ratio * (rewards + discounts * following_targets - values)


class Learner:
    def __init__(self, model, lr=3e-4):
        self.model = model
        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.steps = 0
        self.consumed = 0
        self.staging = Staging(next(model.parameters()).device)
        self.last_root_count = 0
        self.last_decode_batches = []

    def update(self, blocks, roots, microbatch=64):
        """编码器保存的激活在 CPU；解码器先取目标，再逐微批反传累积根梯度。

        每个根只执行一次带梯度编码。两次轻量解码均使用当前参数和真实历史；
        第一次无梯度得到整段 V-trace 目标，第二次的所有损失使用相同总样本分母。
        完成全部根反传与有限值检查后，优化器才提交一次。
        """
        device = next(self.model.parameters()).device
        steps = [step for block in blocks for step in block["steps"]]
        if not steps:
            raise ValueError("空逻辑训练批")
        nodes = [s["node"] for s in steps]
        nodes += [b["bootstrap"] for b in blocks if b["bootstrap"] is not None]
        keys = list(dict.fromkeys(n["root"] for n in nodes))
        encoded, graphs, gradients = {}, [], {}
        history_graphs = []
        history_values = torch.zeros(len(nodes), self.model.config.history, dtype=DTYPE)
        history_gradients = torch.zeros_like(history_values)
        self.optimizer.zero_grad(set_to_none=True)
        self.last_root_count = len(keys)
        for group in work_batches(
            keys, microbatch, lambda k: ((len(roots[k]["entities"]) + 1) ** 2,)
        ):
            context = torch.autograd.graph.save_on_cpu() if device.type == "cuda" else nullcontext()
            with context:
                output = self.model.encode(
                    *root_batch([roots[k] for k in group], device, self.staging)
                )
                saved = output.to("cpu")
            graphs.append((group, saved))
            for i, key in enumerate(group):
                encoded[key] = saved[i, : len(roots[key]["entities"]) + 1].detach()
                gradients[key] = torch.zeros_like(encoded[key])
            del output
            indices = [i for i, n in enumerate(nodes) if n["root"] in group]
            context = torch.autograd.graph.save_on_cpu() if device.type == "cuda" else nullcontext()
            with context:
                history = self.model.node_histories(
                    [nodes[i] for i in indices], device, self.staging
                ).to("cpu")
            history_values[indices] = history.detach()
            if history.requires_grad:
                history_graphs.append((indices, history))

        def batches(indices):
            return work_batches(
                indices,
                microbatch,
                lambda i: (
                    len(encoded[nodes[i]["root"]]) * self.model.config.width,
                    len(nodes[i]["candidates"]) * self.model.config.width,
                    nodes[i]["history_end"] * self.model.config.history,
                ),
            )

        def forward(indices, grad):
            chunk = [nodes[i] for i in indices]
            packed = (
                torch.nn.utils.rnn.pad_sequence(
                    [encoded[n["root"]] for n in chunk], batch_first=True
                )
                .to(device)
                .requires_grad_(grad)
            )
            prefix, pm, _, _, candidates, sources, targets, mask = node_batch(
                chunk, device, self.staging, include_events=False
            )
            history = history_values[indices].to(device).requires_grad_(grad)
            logits, values = self.model.decode(
                packed, prefix, pm, history, candidates, sources, targets, mask
            )
            distribution = logits.log_softmax(-1)
            chosen = torch.tensor(
                [steps[i]["selected"] if i < len(steps) else 0 for i in indices], device=device
            )
            logp = distribution.gather(1, chosen[:, None]).squeeze(1)
            entropy = -(distribution.masked_fill(~mask, 0) * distribution.exp()).sum(-1)
            return packed, history, logp, values, entropy

        values = torch.empty(len(nodes), device=device, dtype=DTYPE)
        logps = torch.empty(len(nodes), device=device, dtype=DTYPE)
        with torch.no_grad():
            for indices in batches(range(len(nodes))):
                _, _, lp, value, _ = forward(indices, False)
                values[indices], logps[indices] = value, lp
            targets, advantages = [], []
            offset, tail = 0, len(steps)
            for block in blocks:
                size = len(block["steps"])
                sl = slice(offset, offset + size)
                tensors = [
                    torch.tensor([s[k] for s in block["steps"]], device=device, dtype=DTYPE)
                    for k in ("reward", "discount", "logp")
                ]
                bootstrap = values.new_zeros(()) if block["bootstrap"] is None else values[tail]
                target, advantage = vtrace(
                    tensors[0], tensors[1], values[sl], bootstrap, logps[sl], tensors[2]
                )
                targets.append(target)
                advantages.append(advantage)
                offset += size
                tail += int(block["bootstrap"] is not None)
            targets, advantages = torch.cat(targets), torch.cat(advantages)

        total = values.new_zeros(())
        self.last_decode_batches = []
        for indices in batches(range(len(steps))):
            packed, history, lp, value, entropy = forward(indices, True)
            loss = (
                -lp * advantages[indices]
                + 0.5 * (value - targets[indices]).square()
                - 0.01 * entropy
            ).sum() / len(steps)
            loss.backward()
            total += loss.detach()
            root_grad = packed.grad.cpu()
            history_gradients[indices] += history.grad.cpu()
            for i, index in enumerate(indices):
                key = nodes[index]["root"]
                gradients[key].add_(root_grad[i, : len(gradients[key])])
            self.last_decode_batches.append(len(indices))
            del packed, history, lp, value, entropy, loss, root_grad
        for indices, graph in history_graphs:
            graph.backward(history_gradients[indices])
        for group, graph in graphs:
            grad = torch.zeros_like(graph)
            for i, key in enumerate(group):
                grad[i, : len(gradients[key])] = gradients[key]
            graph.backward(grad)
        if not finite([total, *(p.grad for p in self.model.parameters() if p.grad is not None)]):
            raise FloatingPointError("V-trace 损失或梯度非有限；未提交优化步")
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.steps += 1
        self.consumed += len(steps)
        return float(total)
