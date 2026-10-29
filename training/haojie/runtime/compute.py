"""GPU 计算所有者；共用推理与学习数学，协调端仅持有 CPU 恢复状态。"""

import time

import numpy as np
import torch
from torch.nn import functional as F

from ..model import Network, collate_states, collate_options, device_for_bf16


class Compute:
    def initialize_network(self, device):
        torch.set_num_threads(1)
        torch.manual_seed(self.config.seed)
        self.device = torch.device(device)
        self.model = Network().to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.0001,
            foreach=False,
        )
        for parameter in self.model.parameters():
            # 更新次数使用整数；参数和两组动量均保留 BF16。
            self.optimizer.state[parameter] = {
                "step": torch.tensor(0, dtype=torch.int64),
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.zeros_like(parameter),
            }

    def activate(self):
        device = device_for_bf16(self.config.device)
        if device == self.device:
            return
        self.model.to(device)
        for state in self.optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                state[key] = state[key].to(device)
        self.device = device

    def restore_optimizer(self, state):
        self.optimizer.load_state_dict(state)
        for group in self.optimizer.param_groups:
            group["foreach"] = False
        for value in self.optimizer.state.values():
            value["step"] = value["step"].to(device="cpu", dtype=torch.int64)

    def inference_model(self, version):
        if version not in self.inference_models:
            if len(self.inference_models) >= 3:
                self.inference_models.pop(next(iter(self.inference_models)))
            network = Network().to(self.device)
            network.load_state_dict(
                torch.load(
                    self.models / f"model-{version:08d}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            self.inference_models[version] = network.eval()
        network = self.inference_models.pop(version)
        self.inference_models[version] = network
        return network

    def infer(self, jobs):
        self.activate()
        output = [None] * len(jobs)
        grouped = {}
        for i, (version, request, owner) in enumerate(jobs):
            grouped.setdefault((version, request.op), []).append(i)
        start = time.perf_counter()
        with torch.inference_mode():
            for (version, op), indices in grouped.items():
                indices.sort(key=lambda i: len(jobs[i][1].arrays[0]))
                model = self.inference_model(version)
                for offset in range(0, len(indices), self.config.inference_batch):
                    selection = indices[offset : offset + self.config.inference_batch]
                    if op == "encode":
                        tensors = collate_states(
                            [jobs[i][1].arrays for i in selection],
                            self.device,
                            pinned=self.device.type == "cuda",
                        )
                        context, values, aux = model.encode(*tensors)
                        packed = (
                            torch.cat((values[:, None], aux), 1)
                            .view(torch.uint16)
                            .cpu()
                            .numpy()
                        )
                        for row, i in enumerate(selection):
                            _, request, owner = jobs[i]
                            self.contexts[(owner, request.context, version)] = context[
                                row, : len(request.arrays[0])
                            ].clone()
                            output[i] = packed[row].copy()
                        self.stats["inference_roots"] += len(selection)
                    else:
                        contexts = [
                            self.contexts[(jobs[i][2], jobs[i][1].context, version)]
                            for i in selection
                        ]
                        n = max(len(x) for x in contexts)
                        context = torch.zeros(
                            (len(selection), n, self.model.config.width),
                            dtype=torch.bfloat16,
                            device=self.device,
                        )
                        for row, x in enumerate(contexts):
                            context[row, : len(x)] = x
                        tensors = collate_options(
                            [jobs[i][1].arrays[0] for i in selection],
                            self.device,
                            pinned=self.device.type == "cuda",
                        )
                        packed = (
                            model.decode(context, *tensors)
                            .view(torch.uint16)
                            .cpu()
                            .numpy()
                        )
                        for row, i in enumerate(selection):
                            output[i] = packed[row, : len(jobs[i][1].arrays[0])].copy()
                        self.stats["inference_decodes"] += len(selection)
                    self.stats["inference_batches"] += 1
        self.stats["inference_seconds"] += time.perf_counter() - start
        return output

    def update(self, batch):
        self.activate()
        started = time.perf_counter()
        tensors = collate_states(
            [b[0] for b in batch], self.device, pinned=self.device.type == "cuda"
        )
        returns = torch.tensor(
            [b[2] for b in batch], device=self.device, dtype=torch.bfloat16
        )
        value_weight = torch.tensor(
            [b[3] for b in batch], device=self.device, dtype=torch.bfloat16
        )
        auxiliary = torch.tensor(
            np.stack([b[4] for b in batch]), device=self.device, dtype=torch.bfloat16
        )
        aux_mask = torch.tensor(
            [b[5] for b in batch], device=self.device, dtype=torch.bool
        )
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        context, values, predicted_aux = self.model.encode(*tensors)
        groups = {}
        for sample, row in enumerate(batch):
            for options, target, weight in row[1]:
                groups.setdefault(max(1, len(options)).bit_length(), []).append(
                    (sample, options, target, weight)
                )
        policy_losses = torch.zeros(
            len(batch), dtype=torch.bfloat16, device=self.device
        )
        for decisions in groups.values():
            options = collate_options(
                [d[1] for d in decisions],
                self.device,
                pinned=self.device.type == "cuda",
            )
            indices = torch.tensor(
                [d[0] for d in decisions], dtype=torch.int64, device=self.device
            )
            target = torch.zeros_like(options[1], dtype=torch.bfloat16)
            for i, d in enumerate(decisions):
                target[i, : len(d[2])] = torch.from_numpy(d[2]).to(
                    self.device, dtype=torch.bfloat16
                )
            weight = torch.tensor(
                [d[3] for d in decisions], dtype=torch.bfloat16, device=self.device
            )
            logits = self.model.decode(context.index_select(0, indices), *options)
            logp = F.log_softmax(logits, -1).masked_fill(~options[1], 0)
            policy_losses = policy_losses.index_add(
                0, indices, -(target * logp).sum(-1) * weight
            )
        policy_loss = policy_losses.mean()
        value_loss = ((values - returns).square() * value_weight).sum() / (
            value_weight > 0
        ).sum().clamp_min(1).to(torch.bfloat16)
        aux_loss = (
            F.smooth_l1_loss(predicted_aux, auxiliary, reduction="none").mean(-1)
            * aux_mask
        ).sum() / aux_mask.sum().clamp_min(1).to(torch.bfloat16)
        loss = policy_loss + value_loss + 0.1 * aux_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("非有限损失")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 1.0, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.stats["updates"] += 1
        self.stats["trained_samples"] += len(batch)
        self.stats["value_samples"] += sum(b[6] for b in batch)
        self.stats["soft_value_samples"] += sum(b[7] for b in batch)
        self.stats["loss"] = {
            "policy": float(policy_loss.detach()),
            "value": float(value_loss.detach()),
            "auxiliary": float(aux_loss.detach()),
            "gradient_norm": float(norm),
        }
        self.stats["learning_seconds"] += time.perf_counter() - started
