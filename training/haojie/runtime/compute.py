"""GPU 计算所有者；共用推理与学习数学，协调端仅持有 CPU 恢复状态。"""

import time

import numpy as np
import torch
from torch.nn import functional as F

from ..model import Network, collate_states, collate_options, device_for_bf16
from ..profiling import Profile


class Compute:
    # 384 宽模型自对弈可同时用到二十余版；按需装载和发布共用这些槽位。
    inference_model_limit = 32

    def initialize_network(self, device, weights=None):
        self.profile = Profile()
        torch.set_num_threads(1)
        torch.manual_seed(self.config.seed)
        self.device = torch.device(device)
        self.model = (
            Network().to(self.device)
            if weights is None
            else Network.from_weights(weights, self.device)
        )
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
            if len(self.inference_models) >= self.inference_model_limit:
                self.inference_models.pop(next(iter(self.inference_models)))
            network = Network.from_weights(self.weights.get(version), self.device)
            self.inference_models[version] = network.eval()
            self.weights.prune(self.live_versions())
        network = self.inference_models.pop(version)
        self.inference_models[version] = network
        return network

    def live_versions(self):
        return (
            {self.published}
            | {v for _, v in self.bindings.values()}
            | set(self.pending_versions.values())
        )

    def prune_models(self):
        live = self.live_versions()
        self.weights.prune(live)
        for version in list(self.inference_models):
            if version not in live and version not in self.weights.history:
                del self.inference_models[version]

    def infer(self, jobs):
        with self.profile.stage("inference.activate"):
            self.activate()
        with self.profile.stage("inference.group"):
            output = [None] * len(jobs)
            # 各组在同一 CUDA 流上依次提交，整轮共用一次设备到主机回传。
            parts = []
            layouts = []
            grouped = {}
            for i, (version, request, owner) in enumerate(jobs):
                grouped.setdefault((version, request.op), []).append(i)
            self.profile.observe("inference.input_batch", len(jobs))
            self.profile.observe("inference.groups", len(grouped))
        start = time.perf_counter()
        with torch.inference_mode():
            for (version, op), indices in grouped.items():
                with self.profile.stage("inference.group_sort"):
                    indices.sort(key=lambda i: len(jobs[i][1].arrays[0]))
                with self.profile.stage("inference.model_lookup"):
                    model = self.inference_model(version)
                for offset in range(0, len(indices), self.config.inference_batch):
                    selection = indices[offset : offset + self.config.inference_batch]
                    if self.profile.enabled:
                        arrays = [jobs[i][1].arrays for i in selection]
                        self.profile.observe(op + ".batch", len(selection))
                        sizes = [len(a[0]) for a in arrays]
                        self.profile.observe(op + ".elements", sum(sizes))
                        self.profile.observe(
                            op + ".padded_elements", len(sizes) * max(sizes)
                        )
                        if op == "encode":
                            self.profile.observe(
                                "encode.attention_pairs", sum(n * n for n in sizes)
                            )
                            self.profile.observe(
                                "encode.padded_attention_pairs",
                                len(sizes) * max(sizes) ** 2,
                            )
                            records = [len(a[1]) for a in arrays]
                            self.profile.observe("encode.records", sum(records))
                            self.profile.observe(
                                "encode.padded_records", len(records) * max(records)
                            )
                    sampled = self.profile.select(op, 32, len(selection))
                    cuda = sampled and self.device.type == "cuda"
                    if op == "encode":
                        tensors = collate_states(
                            [jobs[i][1].arrays for i in selection],
                            self.device,
                            pinned=self.device.type == "cuda",
                            profile=self.profile,
                            scope="inference.encode",
                            cuda=cuda,
                        )
                        with self.profile.stage(
                            "inference.encode.network_submit", cuda=cuda
                        ):
                            context, values, aux = model.encode(*tensors)
                        with self.profile.stage(
                            "inference.encode.pack_submit", cuda=cuda
                        ):
                            packed_tensor = torch.cat((values[:, None], aux), 1).view(
                                torch.uint16
                            )
                        with self.profile.stage(
                            "inference.encode.scatter_cache", cuda=cuda
                        ):
                            for row, i in enumerate(selection):
                                _, request, owner = jobs[i]
                                self.contexts[(owner, request.context, version)] = (
                                    context[row, : len(request.arrays[0])].clone()
                                )
                        self.stats["inference_roots"] += len(selection)
                    else:
                        with self.profile.stage(
                            "inference.decode.context_pack", cuda=cuda
                        ):
                            contexts = [
                                self.contexts[(jobs[i][2], jobs[i][1].context, version)]
                                for i in selection
                            ]
                            n = max(len(x) for x in contexts)
                            context = torch.zeros(
                                (len(selection), n, model.config.width),
                                dtype=torch.bfloat16,
                                device=self.device,
                            )
                            for row, x in enumerate(contexts):
                                context[row, : len(x)] = x
                        tensors = collate_options(
                            [jobs[i][1].arrays[0] for i in selection],
                            self.device,
                            pinned=self.device.type == "cuda",
                            profile=self.profile,
                            scope="inference.decode",
                            cuda=cuda,
                        )
                        with self.profile.stage(
                            "inference.decode.network_submit", cuda=cuda
                        ):
                            packed_tensor = model.decode(context, *tensors).view(
                                torch.uint16
                            )
                        self.stats["inference_decodes"] += len(selection)
                    parts.append(packed_tensor.reshape(-1))
                    layouts.append((selection, packed_tensor.shape[1], op))
                    self.stats["inference_batches"] += 1
            if parts:
                with self.profile.stage("inference.wave.d2h_wait"):
                    flat = (
                        (parts[0] if len(parts) == 1 else torch.cat(parts))
                        .cpu()
                        .numpy()
                    )
                self.stats["inference_waves"] += 1
                self.profile.collect()
                offset = 0
                for selection, stride, op in layouts:
                    for i in selection:
                        size = stride if op == "encode" else len(jobs[i][1].arrays[0])
                        output[i] = flat[offset : offset + size].copy()
                        offset += stride
        self.stats["inference_seconds"] += time.perf_counter() - start
        return output

    def update(self, batch):
        with self.profile.stage("training.activate"):
            self.activate()
        started = time.perf_counter()
        sampled = self.profile.select("training", 8, len(batch))
        cuda = sampled and self.device.type == "cuda"
        with self.profile.stage("training.prepare"):
            tensors = collate_states(
                [b[0] for b in batch],
                self.device,
                pinned=self.device.type == "cuda",
                profile=self.profile,
                scope="training.prepare.states",
                cuda=cuda,
            )
            returns = torch.tensor(
                [b[2] for b in batch], device=self.device, dtype=torch.bfloat16
            )
            value_weight = torch.tensor(
                [b[3] for b in batch], device=self.device, dtype=torch.bfloat16
            )
            auxiliary = torch.tensor(
                np.stack([b[4] for b in batch]),
                device=self.device,
                dtype=torch.bfloat16,
            )
            aux_mask = torch.tensor(
                [b[5] for b in batch], device=self.device, dtype=torch.bool
            )
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
        with self.profile.stage("training.forward_loss", cuda=cuda):
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
                    profile=self.profile,
                    scope="training.forward_loss.options",
                    cuda=cuda,
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
        with self.profile.stage("training.finite_check"):
            if not torch.isfinite(loss):
                raise FloatingPointError("非有限损失")
        self.profile.collect()
        with self.profile.stage("training.backward", cuda=cuda):
            loss.backward()
        with self.profile.stage("training.clip_grad", cuda=cuda):
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0, error_if_nonfinite=True
            )
        with self.profile.stage("training.optimizer", cuda=cuda):
            self.optimizer.step()
        with self.profile.stage("training.report_sync"):
            self.stats["updates"] += 1
            self.stats["trained_samples"] += len(batch)
            self.stats["value_samples"] += sum(b[6] for b in batch)
            self.stats["soft_value_samples"] += sum(b[7] for b in batch)
            self.stats["trained_teacher_positions"] += sum(b[8] for b in batch)
            self.stats["loss"] = {
                "policy": float(policy_loss.detach()),
                "value": float(value_loss.detach()),
                "auxiliary": float(aux_loss.detach()),
                "gradient_norm": float(norm),
            }
        self.profile.collect()
        self.stats["learning_seconds"] += time.perf_counter() - started
