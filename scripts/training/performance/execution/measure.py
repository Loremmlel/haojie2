"""同进程加载冻结权重，交错短测实际请求和固定索引连续更新；CPU仅显式选择。"""

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from haojie_training.batching import training_batches
from haojie_training.data import INPUT_KEYS, collate_examples, load_dataset
from haojie_training.model import ModelConfig, PolicyValueNet, policy_value_loss
from haojie_training.native.execution import PolicyInference
from haojie_training.runtime import Trainer, autocast, synchronize


def load_reference(folder):
    spec = importlib.util.spec_from_file_location(
        "reference_training",
        folder / "__init__.py",
        submodule_search_locations=[str(folder)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from reference_training.data import (
        collate_examples as old_collate,
        load_dataset as old_load,
        select_batch as old_select,
        sample_indices,
    )
    from reference_training.model import PolicyValueNet as OldModel
    from reference_training.runtime import Trainer as OldTrainer

    return old_collate, old_load, old_select, sample_indices, OldModel, OldTrainer


def update_stages(trainer, cpu_batch):
    """仅诊断时用事件分段；计算与Trainer.step一致，不进入正式吞吐计时。"""
    marks = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    trainer.optimizer.zero_grad(set_to_none=True)
    marks[0].record()
    batch = {
        k: v.to(trainer.device, non_blocking=v.is_pinned())
        for k, v in cpu_batch.items()
    }
    marks[1].record()
    with autocast(trainer.device, trainer.precision):
        loss = policy_value_loss(trainer.model(batch), batch, trainer.value_weight)
    marks[2].record()
    trainer.scaler.scale(loss).backward()
    marks[3].record()
    trainer.scaler.unscale_(trainer.optimizer)
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0, foreach=True)
    trainer.scaler.step(trainer.optimizer)
    trainer.scaler.update()
    marks[4].record()
    synchronize(trainer.device)
    return {
        name: marks[i].elapsed_time(marks[i + 1])
        for i, name in enumerate(
            (
                "transfer_ms",
                "forward_loss_ms",
                "backward_ms",
                "clip_optimizer_scaler_ms",
            )
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "fixture", "reference", "data", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument(
        "--cases", nargs="+", default=["reference", "optimized", "fp16", "bf16"]
    )
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if args.deterministic:
        import os

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    old_collate, old_load, old_select, old_sample, OldModel, OldTrainer = (
        load_reference(args.reference)
    )
    payload = torch.load(args.checkpoint, weights_only=True)
    config = ModelConfig(**payload["config"])
    models = [
        OldModel(config).to(device).eval(),
        PolicyValueNet(config).to(device).eval(),
    ]
    for model in models:
        model.load_state_dict(payload["model"])
    examples = torch.load(args.fixture, weights_only=True)["examples"]
    groups = [examples[i : i + 4] for i in range(0, len(examples) - 3, 4)]
    expected = []
    with torch.inference_mode():
        for group in groups:
            batch = {k: v.to(device) for k, v in old_collate(group, config).items()}
            expected.append(tuple(v.cpu() for v in models[0](batch)))
    torch.save(expected, args.output / "reference.pt")
    # 在看混合精度结果之前固定门槛；排名差异另报，不静默当作逐位等价。
    tolerances = {
        "fp32": {"atol": 2e-6, "rtol": 2e-5},
        "fp16": {"atol": 0.002, "rtol": 0.02},
        "bf16": {"atol": 0.01, "rtol": 0.05},
    }
    runners = {
        case: PolicyInference(
            models[1],
            device,
            case if case in ("bf16", "fp16") else "fp32",
            pinned=case != "pageable",
        )
        for case in args.cases
        if case != "reference"
    }
    calls = 0

    def infer(case, group):
        nonlocal calls
        calls += 1
        if case != "reference":
            return runners[case]([{k: e[k] for k in INPUT_KEYS} for e in group])
        batch = old_collate(group, config)
        with torch.inference_mode():
            logits, values = models[0]({k: v.to(device) for k, v in batch.items()})
            logits, values = logits.cpu(), values.cpu()
        if not torch.isfinite(logits).all() or not torch.isfinite(values).all():
            raise ValueError("非有限输出")
        return logits

    errors = {}
    for case in args.cases:
        precision = case if case in ("bf16", "fp16") else "fp32"
        differences, changes, value_errors, rank_changes = [], 0, [], 0
        for group, (logits, value) in zip(groups, expected):
            actual = infer(case, group)
            torch.testing.assert_close(actual, logits, **tolerances[precision])
            differences.append(float((actual - logits).abs().max()))
            changes += int((actual.argmax(1) != logits.argmax(1)).sum())
            rank_changes += int(
                (
                    actual.argsort(1, descending=True)
                    != logits.argsort(1, descending=True)
                )
                .any(1)
                .sum()
            )
            with torch.inference_mode(), autocast(device, precision):
                _, actual_value = models[1](
                    {
                        k: v.to(device)
                        for k, v in collate_examples(group, config).items()
                    }
                )
            torch.testing.assert_close(
                actual_value.cpu(), value, **tolerances[precision]
            )
            value_errors.append(float((actual_value.cpu() - value).abs().max()))
        errors[case] = {
            "logits_max_abs": max(differences),
            "value_max_abs": max(value_errors),
            "argmax_changed": changes,
            "ranking_changed": rank_changes,
            "requests": len(groups) * 4,
        }
    report = {
        "torch": str(torch.__version__),
        "device": str(device),
        "parameters": sum(p.numel() for p in models[1].parameters()),
        "tolerances": tolerances,
        "errors": errors,
        "inference": [],
        "training": [],
        "deterministic": args.deterministic,
    }
    try:
        report["background"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,power.draw,clocks.sm,memory.used",
                "--format=csv",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    for round_id in range(args.rounds):
        for case in args.cases if round_id % 2 == 0 else args.cases[::-1]:
            latencies = []
            start = time.perf_counter()
            for _ in range(3):
                for group in groups:
                    begin = time.perf_counter()
                    infer(case, group)
                    latencies.append(time.perf_counter() - begin)
            seconds = time.perf_counter() - start
            row = {
                "round": round_id,
                "case": case,
                "seconds": seconds,
                "requests_per_second": len(groups) * 12 / seconds,
                "batch_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] * 1000,
            }
            report["inference"].append(row)
            print(json.dumps(row), flush=True)
    if args.training:
        # 同一默认模型、实局批次的必要梯度检查；混合精度使用预定整体L2门槛。
        gradient_batch = {
            k: v.to(device)
            for k, v in collate_examples(groups[len(groups) // 2], config).items()
        }
        models[0].zero_grad(set_to_none=True)
        policy_value_loss(models[0](gradient_batch), gradient_batch).backward()
        reference_gradients = [p.grad.detach().clone() for p in models[0].parameters()]
        report["gradient_errors"] = {}
        for precision, tolerance in [("fp32", 2e-5), ("fp16", 0.02), ("bf16", 0.05)]:
            if precision != "fp32" and precision not in args.cases:
                continue
            models[1].zero_grad(set_to_none=True)
            with autocast(device, precision):
                policy_value_loss(models[1](gradient_batch), gradient_batch).backward()
            gradients = [p.grad for p in models[1].parameters()]
            numerator = sum(
                (a - b).square().sum() for a, b in zip(gradients, reference_gradients)
            )
            denominator = sum(b.square().sum() for b in reference_gradients)
            error = float((numerator / denominator.clamp_min(1e-20)).sqrt())
            assert error <= tolerance, (precision, error, tolerance)
            report["gradient_errors"][precision] = {
                "relative_l2": error,
                "limit": tolerance,
            }
        for model in models:
            model.zero_grad(set_to_none=True)
        del reference_gradients, gradients, gradient_batch
        begin = time.perf_counter()
        dataset, _ = load_dataset(args.data, config)
        report["load_seconds"] = time.perf_counter() - begin
        begin = time.perf_counter()
        old_dataset, _ = old_load(args.data, config)
        report["reference_load_seconds"] = time.perf_counter() - begin
        for round_id in range(args.rounds):
            for case in args.cases if round_id % 2 == 0 else args.cases[::-1]:
                if case == "pageable":
                    continue
                precision = case if case in ("bf16", "fp16") else "fp32"
                model = (OldModel if case == "reference" else PolicyValueNet)(config)
                model.load_state_dict(payload["model"])
                trainer = (OldTrainer if case == "reference" else Trainer)(
                    model, device, precision
                )
                warm = {
                    k: v.to(device)
                    for k, v in collate_examples(groups[0], config).items()
                }
                for _ in range(4):
                    trainer.step(warm)
                model.load_state_dict(payload["model"])
                trainer.optimizer.state.clear()
                trainer.steps = trainer.updates = 0
                synchronize(device)
                start = time.perf_counter()
                if case == "reference":

                    def batches():
                        for step in range(args.steps):
                            rng = torch.Generator().manual_seed(20261004 + step)
                            yield old_select(
                                old_dataset, old_sample(old_dataset, 4, rng, 64)
                            )

                    batches = batches()
                else:
                    batches = training_batches(
                        dataset,
                        config,
                        seed=20261004,
                        start=0,
                        steps=args.steps,
                        size=4,
                        bucket_size=64,
                        prefetch=True,
                        pin_memory=True,
                    )
                losses = []
                for batch in batches:
                    loss = trainer.step(
                        {
                            k: v.to(device, non_blocking=case != "reference")
                            for k, v in batch.items()
                        }
                    )
                    if not torch.isfinite(loss):
                        raise ValueError("非有限损失")
                    losses.append(float(loss))
                synchronize(device)
                seconds = time.perf_counter() - start
                row = {
                    "round": round_id,
                    "case": case,
                    "seconds": seconds,
                    "updates": trainer.updates,
                    "samples_per_second": trainer.updates * 4 / seconds,
                    "updates_per_second": trainer.updates / seconds,
                    "losses": losses,
                    "health": trainer.health(),
                }
                report["training"].append(row)
                print(
                    json.dumps({k: v for k, v in row.items() if k != "losses"}),
                    flush=True,
                )
                if args.profile and round_id == 0:
                    row["stage_diagnostic"] = update_stages(trainer, batch)
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as prof:
                        trainer.step({k: v.to(device) for k, v in batch.items()})
                        synchronize(device)
                    (args.output / (case + "-update-profile.txt")).write_text(
                        prof.key_averages().table(
                            sort_by="self_cuda_time_total", row_limit=25
                        ),
                        encoding="utf-8",
                    )
                del trainer, model
    report["padding"] = {
        "entities": sum(len(e["entities"]) for g in groups for e in g)
        / sum(4 * max(len(e["entities"]) for e in g) for g in groups),
        "candidates": sum(len(e["candidates"]) for g in groups for e in g)
        / sum(4 * max(len(e["candidates"]) for e in g) for g in groups),
    }
    report["inference_calls"] = calls
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
