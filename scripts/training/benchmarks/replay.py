"""隔离进程交错重放同一批公开请求和 MC-Q 标签；只对固定工作报告加速比例。"""

import argparse
import hashlib
import importlib
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

if __package__:
    from .cycle import hardware_identity
else:
    from cycle import hardware_identity


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare_cases(before, after):
    """输入、初始化和实际消费都匹配才计算比例；架构允许改变输出。"""
    if not before or len(before) != len(after):
        raise ValueError("固定重放需要等量的前后测量")
    for cases in (before, after):
        if any(
            case[key] != cases[0][key] for case in cases for key in ("network", "source_sha256")
        ):
            raise ValueError("同侧重放期间源码或网络版本改变")
    reference = before[0]
    for case in before + after:
        for key in (
            "torch",
            "python",
            "hardware_identity",
            "model_config",
            "parameters",
            "initial_model_sha256",
            "execution",
            "options",
            "requests_sha256",
            "samples_sha256",
            "training_batch_sha256",
            "work",
        ):
            if case[key] != reference[key]:
                raise ValueError(f"固定重放条件不同：{key}")
        if (
            case["timed_updates"] != case["options"]["updates"]
            or case["warmup_updates"] != 1
            or case["weight_max_change"] <= 0
            or not all(
                case["health"][key]
                for key in ("finite_gradients", "finite_parameters", "finite_optimizer")
            )
        ):
            raise ValueError("固定重放未完成健康、有效的更新")
    metrics = {}
    for name in reference["seconds"]:
        a = [case["seconds"][name] for case in before]
        b = [case["seconds"][name] for case in after]
        first, last = statistics.median(a), statistics.median(b)
        metrics[name] = {
            "before_seconds": a,
            "after_seconds": b,
            "median_before_seconds": first,
            "median_after_seconds": last,
            "time_reduction_percent": 100 * (1 - last / first),
            "throughput_gain_percent": 100 * (first / last - 1),
        }
    return metrics


def run_case(args):
    if not (args.source / "training/haojie_training/model.py").is_file():
        raise ValueError("源码目录缺少训练模型")
    sys.path.insert(0, str(args.source.resolve() / "training"))
    importlib.import_module("haojie_training")
    import torch

    from haojie_training.benchmark import tensor_hash
    from haojie_training.console.config.resources import hardware
    from haojie_training.data import collate_examples, collate_inputs
    from haojie_training.model import NETWORK_VERSION, ModelConfig, PolicyValueNet
    from haojie_training.native.execution import PolicyInference
    from haojie_training.runtime import Trainer, resolve_device, synchronize

    network_module = importlib.import_module("haojie_training.model")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(args.seed)
    config = ModelConfig.tiny() if args.tiny else ModelConfig()
    model = PolicyValueNet(config).eval()
    initial_hash = tensor_hash(model.state_dict())
    initial_weight = model.entity_projection.weight.detach().clone()
    device = resolve_device(args.device)
    model.to(device)
    requests = torch.load(args.requests, map_location="cpu", weights_only=True)
    samples = torch.load(args.samples, map_location="cpu", weights_only=True)
    if requests["format"] != "haojie-inference-requests-v1":
        raise ValueError("需要独立采集的公开推理请求")
    if samples["format"] != "haojie-cycle-samples-v1":
        raise ValueError("需要独立采集的真实 MC-Q 样本")
    requests, samples = requests["requests"], samples["samples"]
    if not requests or len(samples) < 8:
        raise ValueError("需要至少一条公开请求及八条真实样本")
    batch = collate_examples(samples[:8], config)
    if not batch["value_mask"].all():
        raise ValueError("固定 MC-Q 批次必须包含八条有效标签")
    result = {
        "network": NETWORK_VERSION,
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "hardware_identity": hardware_identity(hardware()),
        "model_config": asdict(config),
        "parameters": sum(p.numel() for p in model.parameters()),
        "initial_model_sha256": initial_hash,
        "requests_sha256": file_hash(args.requests),
        "samples_sha256": file_hash(args.samples),
        "training_batch_sha256": tensor_hash(batch),
        "options": {
            key: getattr(args, key)
            for key in ("threads", "precision", "device", "seed", "tiny", "updates", "repeats")
        },
        "work": {},
        "seconds": {},
        "request_profiles": [],
    }
    digest = hashlib.sha256()
    for path in sorted((args.source / "training/haojie_training").rglob("*.py")):
        digest.update(path.relative_to(args.source).as_posix().encode())
        digest.update(path.read_bytes())
    result["source_sha256"] = digest.hexdigest()
    for size in (1, 8):
        # 末组循环补齐，保留每个原请求，并确实测量 B8；额外消费明确计数。
        groups = [
            [requests[j % len(requests)] for j in range(i, i + size)]
            for i in range(0, len(requests), size)
        ]
        inputs_hash = hashlib.sha256()
        for group in groups:
            inputs = collate_inputs(group, config)
            inputs_hash.update(tensor_hash(inputs).encode())
            if size == 1:
                valid = inputs["entity_mask"]
                entities = int(valid.sum())
                active = getattr(network_module, "active_tokens", None)
                active_count = int(active(inputs).sum()) if active else entities + 1
                result["request_profiles"].append(
                    {
                        "entities": entities,
                        "deaths": int(((inputs["entities"][..., 0] == 5 / 32) & valid).sum()),
                        "mc_logs": int(((inputs["kinds"] == 250) & valid).sum()),
                        "middle_active_tokens": active_count,
                        "middle_frozen_tokens": entities + 1 - active_count,
                    }
                )
        runner = PolicyInference(model, device, args.precision)
        for group in groups:
            runner(group)
        synchronize(device)
        warmup_calls, warmup_requests = runner.stats.calls, runner.stats.requests
        started = time.perf_counter()
        for _ in range(args.repeats):
            for group in groups:
                runner(group)
        synchronize(device)
        name = f"inference_b{size}"
        result["seconds"][name] = time.perf_counter() - started
        calls = runner.stats.calls - warmup_calls
        consumed = runner.stats.requests - warmup_requests
        if calls != len(groups) * args.repeats or consumed != calls * size:
            raise ValueError("固定重放未消费全部推理请求")
        result["work"][name] = {
            "batch_size": size,
            "calls": calls,
            "requests": consumed,
            "captured_requests": len(requests),
            "inputs_sha256": inputs_hash.hexdigest(),
        }
        del runner
    batch = {key: value.to(device) for key, value in batch.items()}
    trainer = Trainer(model, device, args.precision, objective="decomposed-mc-q-v2")
    first_loss = trainer.step(batch)
    synchronize(device)
    result["warmup_updates"] = trainer.updates
    started = time.perf_counter()
    losses = [trainer.step(batch) for _ in range(args.updates)]
    synchronize(device)
    result["seconds"]["training_b8"] = time.perf_counter() - started
    result["timed_updates"] = trainer.updates - result["warmup_updates"]
    result["health"] = trainer.health()
    result["weight_max_change"] = float(
        (model.entity_projection.weight.detach().cpu() - initial_weight).abs().max()
    )
    if not torch.stack([first_loss, *losses]).isfinite().all():
        raise FloatingPointError("MC-Q 重放出现非有限损失")
    result["first_loss"], result["final_loss"] = float(first_loss), float(losses[-1])
    result["work"]["training_b8"] = {
        "batch_size": 8,
        "updates": result["timed_updates"],
        "samples_consumed": 8 * result["timed_updates"],
    }
    result["execution"] = {
        "device": str(device),
        "cuda_version": torch.version.cuda,
        "objective": trainer.objective,
        "optimizer": type(trainer.optimizer).__name__,
        "optimizer_options": trainer.optimizer.defaults,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "interop_threads": torch.get_num_interop_threads(),
    }
    compare_cases([result], [result])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-source", type=Path)
    parser.add_argument("--after-source", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda", "xpu"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--tiny", action="store_true", help="仅用于脚本快速验证")
    args = parser.parse_args()
    if min(args.threads, args.trials, args.updates, args.repeats) < 1:
        parser.error("线程数和测量次数必须为正整数")
    if args.output.exists():
        parser.error("拒绝覆盖已有报告，请使用新的 --output")
    if args.source:
        result = run_case(args)
    else:
        if args.before_source is None:
            parser.error("需要 --before-source")
        cases = {"before": [], "after": []}
        folder = args.output.with_suffix("")
        folder.mkdir(parents=True, exist_ok=True)
        for trial in range(args.trials):
            for side in ("before", "after") if trial % 2 == 0 else ("after", "before"):
                output = folder / f"{side}-{trial + 1}.json"
                command = [sys.executable, str(Path(__file__).resolve())]
                options = {
                    "source": getattr(args, f"{side}_source").resolve(),
                    "requests": args.requests.resolve(),
                    "samples": args.samples.resolve(),
                    "output": output.resolve(),
                    **{
                        key: getattr(args, key)
                        for key in ("threads", "device", "precision", "seed", "updates", "repeats")
                    },
                }
                for key, value in options.items():
                    command.extend((f"--{key}", str(value)))
                if args.tiny:
                    command.append("--tiny")
                print(f"固定重放 {side} {trial + 1}/{args.trials}", flush=True)
                subprocess.run(command, check=True)
                cases[side].append(json.loads(output.read_text(encoding="utf-8")))
        result = {
            "format": "haojie-fixed-replay-v1",
            "same_work": True,
            "method": (
                "alternating isolated processes; identical seeded parameters and captured tensors; "
                "PolicyInference B1/B8, cyclic tail padding; resident MC-Q batch 8; "
                "one full inference pass and one training update warmup; synchronized timings"
            ),
            "metrics": compare_cases(cases["before"], cases["after"]),
            "cases": cases,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not args.source:
        print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
