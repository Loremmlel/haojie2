"""逐配置隔离进程，比较同模型/同输入的完整训练步；不包含游戏采样和特征编码。"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from .data import synthetic_batch
from .model import ModelConfig, PolicyValueNet
from .runtime import Trainer, autocast, resolve_device, synchronize

ROOT = Path(__file__).resolve().parents[2]


def tensor_hash(tensors: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_case(case: dict) -> dict:
    torch.set_num_threads(case["threads"])
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(case["seed"])
    device = resolve_device(case["device"])
    config = ModelConfig()
    model = PolicyValueNet(config).eval()
    cpu_batch = synthetic_batch(
        config, case["batch"], case["entities"], case["actions"], case["seed"] + 1
    )
    model_sha = tensor_hash(model.state_dict())
    input_sha = tensor_hash(cpu_batch)
    with torch.no_grad():
        reference_logits, reference_value = model(cpu_batch)
    initial_weight = model.entity_projection.weight.detach().clone()
    trainer = Trainer(model, device, case["precision"])
    batch = {key: value.to(device) for key, value in cpu_batch.items()}
    operation_dtypes = []
    hook = trainer.model.blocks[0].qkv.register_forward_hook(
        lambda _module, _inputs, output: operation_dtypes.append(str(output.dtype))
    )
    with torch.no_grad(), autocast(device, case["precision"]):
        candidate_logits, candidate_value = trainer.model(batch)
    hook.remove()
    logits_error = (candidate_logits.cpu() - reference_logits)[cpu_batch["candidate_mask"]]
    reference_rms = reference_logits[cpu_batch["candidate_mask"]].square().mean().sqrt()
    agreement = {
        "policy_rms_relative_error": float(
            logits_error.square().mean().sqrt() / reference_rms.clamp_min(1e-8)
        ),
        "value_max_absolute_error": float((candidate_value.cpu() - reference_value).abs().max()),
        "top1_agreement": float(
            (candidate_logits.cpu().argmax(1) == reference_logits.argmax(1)).float().mean()
        ),
    }
    if device.type != "cpu":
        getattr(torch, device.type).reset_peak_memory_stats(device)
    warmup_start = time.perf_counter()
    warmup_losses = [trainer.step(batch) for _ in range(case["warmup"])]
    synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_start
    if not torch.stack(warmup_losses).isfinite().all():
        raise FloatingPointError("预热loss出现NaN/Inf")
    warmup_skipped = trainer.steps - trainer.updates
    samples_ms = []
    losses = []
    previous_updates = trainer.updates
    for _ in range(case["rounds"]):
        synchronize(device)
        started = time.perf_counter()
        for _ in range(case["steps"]):
            losses.append(trainer.step(batch))
        synchronize(device)
        samples_ms.append((time.perf_counter() - started) * 1000 / case["steps"])
    effective_updates = trainer.updates - previous_updates
    attempted = case["steps"] * case["rounds"]
    health = trainer.health()
    loss_values = torch.stack(losses).cpu()
    weight_delta = float(
        (trainer.model.entity_projection.weight.detach().cpu() - initial_weight).abs().max()
    )
    valid = (
        bool(loss_values.isfinite().all())
        and health["finite_gradients"]
        and health["finite_parameters"]
        and effective_updates == attempted
        and weight_delta > 0
    )
    peak_memory = (
        None
        if device.type == "cpu"
        else {
            "allocated_mib": getattr(torch, device.type).max_memory_allocated(device) / 1024**2,
            "reserved_mib": getattr(torch, device.type).max_memory_reserved(device) / 1024**2,
        }
    )
    transfer_ms = []
    if device.type != "cpu":
        for _ in range(3):
            synchronize(device)
            started = time.perf_counter()
            copied = {key: value.to(device) for key, value in cpu_batch.items()}
            synchronize(device)
            transfer_ms.append((time.perf_counter() - started) * 1000)
            del copied
    median_ms = statistics.median(samples_ms)
    return {
        **case,
        "status": "ok" if valid else "unstable",
        "config": asdict(config),
        "parameters": sum(p.numel() for p in model.parameters()),
        "model_sha256": model_sha,
        "input_sha256": input_sha,
        "torch": str(torch.__version__),
        "device_name": platform.processor()
        if device.type == "cpu"
        else getattr(torch, device.type).get_device_name(device),
        "matmul_output_dtype": operation_dtypes[0],
        "parameter_dtype": str(next(model.parameters()).dtype),
        "samples_ms_per_step": samples_ms,
        "median_ms_per_step": median_ms,
        "samples_per_second": case["batch"] * 1000 / median_ms if valid else None,
        "warmup_seconds": warmup_seconds,
        "warmup_skipped": warmup_skipped,
        "timed_updates": effective_updates,
        "timed_skipped": attempted - effective_updates,
        "first_loss": float(warmup_losses[0]),
        "final_loss": float(loss_values[-1]),
        "weight_max_change": weight_delta,
        "health": health,
        "initial_fp32_agreement": agreement,
        "peak_memory": peak_memory,
        "input_transfer_ms": statistics.median(transfer_ms) if transfer_ms else 0,
    }


def hardware() -> dict:
    result = {
        "platform": platform.platform(),
        "python": sys.version,
        "logical_cpus": os.cpu_count(),
    }
    if sys.platform == "win32":
        query = (
            "$cpu=Get-CimInstance Win32_Processor; $gpu=Get-CimInstance Win32_VideoController; "
            "$ram=Get-CimInstance Win32_ComputerSystem; "
            "@{cpu=$cpu.Name; gpu=@($gpu | Select-Object Name,DriverVersion); "
            "memory_bytes=$ram.TotalPhysicalMemory} | ConvertTo-Json -Depth 4"
        )
        response = subprocess.run(
            ["pwsh.exe", "-NoProfile", "-Command", query],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
        )
        if response.returncode == 0:
            result.update(json.loads(response.stdout))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument(
        "--devices", nargs="+", choices=["cpu", "xpu", "cuda"], default=["cpu", "xpu"]
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=["fp32", "bf16", "fp16"],
        default=["fp32", "bf16", "fp16"],
    )
    parser.add_argument("--batches", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--cpu-threads", nargs="+", type=int, default=[4])
    parser.add_argument("--xpu-threads", type=int, default=4)
    parser.add_argument("--entities", type=int, default=64)
    parser.add_argument("--actions", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/training/benchmark.json")
    args = parser.parse_args()
    if args.case:
        case = json.loads(args.case)
        try:
            result = run_case(case)
        except Exception as error:
            result = {**case, "status": "failed", "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return
    if (
        min(
            *args.batches,
            *args.cpu_threads,
            args.xpu_threads,
            args.entities,
            args.actions,
            args.warmup,
            args.steps,
            args.rounds,
            args.timeout,
        )
        < 1
    ):
        parser.error("测量次数和输入大小必须为正")
    if args.output.exists():
        parser.error("拒绝覆盖已有基准报告，请使用新的--output")
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    report = {
        "format": "haojie-torch-benchmark-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware(),
        "torch": str(torch.__version__),
        "source_sha256": digest.hexdigest(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config": {
            key: value for key, value in vars(args).items() if key not in {"case", "output"}
        },
        "method": (
            "eager; AdamW foreach; FP32 parameters/states; AMP for bf16/fp16; "
            "resident synthetic batches; synchronized complete optimization steps; "
            "1 inter-op thread"
        ),
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for batch in args.batches:
        for precision in args.precisions:
            for device in args.devices:
                for threads in args.cpu_threads if device == "cpu" else [args.xpu_threads]:
                    case = {
                        "device": device,
                        "precision": precision,
                        "batch": batch,
                        "entities": args.entities,
                        "actions": args.actions,
                        "threads": threads,
                        "warmup": args.warmup,
                        "steps": args.steps,
                        "rounds": args.rounds,
                        "seed": 20260922,
                    }
                    label = f"{device}/{precision}/batch{batch}/threads{threads}"
                    print(f"开始 {label}", flush=True)
                    command = [
                        sys.executable,
                        "-m",
                        "haojie_training.benchmark",
                        "--case",
                        json.dumps(case),
                    ]
                    try:
                        response = subprocess.run(
                            command,
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            timeout=args.timeout,
                            env={**os.environ, "PYTHONUTF8": "1"},
                        )
                        result = (
                            json.loads(response.stdout.strip().splitlines()[-1])
                            if response.returncode == 0
                            else {**case, "status": "failed", "exit_code": response.returncode}
                        )
                        if response.stderr:
                            result["stderr"] = response.stderr[-6000:]
                    except subprocess.TimeoutExpired:
                        result = {**case, "status": "timeout", "timeout_seconds": args.timeout}
                    except (ValueError, IndexError) as error:
                        result = {**case, "status": "failed", "error": str(error)}
                    report["results"].append(result)
                    args.output.write_text(
                        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                    )
                    print(
                        f"完成 {label}: {result['status']}, "
                        f"{result.get('median_ms_per_step', 0):.2f} ms/step",
                        flush=True,
                    )
    print(f"报告：{args.output}", flush=True)


if __name__ == "__main__":
    main()
