"""可运行的张量训练/续训入口；合成数据必须显式选择，不把运行成功解释为获得棋力。"""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from .data import load_dataset, sample_indices, select_batch, synthetic_batch
from .evaluate import evaluate, validate_split, value_baselines
from .model import NETWORK_VERSION, ModelConfig, PolicyValueNet
from .runtime import Trainer, checkpoint_config, resolve_device, synchronize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--synthetic", action="store_true")
    inputs.add_argument("--data", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--entities", type=int, default=64)
    parser.add_argument("--actions", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--length-bucket-size", type=int, default=0)
    parser.add_argument("--overfit-examples", type=int, default=0)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.threads, args.entities, args.actions) < 1:
        parser.error("步数、批量、线程和填充长度必须为正")
    if args.overfit_examples < 0 or args.learning_rate <= 0 or args.length_bucket_size < 0:
        parser.error("拟合样本数不能为负，学习率必须为正")
    if (args.validation or args.overfit_examples) and not args.data:
        parser.error("验证集/小样本拟合需要--data")
    if args.report and args.report.exists():
        parser.error("报告已存在，请使用新的实验报告路径")
    if args.checkpoint.exists() and (
        args.resume is None or args.resume.resolve() != args.checkpoint.resolve()
    ):
        parser.error("拒绝覆盖已有检查点；续训请显式指定同一文件为--resume")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    config = (
        checkpoint_config(args.resume)
        if args.resume
        else ModelConfig.tiny()
        if args.tiny
        else ModelConfig()
    )
    dataset = None
    validation, validation_metadata, records = None, None, None
    metadata = {
        "ruleset": "synthetic-only",
        "encoding": "synthetic-v1",
        "synthetic": True,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "entities": args.entities,
        "actions": args.actions,
    }
    if args.data:
        dataset, provenance = load_dataset(args.data, config)
        records = provenance.get("records")
        with args.data.open("rb") as source:
            dataset_sha = hashlib.file_digest(source, "sha256").hexdigest()
        metadata = {
            **{key: value for key, value in provenance.items() if key != "records"},
            "dataset_sha256": dataset_sha,
            "seed": args.seed,
            "batch_size": args.batch_size,
        }
        if args.overfit_examples:
            if args.overfit_examples > len(dataset["value"]):
                parser.error("拟合样本数超过训练集")
            rng = torch.Generator().manual_seed(args.seed)
            indices = torch.randperm(len(dataset["value"]), generator=rng)[: args.overfit_examples]
            dataset = select_batch(dataset, indices)
            records = [records[i] for i in indices.tolist()] if records else None
            metadata["overfit_indices"] = indices.tolist()
        if args.validation:
            validation, validation_metadata = load_dataset(args.validation, config)
            validate_split(provenance, validation_metadata)
            with args.validation.open("rb") as source:
                metadata["validation_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    metadata["learning_rate"] = args.learning_rate
    if args.length_bucket_size:
        metadata["length_bucket_size"] = args.length_bucket_size
    device = resolve_device(args.device)
    trainer = Trainer(
        PolicyValueNet(config),
        device,
        args.precision,
        lr=args.learning_rate,
        value_weight=args.value_weight,
    )
    if args.resume:
        trainer.restore(args.resume, metadata)

    baselines = value_baselines(dataset, records) if dataset is not None else None

    def metrics():
        result = {}
        if dataset is not None:
            result["train"] = evaluate(
                trainer.model, dataset, device, args.precision, args.batch_size, records, baselines
            )
        if validation is not None:
            result["validation"] = evaluate(
                trainer.model,
                validation,
                device,
                args.precision,
                args.batch_size,
                validation_metadata.get("records"),
                baselines,
            )
        return result

    before = metrics()
    print(json.dumps({"before": before}, ensure_ascii=False), flush=True)
    started = time.perf_counter()
    try:
        for _ in range(args.steps):
            if dataset is None:
                batch = synthetic_batch(
                    config, args.batch_size, args.entities, args.actions, args.seed + trainer.steps
                )
            else:
                rng = torch.Generator().manual_seed(args.seed + trainer.steps)
                indices = sample_indices(dataset, args.batch_size, rng, args.length_bucket_size)
                batch = select_batch(dataset, indices)
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = trainer.step(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("训练loss非有限；未保存损坏的检查点")
            if trainer.steps % 10 == 0 or trainer.steps == 1:
                print(
                    json.dumps(
                        {"step": trainer.steps, "loss": float(loss), "updates": trainer.updates}
                    ),
                    flush=True,
                )
    except KeyboardInterrupt:
        trainer.optimizer.zero_grad(set_to_none=True)
        raise SystemExit("已中断；保留已有检查点，未将不完整优化步保存为可续训状态") from None
    synchronize(device)
    training_seconds = time.perf_counter() - started
    health = trainer.health()
    if not health["finite_parameters"] or not health["finite_gradients"] or not trainer.updates:
        raise FloatingPointError(f"训练数值检查失败：{health}")
    trainer.save(args.checkpoint, metadata)
    source_hash = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes())
    report = {
        "format": "haojie-training-run-v1",
        "network": NETWORK_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "torch": str(torch.__version__),
        "training_source_sha256": source_hash.hexdigest(),
        "config": asdict(config),
        "threads": torch.get_num_threads(),
        "device": str(device),
        "precision": args.precision,
        "value_weight": args.value_weight,
        "parameters": sum(p.numel() for p in trainer.model.parameters()),
        "steps": trainer.steps,
        "updates": trainer.updates,
        "loss": float(loss),
        "training_seconds": training_seconds,
        "checkpoint": str(args.checkpoint),
        "metadata": metadata,
        "health": health,
        "before": before,
        "after": metrics(),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as destination:
            json.dump(report, destination, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "metadata"},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
