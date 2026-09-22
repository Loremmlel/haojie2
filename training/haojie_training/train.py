"""可运行的张量训练/续训入口；合成数据必须显式选择，不把运行成功解释为获得棋力。"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from .data import load_dataset, synthetic_batch
from .model import ModelConfig, PolicyValueNet
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
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.threads, args.entities, args.actions) < 1:
        parser.error("步数、批量、线程和填充长度必须为正")
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
        with args.data.open("rb") as source:
            dataset_sha = hashlib.file_digest(source, "sha256").hexdigest()
        metadata = {
            **provenance,
            "dataset_sha256": dataset_sha,
            "seed": args.seed,
            "batch_size": args.batch_size,
        }
    device = resolve_device(args.device)
    trainer = Trainer(PolicyValueNet(config), device, args.precision)
    if args.resume:
        trainer.restore(args.resume, metadata)
    started = time.perf_counter()
    try:
        for _ in range(args.steps):
            if dataset is None:
                batch = synthetic_batch(
                    config, args.batch_size, args.entities, args.actions, args.seed + trainer.steps
                )
            else:
                rng = torch.Generator().manual_seed(args.seed + trainer.steps)
                indices = torch.randint(len(dataset["value"]), (args.batch_size,), generator=rng)
                batch = {key: tensor[indices] for key, tensor in dataset.items()}
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
    health = trainer.health()
    if not health["finite_parameters"] or not health["finite_gradients"] or not trainer.updates:
        raise FloatingPointError(f"训练数值检查失败：{health}")
    trainer.save(args.checkpoint, metadata)
    print(
        json.dumps(
            {
                "device": str(device),
                "precision": args.precision,
                "parameters": sum(p.numel() for p in trainer.model.parameters()),
                "steps": trainer.steps,
                "updates": trainer.updates,
                "loss": float(loss),
                "elapsed_seconds": time.perf_counter() - started,
                "checkpoint": str(args.checkpoint),
                "metadata": metadata,
                "health": health,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
