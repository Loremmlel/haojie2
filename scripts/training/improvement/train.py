"""固定最终步数的结构/数据消融；共享初始化与批次，保存可推理研究检查点。"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from haojie_training.data import load_dataset
from haojie_training.evaluate import evaluate
from haojie_training.model import ModelConfig
from haojie_training.runtime import Trainer

from .data import TrainingData, digest
from .model import ARMS, ImprovementNet

FORMAT = "haojie-improvement-policy-v1"


def model_config(arm):
    return (
        ModelConfig(width=72, layers=2, heads=4, ffn=288)
        if arm == "capacity"
        else ModelConfig.tiny()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--data",
        nargs="+",
        type=Path,
        default=[Path("artifacts/training/expansion-20260925/encoded/train.pt")],
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("artifacts/training/expansion-20260925/encoded/validation.pt"),
    )
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026101901, 2026101902])
    parser.add_argument("--mixture", type=float, nargs="+")
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--balanced", action="store_true")
    sampling.add_argument("--family-balanced", action="store_true")
    parser.add_argument("--choice-loss", action="store_true")
    parser.add_argument("--initialize", type=Path)
    parser.add_argument("--features", nargs="*")
    args = parser.parse_args()
    if args.steps < 1 or len(args.arms) != len(set(args.arms)):
        parser.error("步数必须为正且消融臂不能重复")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    validation, validation_meta = load_dataset(args.validation, ModelConfig.tiny())
    training = TrainingData(
        args.data,
        validation_meta,
        ModelConfig.tiny(),
        args.mixture,
        args.balanced,
        args.family_balanced,
    )
    if any(
        meta["encoding"] != "haojie-entities-factorized-v1" or meta["synthetic"]
        for _, meta in training.datasets
    ):
        raise ValueError("本轮要求真实v1编码")
    root = Path.cwd()
    sources = sorted(
        [
            *Path(__file__).parent.glob("*.py"),
            *Path("training/haojie_training").glob("*.py"),
            Path("scripts/training/search/recovery/spatial.py"),
        ]
    )
    source_hashes = {str(p.resolve().relative_to(root)): digest(p) for p in sources}
    protocol = {
        "format": FORMAT,
        "arguments": {
            k: [str(x) for x in v]
            if isinstance(v, list) and v and isinstance(v[0], Path)
            else str(v)
            if isinstance(v, Path)
            else v
            for k, v in vars(args).items()
        },
        "inputs": {
            str(p.resolve()): digest(p)
            for p in [*args.data, args.validation, *([args.initialize] if args.initialize else [])]
        },
        "sources": source_hashes,
        "batch_size": 32,
        "learning_rate": 0.0003,
        "value_weight": 0,
        "precision": "fp32",
        "checkpoint_selection": "fixed final step",
        "training_groups": sorted(training.groups),
        "mixture": training.weights.tolist(),
        "examples": training.count,
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    for name, sha in source_hashes.items():
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(name).read_bytes())
        assert digest(target) == sha
    parent = (
        torch.load(args.initialize, map_location="cpu", weights_only=True)
        if args.initialize
        else None
    )
    seen = training.groups | (set(parent["training_groups"]) if parent else set())
    if seen & set(validation_meta["groups"]):
        raise ValueError("初始化检查点已经看过验证族")
    results = []
    for seed in args.seeds:
        trainers = {}
        for arm in args.arms:
            torch.manual_seed(seed)
            model = ImprovementNet(model_config(arm), arm, args.features)
            if parent:
                if (
                    parent["format"] != FORMAT
                    or parent["config"] != asdict(model.config)
                    or parent["features"] != list(model.features)
                ):
                    raise ValueError("父模型结构不一致")
                model.load_state_dict(parent["model"])
            trainers[arm] = Trainer(model, torch.device("cuda"), "fp32", value_weight=0)
        if "spatial" in trainers:
            base = trainers["spatial"].model.state_dict()
            for arm, trainer in trainers.items():
                if arm != "capacity":
                    for name, value in base.items():
                        assert torch.equal(value, trainer.model.state_dict()[name]), name
        rng = torch.Generator().manual_seed(seed)
        import hashlib

        batch_hash = hashlib.sha256()
        start = time.perf_counter()
        for step in range(args.steps):
            batch, ids = training.sample(32, rng)
            batch_hash.update(ids.numpy().tobytes())
            batch = {key: value.to("cuda") for key, value in batch.items()}
            for trainer in trainers.values():
                if args.choice_loss:
                    trainer.optimizer.zero_grad(set_to_none=True)
                    logits, _ = trainer.model(batch)
                    losses = -(batch["policy"] * logits.log_softmax(-1)).sum(-1)
                    mask = (batch["candidate_mask"].sum(-1) > 1).float()
                    loss = (losses * mask).sum() / mask.sum().clamp_min(1)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0, foreach=True)
                    trainer.optimizer.step()
                    trainer.steps += 1
                else:
                    loss = trainer.step(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("消融出现非有限损失")
            if (step + 1) % 500 == 0:
                print(
                    json.dumps(
                        {"seed": seed, "step": step + 1, "seconds": time.perf_counter() - start}
                    ),
                    flush=True,
                )
        seconds = time.perf_counter() - start
        for name, sha in source_hashes.items():
            if digest(name) != sha:
                raise ValueError(f"执行期间源码改变：{name}")
        for arm, trainer in trainers.items():
            health = trainer.health()
            assert (
                health["finite_parameters"]
                and health["finite_gradients"]
                and trainer.updates == args.steps
            )
            metrics = evaluate(
                trainer.model,
                validation,
                torch.device("cuda"),
                "fp32",
                32,
                validation_meta["records"],
            )
            metadata = training.datasets[0][1]
            checkpoint = args.output / f"{seed}-{arm}.pt"
            torch.save(
                {
                    "format": FORMAT,
                    "arm": arm,
                    "features": list(trainer.model.features),
                    "config": asdict(trainer.model.config),
                    "model": trainer.model.state_dict(),
                    "optimizer": trainer.optimizer.state_dict(),
                    "updates": trainer.updates,
                    "parent": str(args.initialize) if parent else None,
                    "parent_sha256": digest(args.initialize) if parent else None,
                    "training_groups": sorted(seen),
                    "protocol_sha256": digest(args.output / "protocol.json"),
                    "metadata": {
                        k: metadata[k]
                        for k in ("synthetic", "ruleset", "encoding", "schema", "source_sha256")
                    },
                },
                checkpoint,
            )
            row = {
                "seed": seed,
                "arm": arm,
                "features": list(trainer.model.features),
                "updates": trainer.updates,
                "parameters": sum(p.numel() for p in trainer.model.parameters()),
                "health": health,
                "batch_sha256": batch_hash.hexdigest(),
                "training_seconds_all_arms": seconds,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest(checkpoint),
                "validation": metrics,
            }
            results.append(row)
            (args.output / f"{seed}-{arm}.json").write_text(
                json.dumps(row, indent=2), encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "arm": arm,
                        "nll": metrics["policy_loss"],
                        "path": metrics["teacher_path_accuracy"],
                        "deploy": metrics["by_command_stage"].get("deploy:point"),
                    }
                ),
                flush=True,
            )
        (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
