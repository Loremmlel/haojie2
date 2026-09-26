"""冻结预算的配对模仿试验：相同种子、训练批次、公开输入，比较候选空间上下文。"""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from spatial import SpatialPolicyNet

from haojie_training.data import load_dataset, sample_indices, select_batch
from haojie_training.evaluate import evaluate, validate_split, value_baselines
from haojie_training.model import ModelConfig, PolicyValueNet
from haojie_training.runtime import Trainer


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026092603, 2026092604])
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=["baseline", "spatial", "capacity"],
        default=["baseline", "spatial"],
    )
    parser.add_argument(
        "--data", type=Path, default=Path("artifacts/training/expansion-20260925/encoded/train.pt")
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("artifacts/training/expansion-20260925/encoded/validation.pt"),
    )
    args = parser.parse_args()
    if args.steps < 1 or len(set(args.arms)) != len(args.arms):
        parser.error("步数须为正且实验臂不能重复")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    config = ModelConfig.tiny()
    paths = {"train": args.data, "validation": args.validation}
    datasets = {s: load_dataset(p, config) for s, p in paths.items()}
    validate_split(datasets["train"][1], datasets["validation"][1])
    if datasets["train"][1]["encoding"] != "haojie-entities-factorized-v1":
        raise ValueError("空间对照只支持已经验证的v1公开编码")
    configs = {
        arm: ModelConfig(width=72, layers=2, heads=4, ffn=288) if arm == "capacity" else config
        for arm in args.arms
    }
    root = Path(__file__).resolve().parents[4]
    protocol = {
        "config": asdict(config),
        "seeds": args.seeds,
        "steps": args.steps,
        "batch_size": 32,
        "learning_rate": 0.0003,
        "precision": "fp32",
        "value_weight": 0,
        "length_bucket_size": 128,
        "inputs": {s: {"path": str(p), "sha256": digest(p)} for s, p in paths.items()},
        "sources": {
            p.resolve().relative_to(root).as_posix(): digest(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("spatial.py"),
                *Path("training/haojie_training").glob("*.py"),
            ]
        },
        "arms": args.arms,
        "configs": {a: asdict(c) for a, c in configs.items()},
        "checkpoint_selection": "fixed final step",
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    for name, expected in protocol["sources"].items():
        snapshot = args.output / "source" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((root / name).read_bytes())
        assert digest(snapshot) == expected
    train, metadata = datasets["train"]
    baselines = value_baselines(train, metadata["records"])
    results = []
    for seed in args.seeds:
        trainers = {}
        for name in args.arms:
            cls = SpatialPolicyNet if name == "spatial" else PolicyValueNet
            torch.manual_seed(seed)
            trainers[name] = Trainer(
                cls(configs[name]), torch.device("cuda"), "fp32", lr=0.0003, value_weight=0
            )
        # 两臂原有参数从完全相同初始化开始；新增卷积是本轮唯一结构差异。
        if "baseline" in trainers and "spatial" in trainers:
            for key, value in trainers["baseline"].model.state_dict().items():
                assert torch.equal(value, trainers["spatial"].model.state_dict()[key])
        rng = torch.Generator().manual_seed(seed)
        batch_hash = hashlib.sha256()
        started = time.perf_counter()
        for step in range(args.steps):
            ids = sample_indices(train, 32, rng, 128)
            batch_hash.update(ids.numpy().tobytes())
            batch = {k: v.to("cuda") for k, v in select_batch(train, ids).items()}
            for trainer in trainers.values():
                if not torch.isfinite(trainer.step(batch)):
                    raise FloatingPointError("配对训练出现非有限损失")
            if (step + 1) % 1000 == 0:
                print(
                    json.dumps(
                        {"seed": seed, "step": step + 1, "elapsed": time.perf_counter() - started}
                    ),
                    flush=True,
                )
        for name, trainer in trainers.items():
            health = trainer.health()
            assert health["finite_parameters"] and health["finite_gradients"]
            row = {
                "seed": seed,
                "arm": name,
                "updates": trainer.updates,
                "health": health,
                "batch_sha256": batch_hash.hexdigest(),
                "paired_seconds": time.perf_counter() - started,
                "parameters": sum(p.numel() for p in trainer.model.parameters()),
            }
            for split, (data, meta) in datasets.items():
                row[split] = evaluate(
                    trainer.model,
                    data,
                    torch.device("cuda"),
                    "fp32",
                    32,
                    meta["records"],
                    baselines,
                )
            checkpoint = args.output / f"{seed}-{name}.pt"
            torch.save(
                {
                    "format": "haojie-recovery-policy-pilot-v1",
                    "arm": name,
                    "config": asdict(trainer.model.config),
                    "model": trainer.model.state_dict(),
                    "updates": trainer.updates,
                    "protocol_sha256": digest(args.output / "protocol.json"),
                    "training_groups": metadata["groups"],
                },
                checkpoint,
            )
            row["checkpoint_sha256"] = digest(checkpoint)
            results.append(row)
            (args.output / f"{seed}-{name}.json").write_text(
                json.dumps(row, indent=2), encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "arm": name,
                        "nll": row["validation"]["policy_loss"],
                        "deployment": row["validation"]["by_command_stage"]["deploy:point"],
                    }
                ),
                flush=True,
            )
    (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
