"""冻结检查点的完整教师复核；按族额外报告，绝不在此更新权重或挑选检查点。"""

import argparse
import json
from pathlib import Path

import torch
from haojie_training.data import load_dataset, select_batch
from haojie_training.evaluate import evaluate
from haojie_training.model import ModelConfig

from .data import digest
from .serve import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    data, metadata = load_dataset(args.data, ModelConfig.tiny())
    results = []
    for path in args.checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if set(payload["training_groups"]) & set(metadata["groups"]):
            raise ValueError("检查点训练族与复核交叉")
        _, model, original, _ = load_checkpoint(path)
        if any(
            original[k] != metadata[k] for k in ("encoding", "ruleset", "schema", "source_sha256")
        ):
            raise ValueError("复核规则/编码来源不同")
        model = model.to("cuda").eval()
        metrics = evaluate(model, data, torch.device("cuda"), "fp32", 32, metadata["records"])
        per_group = {}
        for group in metadata["groups"]:
            ids = [i for i, r in enumerate(metadata["records"]) if r["group"] == group]
            per_group[group] = evaluate(
                model,
                select_batch(data, torch.tensor(ids)),
                torch.device("cuda"),
                "fp32",
                32,
                [metadata["records"][i] for i in ids],
            )
        results.append(
            {
                "checkpoint": str(path),
                "checkpoint_sha256": digest(path),
                "data_sha256": digest(args.data),
                "metrics": metrics,
                "per_group": per_group,
            }
        )
        print(
            json.dumps(
                {
                    "checkpoint": str(path),
                    "nll": metrics["policy_loss"],
                    "path": metrics["teacher_path_accuracy"],
                    "deploy": metrics["by_command_stage"].get("deploy:point"),
                }
            ),
            flush=True,
        )
        del model
    (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
