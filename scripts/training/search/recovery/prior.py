"""部署的无棋盘基线：仅从训练教师落点拟合席位/卡种频次，不使用验证标签选先验。"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch

from haojie_training.data import load_dataset, select_batch
from haojie_training.evaluate import validate_split
from haojie_training.model import ModelConfig


def positions(path):
    data, metadata = load_dataset(path, ModelConfig.tiny())
    indices = [
        i
        for i, r in enumerate(metadata["records"])
        if r["command"] == "deploy" and r["stage"] == "point"
    ]
    rows = []
    for start in range(0, len(indices), 32):
        ids = indices[start : start + 32]
        batch = select_batch(data, torch.tensor(ids))
        for b, index in enumerate(ids):
            chosen = int(batch["policy"][b].argmax())
            source = int(batch["sources"][b, chosen])
            assert source >= 0
            xy = (batch["candidates"][b, :, 36:38] * torch.tensor([9, 13])).round().long()
            row = metadata["records"][index]
            rows.append(
                {
                    "actor": row["actor"],
                    "kind": int(batch["kinds"][b, source]),
                    "group": row["group"],
                    "selected": chosen,
                    "points": [tuple(p) for p in xy[batch["candidate_mask"][b]].tolist()],
                }
            )
    return rows, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    training, tm = positions(args.train)
    validation, vm = positions(args.validation)
    validate_split(tm, vm)
    tables = {name: defaultdict(Counter) for name in ("seat", "seat-kind")}
    for row in training:
        point = row["points"][row["selected"]]
        tables["seat"][row["actor"]][point] += 1
        tables["seat-kind"][(row["actor"], row["kind"])][point] += 1
    report = {
        "training_examples": len(training),
        "validation_examples": len(validation),
        "smoothing": 1,
        "inputs": {"train": str(args.train), "validation": str(args.validation)},
        "results": {},
    }
    for name in tables:
        hits, losses = [], []
        for row in validation:
            key = row["actor"] if name == "seat" else (row["actor"], row["kind"])
            counts = tables[name].get(key, tables["seat"][row["actor"]])
            weights = [counts[p] + 1 for p in row["points"]]
            selected = max(range(len(weights)), key=weights.__getitem__)
            hits.append(selected == row["selected"])
            losses.append(-math.log(weights[row["selected"]] / sum(weights)))
        report["results"][name] = {
            "accuracy": sum(hits) / len(hits),
            "nll": sum(losses) / len(losses),
        }
    report["results"]["uniform"] = {
        "expected_accuracy": sum(1 / len(r["points"]) for r in validation) / len(validation),
        "nll": sum(math.log(len(r["points"])) for r in validation) / len(validation),
    }
    with args.output.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
