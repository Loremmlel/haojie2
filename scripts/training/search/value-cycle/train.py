"""自对弈价值回接的首轮固定训练；来源只读，保留旧验证族并追加一个已知终局开发验证族。"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from haojie_training.data import FORMAT, SHARD_FORMAT, load_dataset, select_batch
from haojie_training.model import ModelConfig


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


source, output = map(Path, sys.argv[1:3])
output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(2)
protocol = {
    "format": "haojie-neural-leaf-cycle-v1",
    "input_manifest_sha256": digest(source / "manifest.json"),
    "train_seeds": [2026092804, 2026092805],
    "validation_seeds": [2026092803, 2026092806],
    "training_seed": 2026092862,
    "steps": 256,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "value_weight": 1.0,
    "neural_leaf_scale": 0.25,
    "note": "root-only value bootstrap; 2806 is a declared development holdout, not a blind test; old manifests are unchanged",
    "script_sha256": digest(Path(__file__)),
}
(output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
inputs = [
    load_dataset(source / f"{s}.pt", ModelConfig()) for s in ("train", "validation")
]
split_reports = {}
for split in ("train", "validation"):
    seeds = protocol[f"{split}_seeds"]
    records, shards = [], []
    known = 0
    groups = sorted(
        {g for _, m in inputs for g in m["groups"] if int(g.rsplit(":", 1)[1]) in seeds}
    )
    metadata = {
        k: v
        for k, v in inputs[0][1].items()
        if k not in ("records", "groups", "game_ids", "split")
    }
    metadata.update(
        split=split,
        groups=groups,
        cycle_protocol_sha256=digest(output / "protocol.json"),
    )
    for dataset, original in inputs:
        indices = [
            i
            for i, r in enumerate(original["records"])
            if r["step"] == 0 and r["group"] in groups
        ]
        records.extend(original["records"][i] for i in indices)
        for start in range(0, len(indices), 256):
            batch = select_batch(dataset, torch.tensor(indices[start : start + 256]))
            known += int(batch["value_mask"].sum())
            name = f"{split}-{len(shards):05d}.pt"
            path = output / name
            torch.save({"format": FORMAT, "metadata": metadata, "tensors": batch}, path)
            shards.append(
                {"file": name, "sha256": digest(path), "examples": len(batch["value"])}
            )
    metadata.update(records=records, game_ids=sorted({r["game_id"] for r in records}))
    torch.save(
        {"format": SHARD_FORMAT, "metadata": metadata, "shards": shards},
        output / f"{split}.pt",
    )
    split_reports[split] = {
        "groups": groups,
        "examples": len(records),
        "value_labels": known,
    }
assert (
    split_reports["train"]["examples"] == split_reports["train"]["value_labels"] == 995
)
assert (
    split_reports["validation"]["examples"] == 2160
    and split_reports["validation"]["value_labels"] == 651
)
(output / "splits.json").write_text(
    json.dumps(split_reports, indent=2), encoding="utf-8"
)
subprocess.run(
    [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "haojie_training.train",
        "--data",
        str(output / "train.pt"),
        "--validation",
        str(output / "validation.pt"),
        "--device",
        "cuda",
        "--precision",
        "bf16",
        "--threads",
        "2",
        "--steps",
        "256",
        "--batch-size",
        "16",
        "--seed",
        "2026092862",
        "--learning-rate",
        "0.0001",
        "--value-weight",
        "1",
        "--length-bucket-size",
        "128",
        "--checkpoint",
        str(output / "model.pt"),
        "--report",
        str(output / "report.json"),
    ],
    check=True,
)
