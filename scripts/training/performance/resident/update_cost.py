"""正式审核后只做一次短更新成本测量；训练副本不保存、不参与行为采样。"""

import json
import sys
import time
from pathlib import Path

import torch

from haojie_training.data import load_dataset, select_batch
from haojie_training.model import ModelConfig, PolicyValueNet
from haojie_training.prepare import digest
from haojie_training.runtime import Trainer

source, checkpoint, output = map(Path, sys.argv[1:])
assert not output.exists()
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(20261004)
began = time.perf_counter()
model_hash = digest(checkpoint)
payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
config = ModelConfig(**payload["config"])
rules_hash = payload["metadata"]["rules_package_sha256"]
model = PolicyValueNet(config)
model.load_state_dict(payload["model"])
trainer = Trainer(model, torch.device("cuda"), "bf16")
del payload
games = []
for line in (source / "audits.jsonl").read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    if (
        row.get("eligible")
        and row.get("split") == "train"
        and row.get("samples", 0) >= 4
    ):
        games.append(row)
total = sum(r["samples"] for r in games)
assert total >= 128
selected = []
for i in range(32):
    position = int((i + 0.5) * total / 32)
    for game in games:
        if position < game["samples"]:
            break
        position -= game["samples"]
    folder = (
        source
        / f"{game['task'] // 1000:06d}"
        / f"game-{game['task']:09d}-attempt-{game['attempt']:03d}"
    )
    # 分片为32样本；末片不足4时取前片。位置按整个训练划分等距预定，不挑最快形状。
    shard = min(position // 32, max(0, (game["samples"] - 4) // 32))
    selected.append((folder / f"shard-{shard:06d}.pt", position % 32))
setup = time.perf_counter() - began
rows = []
for i, (path, offset) in enumerate([*selected[:2], *selected]):
    started = time.perf_counter()
    data, metadata = load_dataset(path, config)
    assert metadata["rules_package_sha256"] == rules_hash
    assert metadata["behavior_models"] == [model_hash]
    offset = min(offset, len(data["value"]) - 4)
    batch = select_batch(data, torch.arange(offset, offset + 4))
    del data
    loaded = time.perf_counter()
    shape = {
        "entities": batch["entities"].shape[1],
        "candidates": batch["candidates"].shape[1],
    }
    batch = {k: v.to("cuda") for k, v in batch.items()}
    loss = trainer.step(batch)
    torch.cuda.synchronize()
    finished = time.perf_counter()
    assert torch.isfinite(loss)
    rows.append(
        {
            "warmup": i < 2,
            "file": str(path),
            "offset": offset,
            **shape,
            "load_seconds": loaded - started,
            "update_seconds": finished - loaded,
            "seconds": finished - started,
            "loss": float(loss),
        }
    )
    del batch, loss
measured = rows[2:]
health = trainer.health()
assert all(
    health[k] for k in ("finite_parameters", "finite_gradients", "finite_optimizer")
)
assert trainer.updates == 34 and digest(checkpoint) == model_hash
result = {
    "model": model_hash,
    "precision": "bf16",
    "batch": 4,
    "threads": 1,
    "training_samples": total,
    "measured_samples": 128,
    "updates": trainer.updates,
    "setup_seconds": setup,
    "warmup_seconds": sum(r["seconds"] for r in rows[:2]),
    "samples_per_second_including_load": 128 / sum(r["seconds"] for r in measured),
    "samples_per_second_update_only": 128 / sum(r["update_seconds"] for r in measured),
    "health": health,
    "steps": rows,
    "total_seconds": time.perf_counter() - began,
    "note": "32批短测；训练副本未保存，检查点字节未变；只估算更新成本，不代表棋力或精确长期吞吐",
}
output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k != "steps"}))
