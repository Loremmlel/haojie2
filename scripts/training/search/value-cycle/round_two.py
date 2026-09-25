"""用上一轮模型参与生成的新自对弈标签更新同一网络，核对谱系、隔离与真实优化步。"""

import gzip
import hashlib
import json
import sys
from pathlib import Path

import torch

from haojie_training.data import load_dataset, select_batch
from haojie_training.model import ModelConfig, PolicyValueNet, policy_value_loss
from haojie_training.runtime import Trainer
from haojie_training.train import initialize_weights


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


parent_dir, data_dir, output = map(Path, sys.argv[1:4])
output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(2)
torch.manual_seed(2026092863)
parent = parent_dir / "model.pt"
parent_sha = digest(parent)
payload = torch.load(parent, map_location="cpu", weights_only=True)
assert payload["steps"] == payload["updates"] == 256
config = ModelConfig(**payload["config"])
manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
games = {g["game_id"]: g for g in manifest["games"]}
assert all(g["value_model_sha256"] == parent_sha for g in games.values())
assert not {2026092901} & {g["seed"] for g in games.values()}
old_groups = {
    g
    for s in json.loads((parent_dir / "splits.json").read_text()).values()
    for g in s["groups"]
}
assert not old_groups & {g["group"] for g in games.values()}
active = set()
for source in manifest["sources"]:
    path = Path(source["path"])
    assert digest(path) == source["sha256"]
    for line in gzip.open(path, "rt", encoding="utf-8"):
        row = json.loads(line)
        if row["type"] == "game":
            identity = row["gameId"]
            assert (
                row["experimentKind"] == "selfplay"
                and row["valueModelSha256"] == parent_sha
            )
        elif (
            row["type"] == "sample"
            and row["policyMode"] == "search"
            and row["valueStats"]["calls"] > 0
        ):
            active.add((identity, row["index"]))
datasets, metas, checks = {}, {}, {}
for split in ("train", "validation"):
    data, meta = load_dataset(data_dir / f"{split}.pt", config)
    for index, record in enumerate(meta["records"]):
        game = games[record["game_id"]]
        known = game["terminated"] and record["step"] == 0
        assert bool(data["value_mask"][index]) == known
        assert float(data["value"][index]) == (
            game["returns"][str(record["actor"])] if known else 0.0
        )
    datasets[split], metas[split] = data, meta
    checks[split] = {
        "groups": meta["groups"],
        "examples": len(meta["records"]),
        "value_labels": int(data["value_mask"].sum()),
    }
assert not set(metas["train"]["groups"]) & set(metas["validation"]["groups"])
indices = []
for index, record in enumerate(metas["train"]["records"]):
    if record["step"] != 0 or (record["game_id"], record["index"]) not in active:
        continue
    example = select_batch(datasets["train"], torch.tensor([index]))
    if int((example["policy"][0] > 0).sum()) > 1:
        indices.append(index)
    if len(indices) == 4:
        break
assert len(indices) == 4
batch = select_batch(datasets["train"], torch.tensor(indices))
model = PolicyValueNet(config)
trainer = Trainer(model, torch.device("cuda"), "bf16", lr=1e-4, value_weight=1.0)
metadata = {k: v for k, v in metas["train"].items() if k != "records"}
metadata["selected_indices"] = indices
initialize_weights(trainer, parent, metadata, metas["validation"]["groups"])
initial = next(trainer.model.parameters()).detach().cpu().clone()
gpu_batch = {k: v.to("cuda") for k, v in batch.items()}
losses = [float(trainer.step(gpu_batch)) for _ in range(2)]
health = trainer.health()
assert trainer.steps == trainer.updates == 2 and all(
    health[k] for k in ("finite_parameters", "finite_gradients")
)
assert torch.isfinite(torch.tensor(losses)).all()
assert not torch.equal(initial, next(trainer.model.parameters()).detach().cpu())
trainer.save(output / "model.pt", metadata)
reloaded = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
assert all(torch.isfinite(v).all() for v in reloaded["model"].values())
assert all(
    not isinstance(v, torch.Tensor) or torch.isfinite(v).all()
    for s in reloaded["optimizer"]["state"].values()
    for v in s.values()
)
cpu_model = PolicyValueNet(config).eval()
cpu_model.load_state_dict(reloaded["model"])
with torch.inference_mode():
    prediction = cpu_model(batch)
    assert all(torch.isfinite(v).all() for v in prediction)
    fp32_loss = float(policy_value_loss(prediction, batch))
assert digest(parent) == parent_sha
report = {
    "passed": True,
    "purpose": "neural-feedback-cycle-smoke-not-strength",
    "parent_sha256": parent_sha,
    "parent_updates": 256,
    "round_updates": 2,
    "optimizer_reset_explicit": True,
    "actual_new_neural_root_indices": indices,
    "losses": losses,
    "health": health,
    "fp32_loss": fp32_loss,
    "splits": checks,
    "checkpoint_sha256": digest(output / "model.pt"),
    "script_sha256": digest(Path(__file__)),
}
with (output / "report.json").open("x", encoding="utf-8") as destination:
    json.dump(report, destination, ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False))
