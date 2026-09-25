"""核验整族搜索张量与真实价值，执行两次优化及FP32重载；这是通路验收，不评估棋力。"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from haojie_training.data import load_dataset, select_batch
from haojie_training.model import ModelConfig, PolicyValueNet, policy_value_loss
from haojie_training.runtime import Trainer


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(2026092861)
    config = ModelConfig()
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    games = {g["game_id"]: g for g in manifest["games"]}
    assert all(g["source"] == "search-selfplay" for g in games.values())
    datasets, metadata, results = {}, {}, {}
    picked = {}
    for split in ("train", "validation"):
        data, meta = load_dataset(args.data / f"{split}.pt", config)
        assert meta["policy_source"] == "teacher-assisted-conditional-visits-v1"
        assert meta["groups"] == manifest["splits"][split]["groups"]
        datasets[split], metadata[split] = data, meta
        records = meta["records"]
        soft = roots = known = 0
        for start in range(0, len(records), 32):
            batch = select_batch(
                data, torch.arange(start, min(start + 32, len(records)))
            )
            for local, record in enumerate(records[start : start + 32]):
                index = start + local
                game = games[record["game_id"]]
                expected = game["terminated"] and record["step"] == 0
                assert bool(batch["value_mask"][local]) == expected
                assert float(batch["value"][local]) == (
                    game["returns"][str(record["actor"])] if expected else 0.0
                )
                is_soft = int((batch["policy"][local] > 0).sum()) > 1
                soft += is_soft
                roots += record["step"] == 0
                known += expected
                if split == "train":
                    if is_soft:
                        picked.setdefault("soft", index)
                    if not expected:
                        picked.setdefault("masked", index)
                    if expected:
                        picked.setdefault(f"value-{record['actor']}", index)
        assert known == manifest["splits"][split]["value_labels"]
        assert roots == manifest["splits"][split]["decisions"]
        results[split] = {
            "examples": len(records),
            "roots": roots,
            "soft": soft,
            "value_labels": known,
            "groups": meta["groups"],
            "index_sha256": digest(args.data / f"{split}.pt"),
        }
    assert not set(metadata["train"]["groups"]) & set(metadata["validation"]["groups"])
    assert not {2026092801, 2026092802} & {g["seed"] for g in games.values()}
    assert set(picked) == {"soft", "masked", "value-1", "value-2"}
    indices = list(picked.values())
    cpu_batch = select_batch(datasets["train"], torch.tensor(indices))
    assert bool(((cpu_batch["policy"] > 0).sum(1) > 1).any())
    assert cpu_batch["value_mask"].any() and not cpu_batch["value_mask"].all()
    device = torch.device("cuda")
    trainer = Trainer(PolicyValueNet(config), device, "bf16", lr=1e-4, value_weight=0.1)
    initial = next(trainer.model.parameters()).detach().cpu().clone()
    batch = {key: value.to(device) for key, value in cpu_batch.items()}
    losses = [float(trainer.step(batch)) for _ in range(2)]
    health = trainer.health()
    assert all(health[k] for k in ("finite_parameters", "finite_gradients"))
    assert trainer.steps == trainer.updates == 2
    assert torch.isfinite(torch.tensor(losses)).all()
    assert not torch.equal(initial, next(trainer.model.parameters()).detach().cpu())
    provenance = {
        "purpose": "pipeline-only-not-strength",
        "manifest_sha256": digest(args.data / "manifest.json"),
        "selected_indices": indices,
        "selection_roles": picked,
        "seed": 2026092861,
        "ruleset": manifest["ruleset"],
        "encoding": manifest["encoding"],
        "schema": manifest["schema"],
        "policy_source": manifest["policy_source"],
    }
    checkpoint = args.output / "model.pt"
    trainer.save(checkpoint, provenance)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["steps"] == payload["updates"] == 2
    assert all(torch.isfinite(v).all() for v in payload["model"].values())
    for state in payload["optimizer"]["state"].values():
        assert all(
            not isinstance(v, torch.Tensor) or torch.isfinite(v).all()
            for v in state.values()
        )
    restored = PolicyValueNet(config).eval()
    restored.load_state_dict(payload["model"])
    with torch.inference_mode():
        logits, values = restored(cpu_batch)
        assert torch.isfinite(logits).all() and torch.isfinite(values).all()
        fp32_loss = float(policy_value_loss((logits, values), cpu_batch, 0.1))
    report = {
        "passed": True,
        "purpose": "pipeline-only-not-strength",
        "splits": results,
        "provenance": provenance,
        "parameters": sum(p.numel() for p in restored.parameters()),
        "device": torch.cuda.get_device_name(),
        "precision": "bf16",
        "value_weight": 0.1,
        "steps": trainer.steps,
        "updates": trainer.updates,
        "skipped_updates": 0,
        "losses": losses,
        "health": health,
        "fp32_reload_loss": fp32_loss,
        "fp32_outputs_finite": True,
        "optimizer_finite": True,
        "checkpoint_sha256": digest(checkpoint),
        "script_sha256": digest(__file__),
    }
    with (args.output / "report.json").open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
