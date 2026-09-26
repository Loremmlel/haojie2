"""冻结策略主干，仅用实际完整对局训练价值头；独立族分组门槛决定能否接入研究搜索。"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from haojie_training.data import load_dataset, select_batch
from haojie_training.evaluate import value_baselines, value_diagnostics, value_quality
from haojie_training.model import ModelConfig

from scripts.training.improvement.data import digest
from scripts.training.improvement.serve import load_checkpoint


@torch.inference_mode()
def features(model, data, indices=None):
    """捕获现有价值头输入，不改变策略计算、公开输入或主干参数；只缓存小型张量表示。"""
    parts = []
    handle = model.value.register_forward_pre_hook(
        lambda _, inputs: parts.append(inputs[0].detach().cpu())
    )
    indices = torch.arange(len(data["value"])) if indices is None else indices
    try:
        for start in range(0, len(indices), 32):
            batch = select_batch(data, indices[start : start + 32])
            model({k: v.to(next(model.parameters()).device) for k, v in batch.items()})
    finally:
        handle.remove()
    return torch.cat(parts)


def known_data(path):
    data, metadata = load_dataset(path, ModelConfig.tiny())
    if metadata.get("policy_source") != "value-only-behavior":
        raise ValueError("价值试验只接受共享重放校验的实际行为收益，不能使用纠错反事实")
    selected = torch.where(data["value_mask"])[0]
    records = [metadata["records"][i] for i in selected.tolist()]
    if not records or any(r["step"] != 0 for r in records):
        raise ValueError("价值数据必须含真实终局根收益")
    return data, {**metadata, "records": records}, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=2026101601)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("步数必须为正")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    training, train_meta, train_ids = known_data(args.data)
    validation, valid_meta, valid_ids = known_data(args.validation)
    parent = torch.load(args.parent, map_location="cpu", weights_only=True)
    seen = set(train_meta["groups"]) | set(parent["training_groups"])
    if seen & set(valid_meta["groups"]):
        raise ValueError("价值校准族与策略或价值训练族交叉")
    for metadata in (train_meta, valid_meta):
        if any(
            metadata[k] != parent["metadata"][k]
            for k in ("encoding", "ruleset", "schema", "source_sha256")
        ):
            raise ValueError("输入规则或编码版本不同")
    protocol = {
        "format": "haojie-frozen-policy-value-fit-v1",
        "steps": args.steps,
        "seed": args.seed,
        "batch_size": 64,
        "learning_rate": 0.001,
        "device": args.device,
        "policy_updates": 0,
        "selection": "fixed final step; no calibration updates",
        "training_groups": sorted(seen),
        "validation_groups": valid_meta["groups"],
        "inputs": {str(p): digest(p) for p in (args.parent, args.data, args.validation)},
        "script_sha256": digest(__file__),
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (args.output / "train.py").write_bytes(Path(__file__).read_bytes())
    _, model, _, _ = load_checkpoint(args.parent)
    model.to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # 分片逐批前向，禁止把整库补到同一最大实体数；这里只保留每个局面的64维表示。
    train_x = features(model, training, train_ids).clone().to(args.device)
    training = {k: training[k][train_ids] for k in ("value", "value_mask")}
    valid_x = features(model, validation, valid_ids).clone().to(args.device)
    validation = {k: validation[k][valid_ids] for k in ("value", "value_mask")}
    train_y = training["value"].to(args.device)
    # 按种子族和真实收益符号均衡抽样，长局及一方连走不能淹没少数胜负。
    buckets = {}
    for i, row in enumerate(train_meta["records"]):
        buckets.setdefault(row["group"], {}).setdefault(float(training["value"][i]), []).append(i)
    if any(not {-1.0, 1.0} <= set(outcomes) for outcomes in buckets.values()):
        raise ValueError("每个价值训练族都必须同时覆盖胜负实际操作者")
    groups = list(buckets.values())
    rng = torch.Generator().manual_seed(args.seed)

    def pick(n):
        return int(torch.randint(n, (), generator=rng))

    optimizer = torch.optim.AdamW(model.value.parameters(), lr=0.001)
    torch.manual_seed(args.seed)
    for layer in model.value.modules():
        if isinstance(layer, torch.nn.Linear):
            layer.reset_parameters()
    for parameter in model.value.parameters():
        parameter.requires_grad_(True)
    batch_hash = hashlib.sha256()
    for _ in range(args.steps):
        indices = []
        for _ in range(64):
            group = groups[pick(len(groups))]
            choices = group[(-1.0, 1.0)[pick(2)]]
            indices.append(choices[pick(len(choices))])
        ids = torch.tensor(indices, device=args.device)
        batch_hash.update(ids.cpu().numpy().tobytes())
        optimizer.zero_grad(set_to_none=True)
        prediction = model.value(train_x[ids]).squeeze(-1)
        loss = (prediction - train_y[ids]).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.value.parameters(), 1)
        optimizer.step()
        if not torch.isfinite(loss):
            raise FloatingPointError("价值训练出现非有限损失")
    with torch.inference_mode():
        predictions = model.value(valid_x).squeeze(-1).cpu()
    baselines = value_baselines(training, train_meta["records"])
    diagnostics = value_diagnostics(predictions, validation, valid_meta["records"], baselines)
    gate = value_quality(diagnostics)
    phases = {}
    for phase in sorted({r["phase"] for r in valid_meta["records"]}):
        ids = torch.tensor([i for i, r in enumerate(valid_meta["records"]) if r["phase"] == phase])
        subset = {k: validation[k][ids] for k in ("value", "value_mask")}
        phase_report = value_diagnostics(
            predictions[ids], subset, [valid_meta["records"][i] for i in ids], baselines
        )
        phases[phase] = {"gate": value_quality(phase_report), "diagnostics": phase_report}
    covered = [phase for phase, row in phases.items() if row["gate"]["passed"]]
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise FloatingPointError("价值训练产物包含非有限参数")
    if any(
        not torch.equal(v, parent["model"][k])
        for k, v in state.items()
        if not k.startswith("value.")
    ):
        raise AssertionError("价值训练改变了策略参数")
    if digest(__file__) != protocol["script_sha256"]:
        raise ValueError("价值训练期间源码改变")
    checkpoint = args.output / "model.pt"
    certificate = {
        "passed": gate["passed"] and "play" in covered,
        "covered_phases": covered,
        "gate": gate,
        "diagnostics": diagnostics,
        "by_phase": phases,
        "policy_unchanged": True,
        "batch_sha256": batch_hash.hexdigest(),
    }
    parent.update(
        model=state,
        parent=str(args.parent),
        parent_sha256=digest(args.parent),
        training_groups=sorted(seen),
        protocol_sha256=digest(args.output / "protocol.json"),
        value_certificate=certificate,
        value_updates=args.steps,
    )
    parent.pop("optimizer", None)
    torch.save(parent, checkpoint)
    certificate["checkpoint_sha256"] = digest(checkpoint)
    (args.output / "certificate.json").write_text(
        json.dumps(certificate, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "passed": certificate["passed"],
                "covered_phases": covered,
                "gate": gate,
                "overall": diagnostics["overall"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
