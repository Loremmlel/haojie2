"""教师拟合指标；强制单候选与真正需要选择的样本分别统计，不代表对战胜率。"""

import torch

from .data import select_batch
from .runtime import autocast


def value_baselines(dataset, records=None):
    """仅用训练集真实根收益拟合常数；验证标签不得参与基线拟合。"""
    mask = dataset["value_mask"]
    targets = dataset["value"]
    result = {"mean": float(targets[mask].mean()) if mask.any() else 0.0, "by_actor": {}}
    if records is not None:
        actors = torch.tensor([r["actor"] for r in records])
        for actor in (1, 2):
            selected = mask & (actors == actor)
            if selected.any():
                result["by_actor"][str(actor)] = float(targets[selected].mean())
    return result


def value_diagnostics(predictions, dataset, records, baselines):
    """按决策席位和整局报告误差；未结束局及参数分步不参与价值统计。"""
    targets, known = dataset["value"], dataset["value_mask"]
    actors = torch.tensor([r["actor"] for r in records]) if records is not None else None
    seat_prediction = (
        torch.tensor(
            [baselines["by_actor"].get(str(int(actor)), baselines["mean"]) for actor in actors]
        )
        if actors is not None
        else torch.full_like(targets, baselines["mean"])
    )

    def summarize(mask):
        count = int(mask.sum())
        if not count:
            return {"labels": 0}
        truth, predicted = targets[mask], predictions[mask]
        return {
            "labels": count,
            "wins": int((truth > 0).sum()),
            "losses": int((truth < 0).sum()),
            "target_mean": float(truth.mean()),
            "prediction_mean": float(predicted.mean()),
            "model_mse": float((predicted - truth).square().mean()),
            "zero_mse": float(truth.square().mean()),
            "train_mean_mse": float((truth - baselines["mean"]).square().mean()),
            "train_actor_mean_mse": float((truth - seat_prediction[mask]).square().mean()),
        }

    result = {"baselines": baselines, "overall": summarize(known)}
    if actors is not None:
        result["by_actor"] = {str(a): summarize(known & (actors == a)) for a in (1, 2)}
        games = {}
        for index, record in enumerate(records):
            if known[index]:
                games.setdefault(record["game_id"], []).append(index)
        errors = [
            (predictions[indices] - targets[indices]).square().mean() for indices in games.values()
        ]
        result["games"] = len(games)
        result["game_macro_mse"] = float(torch.stack(errors).mean()) if errors else None
    return result


@torch.inference_mode()
def evaluate(model, dataset, device, precision, batch_size, records=None, baselines=None):
    was_training = model.training
    model.eval()
    nll = correct = choices = choice_correct = roots = root_correct = known = squared = 0
    size = len(dataset["value"])
    predictions = []
    try:
        for start in range(0, size, batch_size):
            end = min(size, start + batch_size)
            batch = select_batch(dataset, torch.arange(start, end))
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast(device, precision):
                logits, value = model(batch)
            predictions.append(value.cpu())
            hit = logits.argmax(1) == batch["policy"].argmax(1)
            multiple = batch["candidate_mask"].sum(1) > 1
            nll += float(-(batch["policy"] * logits.log_softmax(1)).sum())
            correct += int(hit.sum())
            choices += int(multiple.sum())
            choice_correct += int((hit & multiple).sum())
            mask = batch["value_mask"]
            known += int(mask.sum())
            squared += float(((value - batch["value"]).square() * mask).sum())
            if records is not None:
                root = torch.tensor([r["step"] == 0 for r in records[start:end]], device=device)
                root &= multiple
                roots += int(root.sum())
                root_correct += int((hit & root).sum())
    finally:
        model.train(was_training)
    return {
        "examples": size,
        "policy_loss": nll / size,
        "accuracy": correct / size,
        "choice_examples": choices,
        "choice_accuracy": choice_correct / choices if choices else None,
        "root_choices": roots,
        "root_accuracy": root_correct / roots if roots else None,
        "value_labels": known,
        "value_mse": squared / known if known else None,
        "value_diagnostics": value_diagnostics(
            torch.cat(predictions),
            dataset,
            records,
            baselines or {"mean": 0.0, "by_actor": {}},
        ),
    }


def validate_split(training: dict, validation: dict):
    for key in ["ruleset", "encoding", "schema", "source_sha256", "synthetic"]:
        if training.get(key) != validation.get(key):
            raise ValueError(f"训练集与验证集的{key}不一致")
    train_groups, validation_groups = training.get("groups", []), validation.get("groups", [])
    if not train_groups or not validation_groups or set(train_groups) & set(validation_groups):
        raise ValueError("必须使用有整局分组且相互不重叠的训练/验证集")


def main():
    """重新评估已有检查点；保持原数据划分，输出新报告，不改写历史实验。"""
    import argparse
    import hashlib
    import json
    from pathlib import Path

    from .data import load_dataset
    from .model import PolicyValueNet
    from .runtime import checkpoint_config, resolve_device

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.report.exists() or min(args.batch_size, args.threads) < 1:
        parser.error("报告须为新路径，批量与线程数须为正")
    torch.set_num_threads(args.threads)
    config = checkpoint_config(args.checkpoint)
    train, train_meta = load_dataset(args.data, config)
    validation, validation_meta = load_dataset(args.validation, config)
    validate_split(train_meta, validation_meta)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    digests = {}
    for key, path in (("dataset_sha256", args.data), ("validation_sha256", args.validation)):
        with path.open("rb") as source:
            digests[key] = hashlib.file_digest(source, "sha256").hexdigest()
        if digests[key] != payload["metadata"].get(key):
            raise ValueError("诊断数据与检查点原始训练/验证文件不一致")
    with args.checkpoint.open("rb") as source:
        digests["checkpoint_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    if "overfit_indices" in payload["metadata"]:
        indices = payload["metadata"]["overfit_indices"]
        train = select_batch(train, torch.tensor(indices))
        if train_meta.get("records") is not None:
            train_meta = {**train_meta, "records": [train_meta["records"][i] for i in indices]}
    device = resolve_device(args.device)
    model = PolicyValueNet(config).to(device)
    model.load_state_dict(payload["model"])
    baselines = value_baselines(train, train_meta.get("records"))
    report = {
        "checkpoint": str(args.checkpoint),
        "digests": digests,
        "device": str(device),
        "precision": args.precision,
        "value_weight": payload.get("value_weight", 1.0),
    }
    for name, data, metadata in (
        ("train", train, train_meta),
        ("validation", validation, validation_meta),
    ):
        report[name] = evaluate(
            model, data, device, args.precision, args.batch_size, metadata.get("records"), baselines
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
