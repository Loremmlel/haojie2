"""教师拟合指标；强制单候选与真正需要选择的样本分别统计，不代表对战胜率。"""

import torch

from .data import select_batch
from .runtime import autocast


@torch.inference_mode()
def evaluate(model, dataset, device, precision, batch_size, records=None):
    was_training = model.training
    model.eval()
    nll = correct = choices = choice_correct = roots = root_correct = known = squared = 0
    size = len(dataset["value"])
    try:
        for start in range(0, size, batch_size):
            end = min(size, start + batch_size)
            batch = select_batch(dataset, torch.arange(start, end))
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast(device, precision):
                logits, value = model(batch)
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
    }


def validate_split(training: dict, validation: dict):
    for key in ["ruleset", "encoding", "schema", "source_sha256", "synthetic"]:
        if training.get(key) != validation.get(key):
            raise ValueError(f"训练集与验证集的{key}不一致")
    train_groups, validation_groups = training.get("groups", []), validation.get("groups", [])
    if not train_groups or not validation_groups or set(train_groups) & set(validation_groups):
        raise ValueError("必须使用有整局分组且相互不重叠的训练/验证集")
