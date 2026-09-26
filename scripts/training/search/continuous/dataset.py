"""连续实验的数据边界：复用重放编码，保留完整命令前缀，再组合历史池和固定验证族。"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from haojie_training.data import (
    FORMAT,
    SHARD_FORMAT,
    collate_examples,
    load_dataset,
    select_batch,
)
from haojie_training.evaluate import evaluate, validate_split, value_baselines, value_quality
from haojie_training.model import ModelConfig, PolicyValueNet
from haojie_training.prepare import load_file


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def save_split(output, name, chunks, metadata, records):
    """逐块落盘；源张量只读，分片身份与索引使用同一份来源摘要。"""
    shards = []
    known = 0
    for batch in chunks:
        file = f"{name}-{len(shards):05d}.pt"
        torch.save({"format": FORMAT, "metadata": metadata, "tensors": batch}, output / file)
        shards.append(
            {
                "file": file,
                "sha256": digest(output / file),
                "examples": len(batch["value"]),
            }
        )
        known += int(batch["value_mask"].sum())
    if not shards:
        raise ValueError("没有实际搜索根，不能把空数据标记为训练完成")
    meta = {
        **metadata,
        "records": records,
        "groups": sorted({r["group"] for r in records}),
        "game_ids": sorted({r["game_id"] for r in records}),
    }
    torch.save(
        {"format": SHARD_FORMAT, "metadata": meta, "shards": shards},
        output / f"{name}.pt",
    )
    return {
        "groups": meta["groups"],
        "examples": len(records),
        "value_labels": known,
        "shards": len(shards),
    }


def encode(path, output):
    config = ModelConfig()
    header, games, source = load_file(path, "node", config, None, True)
    records, examples = [], []
    for identity, game in games.items():
        outcome = game["outcome"]
        if outcome is None or outcome["interrupted"]:
            raise ValueError("中断或缺失结束标记的对局不能进入连续训练池")
        for row, example in zip(game["records"], game["examples"]):
            known = outcome["terminated"] and row["step"] == 0
            assert bool(example["value_mask"]) == known
            assert float(example["value"]) == (
                outcome["returns"][str(row["actor"])] if known else 0
            )
            examples.append(example)
            records.append({**row, "game_id": identity, "group": game["metadata"]["group"]})
    metadata = {
        "synthetic": False,
        "ruleset": header["ruleset"],
        "encoding": header["schema"]["encoding"],
        "schema": header["schema"],
        "source_sha256": header["source_sha256"],
        "sources": [source],
        "policy_source": header["policy_source"],
    }
    result = save_split(
        output,
        "decisions",
        (collate_examples(examples[i : i + 256], config) for i in range(0, len(examples), 256)),
        metadata,
        records,
    )
    return {
        "source": source,
        "decisions": result,
        "games": [{**g["metadata"], **g["outcome"]} for g in games.values()],
    }


def combine(spec, output):
    config = ModelConfig()
    inputs = {
        split: [load_dataset(Path(p), config) for p in paths] for split, paths in spec.items()
    }
    reference = inputs["train"][0][1]
    sources = [
        {"path": str(Path(p).resolve()), "sha256": digest(p)}
        for paths in spec.values()
        for p in paths
    ]
    metadata = {k: reference[k] for k in ("synthetic", "ruleset", "encoding", "schema")}
    # 这里只组合已经验证的同编码张量；各源原始指纹保留，组合摘要不冒充引擎源码哈希。
    metadata.update(
        source_sha256=hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        tensor_sources=sources,
        policy_source="teacher-assisted-conditional-visits-v1",
    )
    split_records, selected = {}, {}
    for split, pairs in inputs.items():
        split_records[split], selected[split] = [], []
        seen = set()
        for data, meta in pairs:
            if any(
                meta.get(k) != reference.get(k)
                for k in ("synthetic", "ruleset", "encoding", "schema")
            ):
                raise ValueError("历史池规则或编码不一致")
            indices = []
            for i, record in enumerate(meta["records"]):
                if record["step"] != 0:
                    assert not bool(data["value_mask"][i]) and float(data["value"][i]) == 0
                identity = (record["game_id"], record["index"], record["step"])
                if identity in seen:
                    raise ValueError("历史池重复根决策")
                seen.add(identity)
                indices.append(i)
                split_records[split].append(record)
            selected[split].append((data, indices))
    groups = {s: {r["group"] for r in rows} for s, rows in split_records.items()}
    if not groups["train"] or not groups["validation"] or groups["train"] & groups["validation"]:
        raise ValueError("训练池与固定验证族为空或交叉")
    report = {"tensor_sources": sources, "splits": {}}
    for split, pairs in selected.items():
        chunks = (
            select_batch(data, torch.tensor(indices[i : i + 256]))
            for data, indices in pairs
            for i in range(0, len(indices), 256)
        )
        report["splits"][split] = save_split(
            output, split, chunks, {**metadata, "split": split}, split_records[split]
        )
    return report


def check(spec):
    report = json.loads(Path(spec["report"]).read_text(encoding="utf-8"))
    payload = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
    assert report["steps"] == report["updates"] == payload["updates"] == spec["steps"]
    assert report["health"]["finite_parameters"] and report["health"]["finite_gradients"]
    assert all(torch.isfinite(v).all() for v in payload["model"].values())
    assert all(
        not isinstance(v, torch.Tensor) or torch.isfinite(v).all()
        for state in payload["optimizer"]["state"].values()
        for v in state.values()
    )
    config = ModelConfig(**payload["config"])
    data, metadata = load_dataset(Path(spec["validation"]), config)
    indices = data["value_mask"].nonzero().flatten()[:32]
    if len(indices) == 0:
        raise ValueError("固定验证集没有真实价值标签")
    model = PolicyValueNet(config).eval()
    model.load_state_dict(payload["model"])
    metrics = evaluate(
        model,
        select_batch(data, indices),
        torch.device("cpu"),
        "fp32",
        16,
        [metadata["records"][i] for i in indices.tolist()],
    )

    def finite(value):
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, list):
            return all(finite(v) for v in value)
        return not isinstance(value, float) or math.isfinite(value)

    assert finite(report) and finite(metrics)
    return {
        "passed": True,
        "updates": payload["updates"],
        "checkpoint_sha256": digest(spec["checkpoint"]),
        "fp32_probe_indices": indices.tolist(),
        "fp32_probe": metrics,
        "note": "固定32个已知价值状态的数值复核，不是独立棋力评测",
    }


def qualify(spec):
    """长跑前重算全验证集，旧战术题或有限数值不能代替真实价值泛化。"""
    payload = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
    config = ModelConfig(**payload["config"])
    data, metadata = load_dataset(Path(spec["validation"]), config)
    train, train_metadata = load_dataset(Path(spec["train"]), config)
    validate_split(train_metadata, metadata)
    if any(
        payload["metadata"].get(key) != metadata.get(key)
        for key in ("ruleset", "encoding", "schema", "synthetic")
    ):
        raise ValueError("检查点与价值验证集的规则或编码不一致")
    if not all(torch.isfinite(value).all() for value in payload["model"].values()):
        raise ValueError("价值模型包含非有限参数")
    seen = set(
        payload["metadata"].get("seen_training_groups", payload["metadata"].get("groups", []))
    )
    if not seen or seen & set(metadata["groups"]):
        raise ValueError("价值验证族缺少可核验谱系或已经参与训练")
    # 覆盖不足直接拒绝，避免对单胜方大库做无意义的完整前向。
    from haojie_training.evaluate import value_diagnostics

    coverage = value_diagnostics(
        torch.zeros_like(data["value"]),
        data,
        metadata["records"],
        value_baselines(train, train_metadata["records"]),
    )
    if any(coverage["by_winner"][w]["groups"] < 4 for w in ("1", "2")):
        return {
            "passed": False,
            "quality": value_quality(coverage),
            "coverage": coverage,
            "checkpoint_sha256": digest(spec["checkpoint"]),
        }
    model = PolicyValueNet(config).eval()
    model.load_state_dict(payload["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate(
        model.to(device),
        data,
        device,
        "fp32",
        16,
        metadata["records"],
        value_baselines(train, train_metadata["records"]),
    )
    quality = value_quality(metrics["value_diagnostics"])
    return {
        "passed": quality["passed"],
        "quality": quality,
        "metrics": metrics,
        "checkpoint_sha256": digest(spec["checkpoint"]),
        "precision": "fp32",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["encode", "combine", "check", "qualify"])
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    report = (
        encode(args.input, args.output)
        if args.mode == "encode"
        else (
            combine(json.loads(args.input.read_text(encoding="utf-8")), args.output)
            if args.mode == "combine"
            else (qualify if args.mode == "qualify" else check)(
                json.loads(args.input.read_text(encoding="utf-8"))
            )
        )
    )
    (args.output / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
