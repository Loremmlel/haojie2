"""调用同一TypeScript编码器，把教师JSONL按整局分组转换为PyTorch张量。"""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import torch

from .data import FLOAT_KEYS, FORMAT, INDEX_KEYS, MASK_KEYS, collate_examples, validate_batch
from .model import ModelConfig

ROOT = Path(__file__).resolve().parents[2]


def encoded_rows(path: Path, node: str):
    """Node仅输出已编码张量和分离的元数据；子进程失败时连同错误退出，不保留半成品。"""
    command = [
        node,
        str(ROOT / "node_modules/tsx/dist/cli.mjs"),
        str(ROOT / "scripts/training/encode.ts"),
        str(path.resolve()),
    ]
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as errors:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=errors, text=True, encoding="utf-8"
        )
        try:
            for line in process.stdout:
                yield json.loads(line)
            if process.wait() != 0:
                errors.seek(0)
                raise ValueError(f"TypeScript编码失败：\n{errors.read()}")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait()


def tensor_example(row: dict, config: ModelConfig) -> dict[str, torch.Tensor]:
    inputs = row["input"]
    if set(inputs) != (FLOAT_KEYS | MASK_KEYS | INDEX_KEYS) - {"policy", "value", "value_mask"}:
        raise ValueError("编码器输出不符合网络输入白名单")
    example = {
        key: torch.tensor(
            value,
            dtype=torch.float32
            if key in FLOAT_KEYS
            else torch.bool
            if key in MASK_KEYS
            else torch.long,
        )
        for key, value in inputs.items()
    }
    selected = row["selected"]
    if type(selected) is not int or not 0 <= selected < len(example["candidate_mask"]):
        raise ValueError("教师选择下标越界")
    example["policy"] = torch.zeros(len(example["candidate_mask"]))
    example["policy"][selected] = 1
    example["value"] = torch.tensor(0.0)
    example["value_mask"] = torch.tensor(False)
    validate_batch({key: value.unsqueeze(0) for key, value in example.items()}, config)
    return example


def prepare(
    paths: list[Path], output: Path, validation_fraction=0.25, split_seed=20260922, node="node"
) -> dict:
    """首批试验整体驻留CPU；整局和相同种子不可跨集合，不为截断/中断局制造价值标签。"""
    if output.exists():
        raise ValueError("输出目录已存在，请使用新的数据版本目录")
    if not 0 < validation_fraction < 1:
        raise ValueError("验证集比例必须在0与1之间")
    config = ModelConfig()
    header, groups, provenance = None, {}, []
    for path in paths:
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        provenance.append({"path": str(path.resolve()), "sha256": digest})
        current = None
        for row in encoded_rows(path, node):
            if row["type"] == "encoding":
                if header is not None and row != header:
                    raise ValueError("不能混合不同规则、编码或源码的数据")
                header = row
                schema = row["schema"]
                for name in ["entity_features", "global_features", "action_features", "kind_count"]:
                    if schema[name] != getattr(config, name):
                        raise ValueError(f"模型与编码器的{name}不匹配")
            elif row["type"] == "game":
                if row["group"] in groups:
                    raise ValueError("同一种子/规则对局重复，拒绝重采样污染统计")
                current = {"metadata": row, "examples": [], "records": [], "outcome": None}
                groups[row["group"]] = current
            elif row["type"] == "example":
                current["examples"].append(tensor_example(row, config))
                current["records"].append(
                    {key: row[key] for key in ["index", "step", "actor", "command", "stage"]}
                )
            elif row["type"] == "outcome":
                current["outcome"] = row
                for example, record in zip(current["examples"], current["records"]):
                    # 价值只训练根决策，避免重复计权，也不把条件动作前缀当成同一状态价值。
                    if row["terminated"] and record["step"] == 0:
                        example["value"].fill_(row["returns"][str(record["actor"])])
                        example["value_mask"].fill_(True)
                current = None
            else:
                raise ValueError("未知编码记录类型")
    if len(groups) < 2 or any(not game["examples"] for game in groups.values()):
        raise ValueError("至少需要两局非空记录，才能按整局分开训练和验证")
    if any(game["outcome"] is None for game in groups.values()):
        raise ValueError("编码流缺少显式终局/截断/中断标记")
    ordered = sorted(
        groups, key=lambda key: hashlib.sha256(f"{split_seed}:{key}".encode()).digest()
    )
    count = min(len(ordered) - 1, max(1, round(len(ordered) * validation_fraction)))
    splits = {"train": ordered[count:], "validation": ordered[:count]}
    metadata = {
        "synthetic": False,
        "ruleset": header["ruleset"],
        "encoding": header["schema"]["encoding"],
        "schema": header["schema"],
        "source_sha256": header["source_sha256"],
        "sources": provenance,
        "split_seed": split_seed,
    }
    report = {**metadata, "splits": {}, "games": []}
    for key, game in groups.items():
        report["games"].append(
            {**game["metadata"], **game["outcome"], "group": key, "examples": len(game["examples"])}
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=output.parent) as folder:
        for split, keys in splits.items():
            examples = [e for key in keys for e in groups[key]["examples"]]
            records = [{"group": key, **r} for key in keys for r in groups[key]["records"]]
            batch = collate_examples(examples, config)
            details = {
                "groups": keys,
                "examples": len(examples),
                "decisions": sum(r["step"] == 0 for r in records),
                "value_labels": int(batch["value_mask"].sum()),
                "forced_examples": int((batch["candidate_mask"].sum(1) == 1).sum()),
                "max_entities": batch["entities"].shape[1],
                "max_candidates": batch["candidates"].shape[1],
                "stages": dict(Counter(r["stage"] for r in records)),
                "commands": dict(Counter(r["command"] for r in records if r["step"] == 0)),
            }
            report["splits"][split] = details
            torch.save(
                {
                    "format": FORMAT,
                    "metadata": {**metadata, "split": split, "groups": keys, "records": records},
                    "tensors": batch,
                },
                Path(folder) / f"{split}.pt",
            )
        (Path(folder) / "manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        Path(folder).rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=20260922)
    parser.add_argument("--node", default=shutil.which("node") or "node")
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = prepare(args.inputs, args.output, args.validation_fraction, args.split_seed, args.node)
    print(json.dumps({"output": str(args.output), "splits": report["splits"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
