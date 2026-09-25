"""调用同一TypeScript编码器，把教师JSONL按整局分组转换为PyTorch张量。"""

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from .data import (
    FLOAT_KEYS,
    FORMAT,
    INDEX_KEYS,
    MASK_KEYS,
    SHARD_FORMAT,
    collate_examples,
    validate_batch,
)
from .model import ModelConfig

ROOT = Path(__file__).resolve().parents[2]


def encoded_rows(path: Path, node: str, search_policy: bool = False):
    """Node仅输出已编码张量和分离的元数据；子进程失败时连同错误退出，不保留半成品。"""
    # 只判别协议路由；规则、命令和终态仍必须由TypeScript完整重放，不能静默丢弃访问分布。
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as source:
        first_line = next((line for line in source if line.strip()), None)
    if first_line is None:
        raise ValueError("训练记录为空")
    first = json.loads(first_line)
    if first.get("policyKind") == "teacher-assisted-restricted-puct-v1" and not search_policy:
        raise ValueError("搜索轨迹必须显式使用--search-policy，不能降为教师one-hot标签")
    command = [
        node,
        str(ROOT / "node_modules/tsx/dist/cli.mjs"),
        str(
            ROOT
            / (
                "scripts/training/search/bootstrap/encoding.ts"
                if search_policy
                else "scripts/training/encode.ts"
            )
        ),
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
    if "policy" in row:
        example["policy"] = torch.tensor(row["policy"], dtype=torch.float32)
    else:
        example["policy"] = torch.zeros(len(example["candidate_mask"]))
        example["policy"][selected] = 1
    example["value"] = torch.tensor(0.0)
    example["value_mask"] = torch.tensor(False)
    validate_batch({key: value.unsqueeze(0) for key, value in example.items()}, config)
    return example


def load_file(
    path: Path,
    node: str,
    config: ModelConfig,
    teacher_difficulty: str | None,
    search_policy: bool = False,
):
    """独立重放一个来源文件并编码样本；只返回局部分组，跨文件一致性由主线程校验。"""
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    header, groups, current = None, {}, None
    for row in encoded_rows(path, node, search_policy):
        if row["type"] == "encoding":
            if header is not None and row != header:
                raise ValueError("不能混合不同规则、编码或源码的数据")
            header = row
            schema = row["schema"]
            for name in ["entity_features", "global_features", "action_features", "kind_count"]:
                if schema[name] != getattr(config, name):
                    raise ValueError(f"模型与编码器的{name}不匹配")
        elif row["type"] == "game":
            identity = row.get("game_id", row["group"])
            if identity in groups:
                raise ValueError("同一教师对局重复，拒绝重采样污染统计")
            current = {"metadata": row, "examples": [], "records": [], "outcome": None}
            groups[identity] = current
        elif row["type"] == "example":
            if teacher_difficulty is not None:
                game = current["metadata"]
                profile = (game.get("teachers") or {}).get(str(row["actor"]))
                difficulty = (
                    (profile or {}).get("difficulty")
                    if game.get("teachers") is not None
                    else game.get("difficulty")
                )
                if difficulty not in {"easy", "medium", "hard"}:
                    raise ValueError("样本缺少可验证的教师来源，不能按难度筛选")
                if difficulty != teacher_difficulty:
                    continue
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
    if header is None:
        raise ValueError("编码流缺少版本头")
    return header, groups, {"path": str(path.resolve()), "sha256": digest}


def prepare(
    paths: list[Path],
    output: Path,
    validation_fraction=0.25,
    split_seed=20260922,
    node="node",
    teacher_difficulty: str | None = None,
    workers: int = 1,
    shard_size: int = 0,
    split_manifest: Path | None = None,
    search_policy: bool = False,
) -> dict:
    """首批试验整体驻留CPU；整局和相同种子不可跨集合，不为截断/中断局制造价值标签。"""
    if output.exists():
        raise ValueError("输出目录已存在，请使用新的数据版本目录")
    if not 0 < validation_fraction < 1:
        raise ValueError("验证集比例必须在0与1之间")
    if teacher_difficulty not in {None, "easy", "medium", "hard"}:
        raise ValueError("教师筛选难度无效")
    if search_policy and teacher_difficulty is not None:
        raise ValueError("搜索访问分布不能使用教师难度筛选")
    if workers < 1:
        raise ValueError("并发数必须大于0")
    if shard_size < 0:
        raise ValueError("分片样本上限不能为负")
    config = ModelConfig()
    header, groups, provenance = None, {}, []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for file_header, file_groups, source in executor.map(
            lambda path: load_file(path, node, config, teacher_difficulty, search_policy), paths
        ):
            if header is not None and file_header != header:
                raise ValueError("不能混合不同规则、编码或源码的数据")
            header = file_header
            if groups.keys() & file_groups.keys():
                raise ValueError("同一教师对局重复，拒绝重采样污染统计")
            groups.update(file_groups)
            provenance.append(source)
    families = {game["metadata"]["group"] for game in groups.values()}
    if len(families) < 2 or any(not game["examples"] for game in groups.values()):
        raise ValueError("至少需要两个不同种子族且各局筛选后非空，才能分开训练和验证")
    if any(game["outcome"] is None for game in groups.values()):
        raise ValueError("编码流缺少显式终局/截断/中断标记")
    ordered = sorted(
        families, key=lambda key: hashlib.sha256(f"{split_seed}:{key}".encode()).digest()
    )
    count = min(len(ordered) - 1, max(1, round(len(ordered) * validation_fraction)))
    splits = {"train": ordered[count:], "validation": ordered[:count]}
    split_reference = None
    if split_manifest is not None:
        reference = json.loads(split_manifest.read_text(encoding="utf-8"))
        if (
            reference["ruleset"] != header["ruleset"]
            or reference["encoding"] != header["schema"]["encoding"]
        ):
            raise ValueError("固定划分的规则/编码不匹配")
        old_train = reference["splits"]["train"]["groups"]
        old_validation = reference["splits"]["validation"]["groups"]
        original = set(old_train) | set(old_validation)
        if (
            not old_train
            or not old_validation
            or set(old_train) & set(old_validation)
            or not original <= families
        ):
            raise ValueError("固定划分缺少原种子族或存在交叉")
        splits = {"train": [*old_train, *sorted(families - original)], "validation": old_validation}
        split_seed = reference["split_seed"]
        split_reference = hashlib.sha256(split_manifest.read_bytes()).hexdigest()
    metadata = {
        "synthetic": False,
        "ruleset": header["ruleset"],
        "encoding": header["schema"]["encoding"],
        "schema": header["schema"],
        "source_sha256": header["source_sha256"],
        "sources": provenance,
        "split_seed": split_seed,
        "teacher_difficulty": teacher_difficulty,
        **({"policy_source": header["policy_source"]} if "policy_source" in header else {}),
        **({"split_reference_sha256": split_reference} if split_reference else {}),
    }
    report = {**metadata, "splits": {}, "games": []}
    for key, game in groups.items():
        report["games"].append(
            {
                **game["metadata"],
                **game["outcome"],
                "game_id": key,
                "examples": len(game["examples"]),
                "selected_decisions": sum(r["step"] == 0 for r in game["records"]),
                "excluded_decisions": game["outcome"]["commands"]
                - sum(r["step"] == 0 for r in game["records"]),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=output.parent) as folder:
        for split, split_families in splits.items():
            keys = [
                key
                for family in split_families
                for key, game in groups.items()
                if game["metadata"]["group"] == family
            ]
            examples = [e for key in keys for e in groups[key]["examples"]]
            records = [
                {"group": groups[key]["metadata"]["group"], "game_id": key, **r}
                for key in keys
                for r in groups[key]["records"]
            ]
            details = {
                "groups": split_families,
                "game_ids": keys,
                "examples": len(examples),
                "decisions": sum(r["step"] == 0 for r in records),
                "value_labels": sum(bool(e["value_mask"]) for e in examples),
                "forced_examples": sum(int(e["candidate_mask"].sum()) == 1 for e in examples),
                "max_entities": max(len(e["entity_mask"]) for e in examples),
                "max_candidates": max(len(e["candidate_mask"]) for e in examples),
                "stages": dict(Counter(r["stage"] for r in records)),
                "commands": dict(Counter(r["command"] for r in records if r["step"] == 0)),
            }
            report["splits"][split] = details
            split_metadata = {
                **metadata,
                "split": split,
                "groups": split_families,
                "game_ids": keys,
                "records": records,
            }
            if shard_size:
                shards = []
                for start in range(0, len(examples), shard_size):
                    batch = collate_examples(examples[start : start + shard_size], config)
                    name = f"{split}-{len(shards):05d}.pt"
                    path = Path(folder) / name
                    torch.save({"format": FORMAT, "metadata": metadata, "tensors": batch}, path)
                    with path.open("rb") as source:
                        digest = hashlib.file_digest(source, "sha256").hexdigest()
                    shards.append({"file": name, "sha256": digest, "examples": len(batch["value"])})
                    del batch
                payload = {"format": SHARD_FORMAT, "metadata": split_metadata, "shards": shards}
                details["shards"] = len(shards)
            else:
                payload = {
                    "format": FORMAT,
                    "metadata": split_metadata,
                    "tensors": collate_examples(examples, config),
                }
            torch.save(payload, Path(folder) / f"{split}.pt")
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
    parser.add_argument("--teacher-difficulty", choices=["easy", "medium", "hard"])
    parser.add_argument("--workers", type=int, default=1, help="并行编码来源文件的进程数")
    parser.add_argument("--shard-size", type=int, default=0)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument(
        "--search-policy", action="store_true", help="只编码自对弈真实搜索访问分布，排除回退及评测"
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = prepare(
        args.inputs,
        args.output,
        args.validation_fraction,
        args.split_seed,
        args.node,
        args.teacher_difficulty,
        args.workers,
        args.shard_size,
        args.split_manifest,
        args.search_policy,
    )
    print(json.dumps({"output": str(args.output), "splits": report["splits"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
