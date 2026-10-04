"""有界单遍原生审核/准备：每次仅保留一个分片，完成凭据确认后才发布数据。"""

import hashlib
import json
import sqlite3
import time
from pathlib import Path

import torch

from ..data import FORMAT, collate_examples
from ..model import ModelConfig
from ..prepare import digest
from .audit import binding
from .client import Client
from .pipeline import example, metadata


def prepare_record(client, path, output, shard_size=64):
    """客户端双帧队列产生背压；暂存张量没有完成凭据时不得进入训练。

    终局标签只在审核完成后逐片填写，不再重放轨迹，不持有整局特征。
    保存兼容既有 load_tensors 的独立训练分片，集合索引用 JSONL 流式消费。
    """
    if shard_size < 1:
        raise ValueError("分片大小必须为正")
    output = Path(output)
    temporary = output.with_name(output.name + ".partial")
    if output.exists():
        raise ValueError("拒绝覆盖已准备数据")
    temporary.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    expected = binding(client.ready)
    client.send({"op": "audit", "record": str(Path(path).resolve()), "encode": True})
    examples, records = [], []
    shards = samples = tensor_bytes = 0
    header = None
    meta = metadata(client.ready)

    def flush():
        nonlocal shards
        batch = collate_examples(examples, ModelConfig())
        torch.save(
            {"format": FORMAT, "metadata": {**meta, "records": list(records)}, "tensors": batch},
            temporary / f"shard-{shards:06d}.pt",
        )
        shards += 1
        examples.clear()
        records.clear()

    while True:
        row = client.receive()
        if row["type"] == "game":
            if header is not None:
                raise ValueError("重复审核头")
            header = row
            lineage = {
                k: header[k] for k in ("rulesHash", "start", "model", "samplerSeed", "policyKind")
            }
            identity = hashlib.sha256(
                json.dumps(lineage, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            group = f"{meta['ruleset']}:{header['start']['rules']}:{header['start']['seed']}"
            split = (
                "validation"
                if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 5 == 0
                else "train"
            )
            meta.update(
                behavior_models=[header["model"]],
                groups=[group],
                game_ids=[identity],
                policy_source="sampled-action-imitation",
                split=split,
            )
        elif row["type"] == "example":
            if header is None:
                raise ValueError("缺少审核头")
            tensor_bytes += row["bytes"]
            examples.append(example(row["input"], row["selected"]))
            records.append(
                {
                    **{k: row[k] for k in ("index", "step", "actor", "command", "stage")},
                    "game_id": identity,
                    "group": group,
                }
            )
            samples += 1
            if len(examples) == shard_size:
                flush()
        elif row["type"] == "outcome":
            pass
        elif row["type"] == "done":
            report = row["report"]
            if report.get("auditor") != expected or header is None:
                raise ValueError("审核器/规则/编码身份不匹配")
            if digest(path) != report.get("inputSha256"):
                raise ValueError("审核期间来源内容改变")
            if report["outcome"]["reason"] == "error":
                raise ValueError("实现错误轨迹不能进入学习")
            break
        else:
            raise ValueError("未知审核消息")
        del row
    if examples:
        flush()
    audit_seconds = time.perf_counter() - started
    labels = size = 0
    outcome = report["outcome"]
    with (temporary / "shards.jsonl").open("x", encoding="utf-8") as manifest:
        for i in range(shards):
            file = temporary / f"shard-{i:06d}.pt"
            payload = torch.load(file, weights_only=True)
            if outcome["terminated"]:
                for j, record in enumerate(payload["metadata"]["records"]):
                    if record["step"] == 0:
                        payload["tensors"]["value"][j] = outcome["returns"][str(record["actor"])]
                        payload["tensors"]["value_mask"][j] = True
                        labels += 1
            payload["metadata"].update(
                sources=[{"path": str(path), "sha256": report["inputSha256"]}], audit=expected
            )
            torch.save(payload, file)
            size += file.stat().st_size
            manifest.write(
                json.dumps(
                    {
                        "file": file.name,
                        "sha256": digest(file),
                        "examples": len(payload["tensors"]["value"]),
                        "split": split,
                    }
                )
                + "\n"
            )
            del payload
    result = {
        "audit": report,
        "samples": samples,
        "value_labels": labels,
        "shards": shards,
        "feature_bytes": tensor_bytes,
        "shard_bytes": size,
        "record_bytes": Path(path).stat().st_size,
        "split": split,
        "game_id": identity,
        "audit_encode_spool_seconds": audit_seconds,
        "seconds": time.perf_counter() - started,
        "max_buffered_examples": shard_size,
        "reader_queue_capacity": 2,
    }
    (temporary / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.rename(output)
    return result


def prepare_pool(executable, source, output, shard_size=64):
    """串行审核任务账本中所有已发布尝试；只有真实终局计入有效完整局。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect(
        f"file:{(Path(source) / 'tasks.sqlite').resolve().as_posix()}?mode=ro", uri=True
    )
    started = time.perf_counter()
    totals = {
        "valid_terminal": 0,
        "attempts": 0,
        "samples": 0,
        "value_labels": 0,
        "record_bytes": 0,
        "feature_bytes": 0,
        "shard_bytes": 0,
    }
    try:
        with (
            Client(executable) as client,
            (output / "audits.jsonl").open("x", encoding="utf-8") as stream,
        ):
            for task, attempt, path, status, latest in db.execute(
                "SELECT a.task,a.attempt,a.path,a.status,t.attempt "
                "FROM attempts a JOIN tasks t ON t.id=a.task ORDER BY a.task,a.attempt"
            ):
                actual = Path(path) if Path(path).exists() else Path(path + ".partial")
                if not actual.exists():
                    continue
                eligible = attempt == latest and status != "error"
                if eligible:
                    result = prepare_record(
                        client,
                        actual,
                        output / f"{task // 1000:06d}" / f"game-{task:09d}-attempt-{attempt:03d}",
                        shard_size,
                    )
                else:
                    # 旧尝试仍审核和计费，但不重复输出同一身份的监督或完整局。
                    client.send({"op": "audit", "record": str(actual), "encode": False})
                    result = {
                        "audit": client.receive()["report"],
                        "record_bytes": actual.stat().st_size,
                    }
                result.update(task=task, attempt=attempt, status=status, eligible=eligible)
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                totals["attempts"] += 1
                totals["valid_terminal"] += bool(
                    result["eligible"]
                    and result["audit"]["outcome"]["terminated"]
                    and result["audit"]["complete"]
                )
                for key in (
                    "samples",
                    "value_labels",
                    "record_bytes",
                    "feature_bytes",
                    "shard_bytes",
                ):
                    totals[key] += result.get(key, 0)
                del result
        totals["seconds"] = time.perf_counter() - started
        (output / "report.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
        return totals
    finally:
        db.close()
