"""复用实体 Transformer 与监督损失；采样行为模仿和终局价值验收，不冒充 PUCT。"""

import hashlib
import json
import queue
import tempfile
import time
from pathlib import Path

import torch

from ..data import collate_examples
from ..model import ModelConfig, PolicyValueNet
from ..prepare import digest, save_split
from ..runtime import Trainer, checkpoint_config, resolve_device
from .audit import audit, validate_audited
from .client import Client
from .execution import PolicyInference


def metadata(ready):
    # decision_stages 是原生树协议扩展，模型沿用原 ENCODING_SCHEMA。
    schema = {k: v for k, v in ready["schema"].items() if k != "decision_stages"}
    return {
        "synthetic": False,
        "ruleset": ready["ruleset"],
        "encoding": schema["encoding"],
        "schema": schema,
        "source_sha256": ready["rulesHash"],
        "rules_package_sha256": ready["rulesHash"],
    }


def initialize(executable, path, seed=20261003, tiny=False):
    if path.exists():
        raise ValueError("拒绝覆盖检查点")
    with Client(executable) as client:
        meta = metadata(client.ready)
    torch.manual_seed(seed)
    trainer = Trainer(
        PolicyValueNet(ModelConfig.tiny() if tiny else ModelConfig()), torch.device("cpu")
    )
    meta.update(
        {
            "initialization_seed": seed,
            "status": "untrained",
            "policy_source": "sampled-action-imitation",
        }
    )
    trainer.save(path, meta)
    return {
        "checkpoint": str(path),
        "sha256": digest(path),
        "parameters": sum(p.numel() for p in trainer.model.parameters()),
    }


def model_from(path, ready, device="cpu"):
    config = checkpoint_config(path)
    payload = torch.load(path, weights_only=True, map_location="cpu")
    expected = metadata(ready)
    if any(
        payload["metadata"].get(key) != expected[key]
        for key in ("ruleset", "encoding", "schema", "rules_package_sha256")
    ):
        raise ValueError("检查点规则/编码/规则包哈希不匹配")
    model = PolicyValueNet(config)
    model.load_state_dict(payload["model"])
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("检查点包含非有限参数")
    return model.to(device).eval(), digest(path)


def example(inputs, selected=0):
    count = len(inputs["candidate_mask"])
    policy = torch.zeros(count)
    policy[selected] = 1
    return {
        **inputs,
        "policy": policy,
        "value": torch.tensor(0.0),
        "value_mask": torch.tensor(False),
    }


def sample(
    executable,
    checkpoint,
    starts,
    output,
    commands=2000,
    plies=200,
    device="cpu",
    seed=20261003,
    *,
    precision="fp32",
    batch_wait_ms=0.0,
):
    """每环境一个常驻进程和在途请求；就绪节点批量前向，不等待其他环境或逐属性 RPC。"""
    output.mkdir(parents=True, exist_ok=False)
    clients = []
    started = time.perf_counter()
    forwards = 0
    inference_seconds = 0.0
    device = resolve_device(str(device))
    if not 0 <= batch_wait_ms <= 10:
        raise ValueError("组批等待须在0至10毫秒之间")
    events = queue.Queue()
    queue_seconds = send_seconds = 0.0
    response_times = []
    try:
        for i, _ in enumerate(starts):
            clients.append(Client(executable, events=events, identity=i))
        model, model_hash = model_from(checkpoint, clients[0].ready, device)
        inference = PolicyInference(model, device, precision)
        for i, (client, start) in enumerate(zip(clients, starts)):
            if client.ready != clients[0].ready:
                raise ValueError("工作进程版本不一致")
            client.send(
                {
                    "op": "sample",
                    "record": str((output / f"game-{i}.jsonl").resolve()),
                    "start": start,
                    "model": model_hash,
                    "samplerSeed": (seed + i) & 0xFFFFFFFF,
                    "maxCommands": commands,
                    "maxPlies": plies,
                }
            )
        active = set(range(len(clients)))
        results = [None] * len(clients)
        while active:
            before = time.perf_counter()
            try:
                messages = [events.get(timeout=900)]
            except queue.Empty:
                raise TimeoutError("原生请求超过保护时间") from None
            deadline = time.perf_counter() + batch_wait_ms / 1000
            while len(messages) < len(active):
                try:
                    messages.append(events.get(timeout=max(0, deadline - time.perf_counter())))
                except queue.Empty:
                    break
            queue_seconds += time.perf_counter() - before
            pending = []
            for i, raw in messages:
                message = clients[i].checked(raw)
                if i not in active:
                    raise ValueError("已结束环境重复响应")
                if message["type"] == "done":
                    results[i] = message
                    active.remove(i)
                    if message.get("error"):
                        raise RuntimeError(message["error"])
                elif message["type"] == "infer" and message["model"] == model_hash:
                    pending.append((i, message))
                else:
                    raise ValueError("未知或过期模型请求")
            # 就绪请求按实体长度分组，最大实体填充倍率不超过2；不等未就绪环境。
            pending.sort(key=lambda item: item[1]["entities"])
            while pending:
                limit = pending[0][1]["entities"] * 2
                count = next(
                    (j for j, (_, m) in enumerate(pending) if m["entities"] > limit), len(pending)
                )
                group, pending = pending[:count], pending[count:]
                logits = inference([m["input"] for _, m in group])
                forwards += 1
                for row, (i, message) in enumerate(group):
                    sent = time.perf_counter()
                    clients[i].send(
                        {
                            "id": message["id"],
                            "model": model_hash,
                            "logits": logits[row, : message["candidates"]].tolist(),
                        }
                    )
                    send_seconds += time.perf_counter() - sent
                    response_times.append(time.perf_counter() - message["received_at"])
        inference_seconds = inference.execution_seconds
        report = {
            "games": results,
            "model": model_hash,
            "device": str(device),
            "precision": precision,
            "batch_wait_ms": batch_wait_ms,
            "forwards": forwards,
            "inference_seconds": inference_seconds,
            "collation_seconds": inference.collation_seconds,
            "queue_seconds": queue_seconds,
            "send_seconds": send_seconds,
            "tensor_read_seconds": sum(c.read_seconds for c in clients),
            "decode_validate_seconds": sum(c.decode_seconds for c in clients),
            "tensor_bytes": sum(c.tensor_bytes for c in clients),
            "requests": inference.requests,
            "mean_batch": inference.requests / max(1, forwards),
            "entity_fill": inference.entities / max(1, inference.entity_slots),
            "candidate_fill": inference.candidates / max(1, inference.candidate_slots),
            "response_p95_seconds": sorted(response_times)[int((len(response_times) - 1) * 0.95)]
            if response_times
            else 0,
            "seconds": time.perf_counter() - started,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        for client in clients:
            client.close()


def prepare(executable, paths, output, shard_size=64, *, audited=None):
    if output.exists():
        raise ValueError("数据目录已存在")
    games = []
    meta = None
    audit_identity = None
    paths = list(paths)
    if audited is not None:
        audited = list(audited)
        validate_audited(executable, paths, audited)
    sources = []
    for i, path in enumerate(paths):
        result = audited[i] if audited is not None else audit(executable, path)
        current = metadata(result.ready)
        if meta is not None and meta != current:
            raise ValueError("来源规则版本不一致")
        meta = current
        current_audit = {**result.report["auditor"], "binary_sha256": result.engine_sha256}
        if audit_identity is not None and audit_identity != current_audit:
            raise ValueError("来源审核器身份不一致")
        audit_identity = current_audit
        examples = [example(inputs, selected) for inputs, selected in result.inputs]
        records, header, outcome = result.records, result.header, result.report["outcome"]
        if not examples:
            raise ValueError("空监督或未审核记录")
        if outcome["reason"] == "error":
            raise ValueError("实现错误轨迹不进入学习")
        for item, record in zip(examples, records):
            if outcome["terminated"] and record["step"] == 0:
                item["value"] = torch.tensor(float(outcome["returns"][str(record["actor"])]))
                item["value_mask"] = torch.tensor(True)
        group = f"{meta['ruleset']}:{header['start']['rules']}:{header['start']['seed']}"
        # 改变截断预算不产生新轨迹身份，防止同一前缀在恢复后重复计样。
        lineage = {
            k: header[k] for k in ("rulesHash", "start", "model", "samplerSeed", "policyKind")
        }
        identity = hashlib.sha256(
            json.dumps(lineage, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        records = [{**r, "group": group, "game_id": identity} for r in records]
        games.append((group, identity, examples, records, header["model"]))
        sources.append({"sha256": result.report["inputSha256"], "path": str(path)})
    families = sorted({g[0] for g in games})
    if len(families) < 2 or len({g[1] for g in games}) != len(games):
        raise ValueError("至少两个种子族，拒绝重复对局")
    meta.update(
        {
            "policy_source": "sampled-action-imitation",
            "behavior_models": sorted({g[4] for g in games}),
            "sources": sources,
            # 全部来源已逐一核对同一个审核器，避免在每个来源重复保存完整编码表。
            "audit": audit_identity,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="native-prepare-", dir=output.parent) as temporary:
        folder = Path(temporary)
        report = {"metadata": meta, "splits": {}}
        for name, groups in (("validation", families[:1]), ("train", families[1:])):
            rows = [(e, r) for g in games if g[0] in groups for e, r in zip(g[2], g[3])]
            chunks = (
                collate_examples([e for e, _ in rows[i : i + shard_size]], ModelConfig())
                for i in range(0, len(rows), shard_size)
            )
            report["splits"][name] = save_split(
                folder, name, chunks, {**meta, "split": name}, [r for _, r in rows]
            )
        (folder / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        folder.rename(output)
    return report
