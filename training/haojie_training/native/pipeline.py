"""复用实体 Transformer 与监督损失；采样行为模仿和终局价值验收，不冒充 PUCT。"""

import hashlib
import json
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import torch

from ..data import collate_examples
from ..model import ModelConfig, PolicyValueNet
from ..prepare import digest, save_split
from ..runtime import Trainer, checkpoint_config
from .client import Client
from .audit import audit, validate_audited


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
    executable, checkpoint, starts, output, commands=2000, plies=200, device="cpu", seed=20261003
):
    """每环境一个常驻进程和在途请求；就绪节点批量前向，不等待其他环境或逐属性 RPC。"""
    output.mkdir(parents=True, exist_ok=False)
    clients = []
    started = time.perf_counter()
    forwards = 0
    inference_seconds = 0.0
    try:
        for _ in starts:
            clients.append(Client(executable))
        model, model_hash = model_from(checkpoint, clients[0].ready, device)
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
        active = list(range(len(clients)))
        results = [None] * len(clients)
        executor = ThreadPoolExecutor(max_workers=len(clients))
        try:
            waiting = {executor.submit(clients[i].receive): i for i in active}
            while waiting:
                ready, _ = wait(waiting, return_when=FIRST_COMPLETED)
                # 每环境只保留一个请求；其他环境继续查询，已就绪输入按环境序号形成批次。
                messages = sorted((waiting.pop(future), future.result()) for future in ready)
                pending = []
                for i, message in messages:
                    if message["type"] == "done":
                        results[i] = message
                        if message.get("error"):
                            raise RuntimeError(message["error"])
                    elif message["type"] == "infer" and message["model"] == model_hash:
                        pending.append((i, message))
                    else:
                        raise ValueError("未知或过期模型请求")
                if pending:
                    batch = collate_examples(
                        [example(m["input"]) for _, m in pending], model.config
                    )
                    before = time.perf_counter()
                    with torch.inference_mode():
                        logits, values = model({k: v.to(device) for k, v in batch.items()})
                        logits, values = logits.cpu(), values.cpu()
                    inference_seconds += time.perf_counter() - before
                    if not torch.isfinite(logits).all() or not torch.isfinite(values).all():
                        raise ValueError("模型输出非有限")
                    forwards += 1
                    for row, (i, message) in enumerate(pending):
                        clients[i].send(
                            {
                                "id": message["id"],
                                "model": model_hash,
                                "logits": logits[row, : message["candidates"]].tolist(),
                            }
                        )
                for i, _ in pending:
                    waiting[executor.submit(clients[i].receive)] = i
        finally:
            # 中断先关闭自己启动的环境，唤醒阻塞读取；不等待九百秒超时才处理 Ctrl+C。
            for client in clients:
                client.close()
            executor.shutdown(wait=True, cancel_futures=True)
        report = {
            "games": results,
            "model": model_hash,
            "device": device,
            "forwards": forwards,
            "inference_seconds": inference_seconds,
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
