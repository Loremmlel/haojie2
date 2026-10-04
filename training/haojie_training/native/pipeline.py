"""复用实体 Transformer 与监督损失；采样行为模仿和终局价值验收，不冒充 PUCT。"""

import hashlib
import json
import tempfile
from pathlib import Path

import torch

from ..data import collate_examples
from ..model import ModelConfig, PolicyValueNet
from ..prepare import digest, save_split
from ..runtime import Trainer, checkpoint_config
from .audit import audit, validate_audited
from .client import Client
from .pool import Pool


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
    """兼容单批入口：同一调度实现，每起点仅分配一次。"""
    output.mkdir(parents=True, exist_ok=False)
    jobs = BatchJobs(starts, output, seed, commands, plies)
    pool = Pool(executable, checkpoint, len(starts), device, precision, batch_wait_ms)
    report = pool.run(jobs)
    report["games"] = jobs.results
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


class BatchJobs:
    """有限短测任务；种子与旧 sample 入口逐项兼容，也用于常驻补位对照。"""

    def __init__(self, starts, output, seed, commands, plies):
        self.starts, self.output, self.seed = starts, output, seed
        self.commands, self.plies, self.next = commands, plies, 0
        self.results = [None] * len(starts)

    def stop(self):
        return False

    def claim(self):
        if self.next >= len(self.starts):
            return None
        i = self.next
        self.next += 1
        return {
            "id": i,
            "start": self.starts[i],
            "sampler_seed": (self.seed + i) & 0xFFFFFFFF,
            "record": (self.output / f"game-{i}.jsonl").resolve(),
        }

    def finish(self, task, message):
        self.results[task["id"]] = message

    def failed(self, task, error):
        raise RuntimeError(error)

    def unfinished(self, task):
        pass


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
