"""进程内的已审核编码结果；不是可从外部文件反序列化的缓存凭据。

原生进程对同一次读取重放、校验、编码并计算原始字节哈希。复用时核对当前内容、
完整规则/编码及实际审核器身份，随后只消费已经拥有的张量，不再打开记录生成特征。
"""
from dataclasses import dataclass
from pathlib import Path

from ..prepare import digest
from .client import Client


def binding(ready):
    engine = ready.get("engine", {})
    if engine.get("auditor") != "haojie-native-audit-v1":
        raise ValueError("需要提供内容绑定的正式原生审核器")
    return {"version": engine["auditor"], "build": engine["build"]["sourceSha256"],
            "rulesHash": ready["rulesHash"], "ruleset": ready["ruleset"], "schema": ready["schema"]}


@dataclass(frozen=True)
class AuditedRecord:
    ready: dict
    engine_sha256: str
    header: dict
    report: dict
    inputs: tuple
    records: tuple


def audit(executable, path):
    """同步审核且保留编码；完整行损坏失败，未完成尾部保留已确认前缀和 unknown。"""
    executable = Path(executable).resolve(strict=True)
    engine_hash = digest(executable)
    with Client(executable) as client:
        expected = binding(client.ready)
        client.send({"op": "audit", "record": str(Path(path).resolve()), "encode": True})
        inputs, records, header = [], [], None
        while True:
            row = client.receive()
            if row["type"] == "game":
                header = row
            elif row["type"] == "example":
                inputs.append((row["input"], row["selected"]))
                records.append({k: row[k] for k in ("index", "step", "actor", "command", "stage")})
            elif row["type"] == "outcome":
                pass
            elif row["type"] == "done":
                report = row["report"]
                if report.get("auditor") != expected or header is None:
                    raise ValueError("审核器/规则/编码身份不匹配")
                raw_hash = report.get("inputSha256")
                if not isinstance(raw_hash, str) or len(raw_hash) != 64:
                    raise ValueError("审核缺少原始内容身份")
                return AuditedRecord(client.ready, engine_hash, header, report, tuple(inputs), tuple(records))
            else:
                raise ValueError("未知编码消息")


def validate_audited(executable, paths, results):
    """外部可变路径只用于当前内容核对；通过后输出使用原审核结果的拥有型张量。"""
    if len(paths) != len(results) or not results:
        raise ValueError("已审核结果数量不匹配")
    engine_hash = digest(Path(executable).resolve(strict=True))
    with Client(executable) as client:
        expected = binding(client.ready)
    for path, result in zip(paths, results):
        if not isinstance(result, AuditedRecord) or result.engine_sha256 != engine_hash:
            raise ValueError("已审核结果的实际二进制不匹配")
        if result.report.get("auditor") != expected or binding(result.ready) != expected:
            raise ValueError("审核器/规则/编码身份不匹配")
        if digest(Path(path)) != result.report["inputSha256"]:
            raise ValueError("原始记录内容已经变化，必须重新审核")
