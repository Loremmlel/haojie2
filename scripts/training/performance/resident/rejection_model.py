"""用原失败前缀和同一模型随机状态短测回溯；晚盘排错不计入自然局产能。"""

import json
import sys
from pathlib import Path

import torch

from haojie_training.native.client import Client
from haojie_training.native.pipeline import sample

budget = "--budget" in sys.argv
record, engine, checkpoint, output = map(Path, [a for a in sys.argv[1:] if a != "--budget"])
rows = [json.loads(line)["body"] for line in record.read_text().splitlines()]
played = [row for row in rows if row["type"] == "sample"]
assert not played[-1]["optional"]
start = {
    **rows[0]["start"],
    "prelude": [{"actor": r["actor"], "command": r["command"]} for r in played],
}
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
report = sample(engine, checkpoint, [start], output, 4, 1000, "cuda", played[-1]["samplerState"])
if budget:
    assert report["games"][0]["outcome"]["reason"] == "decode-budget", report
else:
    assert report["games"][0]["metrics"]["rejected"] > 0, report
with Client(engine) as client:
    client.send({"op": "audit", "record": str(output / "game-0.jsonl"), "encode": False})
    audit = client.receive()["report"]
if budget:
    assert audit["outcome"]["reason"] == "decode-budget" and not audit["outcome"]["terminated"], audit
else:
    assert audit["rejected"] > 0 and audit["commands"] == 4, audit
(output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps({"audit": audit, "seconds": report["seconds"]}))
