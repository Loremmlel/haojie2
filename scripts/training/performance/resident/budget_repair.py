"""本轮已确认的预算分类修复：备份账本、保留旧尝试，只迁移已冻结的兼容构建。"""

import json
import sqlite3
import sys
import time
from pathlib import Path

from haojie_training.native.client import Client
from haojie_training.native.resident import Ledger
from haojie_training.prepare import digest

folder, old_engine, new_engine = map(Path, sys.argv[1:])
expected = "4dae7d29d341394f1363dfef14d620f4c1d32cf14f9692a40d88ef379ee3eae3"
assert digest(old_engine) == expected, "只允许已复核的本轮冻结构建迁移"
with Client(old_engine) as old, Client(new_engine) as new:
    for key in ("rulesHash", "ruleset", "schema", "protocol"):
        assert old.ready[key] == new.ready[key], key
with sqlite3.connect(folder / "tasks.sqlite") as source:
    config = json.loads(source.execute("SELECT data FROM config").fetchone()[0])
assert config["engine"] == expected
ledger = Ledger(folder, config, resume=True)
try:
    errors = list(ledger.db.execute("SELECT id,result FROM tasks WHERE status='error'"))
    assert errors and all(json.loads(r["result"])["error"] == "参数解码预算耗尽" for r in errors)
    backup = folder / "tasks.before-budget-repair.sqlite"
    assert not backup.exists() and not (folder / "budget-repair.json").exists()
    with sqlite3.connect(backup) as target:
        ledger.db.backup(target)
    updated = {**config, "engine": digest(new_engine)}
    evidence = {
        "time": time.time(), "before": config, "after": updated,
        "tasks": [r["id"] for r in errors],
        "reason": "仅预算耗尽分类改变；旧错误记录及尝试不改写、不计终局；完成任务跳过",
    }
    (folder / "budget-repair.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    ledger.db.execute("UPDATE config SET data=?", (json.dumps(updated, sort_keys=True),))
    for row in errors:
        ledger.db.execute("UPDATE tasks SET status='decode-budget' WHERE id=?", (row["id"],))
    ledger.db.commit()
    print(json.dumps(evidence))
finally:
    ledger.close()
