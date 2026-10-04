"""单机持久任务账本。SQLite 保存身份/尝试/完成集合，内存仅保留工作槽与统计窗口。"""

import hashlib
import json
import os
import signal
import sqlite3
import time
from pathlib import Path

from ..prepare import digest
from .client import Client
from .pool import Pool


def task_identity(seed, number, rules="mixed"):
    """版本化整数派生；与槽位、并发、完成顺序、进程和运行时限无关。"""
    key = f"haojie-resident-v1:{seed}:{number}".encode()
    raw = hashlib.sha256(key).digest()
    return {
        "id": number,
        "start": {
            "seed": int.from_bytes(raw[:4], "little") or 1,
            "rules": ("classic" if number % 2 == 0 else "shrine") if rules == "mixed" else rules,
        },
        "sampler_seed": int.from_bytes(raw[4:8], "little"),
    }


class Ledger:
    def __init__(self, output, config, *, resume=False, tasks=None, target=None):
        self.output = Path(output).resolve()
        if resume and not (self.output / "tasks.sqlite").is_file():
            raise ValueError("恢复目录缺少任务账本")
        self.output.mkdir(parents=True, exist_ok=resume)
        self.lock = (self.output / "owner.lock").open("a+b")
        self.lock.write(b"0")
        self.lock.flush()
        self.lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise ValueError("输出目录已有运行中的采样器") from None
        self.db = sqlite3.connect(self.output / "tasks.sqlite")
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS config (data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY, start TEXT NOT NULL, sampler_seed INTEGER NOT NULL,
                status TEXT NOT NULL, attempt INTEGER NOT NULL DEFAULT 0, result TEXT);
            CREATE INDEX IF NOT EXISTS task_status ON tasks(status,id);
            CREATE TABLE IF NOT EXISTS attempts (
                task INTEGER, attempt INTEGER, run INTEGER, path TEXT, status TEXT, result TEXT,
                PRIMARY KEY(task,attempt));
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY, started REAL, config TEXT, report TEXT);
        """)
        previous = self.db.execute("SELECT data FROM config").fetchone()
        if previous and json.loads(previous[0]) != config:
            self.close()
            raise ValueError("恢复配置/模型/引擎哈希不匹配")
        if not previous:
            self.db.execute("INSERT INTO config VALUES (?)", (json.dumps(config, sort_keys=True),))
        self.config, self.tasks, self.target = config, tasks, target
        self.commands, self.plies = config["commands"], config["plies"]
        self.db.commit()
        self.run = None
        self.terminal = self.db.execute(
            "SELECT count(*) FROM tasks WHERE status='terminal'"
        ).fetchone()[0]
        self.next_id = self.db.execute("SELECT coalesce(max(id)+1,0) FROM tasks").fetchone()[0]

    def recover(self, executable):
        if self.db.execute("SELECT 1 FROM tasks WHERE status='error' LIMIT 1").fetchone():
            raise ValueError("账本已有规则/协议错误，拒绝跳过错误后继续恢复")
        # 完成文件先发布、账本后提交；崩溃夹缝通过原生审核补记，绝不覆盖旧前缀。
        with Client(executable) as client:
            for row in self.db.execute("SELECT * FROM tasks WHERE status='running'").fetchall():
                attempt = self.db.execute(
                    "SELECT * FROM attempts WHERE task=? AND attempt=?", (row["id"], row["attempt"])
                ).fetchone()
                path = Path(attempt["path"])
                if path.exists():
                    client.send({"op": "audit", "record": str(path), "encode": False})
                    report = client.receive()["report"]
                    result = {"outcome": report["outcome"], "recovered_audit": report}
                    reason = report["outcome"]["reason"]
                    status = (
                        "terminal"
                        if report["outcome"]["terminated"]
                        else (
                            reason
                            if reason in {"commands", "plies", "decode-budget"}
                            else "pending"
                        )
                    )
                    if report["outcome"]["reason"] == "error":
                        raise ValueError("恢复发现规则/协议错误，须先修复原因")
                else:
                    result, status = (
                        {"reason": "interrupted", "prefix": str(path) + ".partial"},
                        "pending",
                    )
                # 恢复只补充分类，保留上次停止时已经落账的请求数与历时。
                result = {**json.loads(attempt["result"] or "{}"), **result}
                self.db.execute(
                    "UPDATE tasks SET status=?,result=? WHERE id=?",
                    (status, json.dumps(result), row["id"]),
                )
                self.db.execute(
                    "UPDATE attempts SET status=?,result=? WHERE task=? AND attempt=?",
                    (status, json.dumps(result), row["id"], row["attempt"]),
                )
            self.db.commit()
        self.terminal = self.db.execute(
            "SELECT count(*) FROM tasks WHERE status='terminal'"
        ).fetchone()[0]

    def begin(self, options):
        cursor = self.db.execute(
            "INSERT INTO runs(started,config) VALUES (?,?)", (time.time(), json.dumps(options))
        )
        self.run = cursor.lastrowid
        self.db.commit()

    def stop(self):
        return self.target is not None and self.terminal >= self.target

    def claim(self):
        if self.stop():
            return None
        row = self.db.execute(
            "SELECT * FROM tasks WHERE status='pending' ORDER BY id LIMIT 1"
        ).fetchone()
        if row:
            task = {
                "id": row["id"],
                "start": json.loads(row["start"]),
                "sampler_seed": row["sampler_seed"],
                "attempt": row["attempt"] + 1,
            }
        else:
            if self.tasks is not None and self.next_id >= self.tasks:
                return None
            task = task_identity(self.config["seed"], self.next_id, self.config["rules"])
            task["attempt"] = 1
            self.next_id += 1
            self.db.execute(
                "INSERT INTO tasks(id,start,sampler_seed,status) VALUES (?,?,?,'pending')",
                (task["id"], json.dumps(task["start"]), task["sampler_seed"]),
            )
        # 每千个任务一组路径，防止百万文件堆在一个目录。
        folder = self.output / "records" / f"{task['id'] // 1000:06d}"
        folder.mkdir(exist_ok=True, parents=True)
        task["record"] = (
            folder / f"game-{task['id']:09d}-attempt-{task['attempt']:03d}.jsonl"
        ).resolve()
        self.db.execute(
            "UPDATE tasks SET status='running',attempt=?,result=NULL WHERE id=?",
            (task["attempt"], task["id"]),
        )
        self.db.execute(
            "INSERT INTO attempts VALUES (?,?,?,?,?,NULL)",
            (task["id"], task["attempt"], self.run, str(task["record"]), "running"),
        )
        self.db.commit()
        return task

    def finish(self, task, result):
        outcome = result["outcome"]
        status = "terminal" if outcome["terminated"] else outcome["reason"]
        task_status = "pending" if status == "cancelled" else status
        result = {
            **result,
            "requests": task["requests"],
            "task_seconds": time.perf_counter() - task["began"],
            "started_wall": task.get("began_wall"),
            "finished_wall": time.time(),
        }
        self.db.execute(
            "UPDATE attempts SET status=?,result=? WHERE task=? AND attempt=?",
            (status, json.dumps(result), task["id"], task["attempt"]),
        )
        self.db.execute(
            "UPDATE tasks SET status=?,result=? WHERE id=?",
            (task_status, json.dumps(result), task["id"]),
        )
        self.db.commit()
        self.terminal += status == "terminal"

    def failed(self, task, error):
        self.db.execute(
            "UPDATE attempts SET status='process-exit',result=? WHERE task=? AND attempt=?",
            (
                json.dumps({"error": error, "requests": task["requests"]}),
                task["id"],
                task["attempt"],
            ),
        )
        self.db.execute("UPDATE tasks SET status='pending' WHERE id=?", (task["id"],))
        self.db.commit()

    def unfinished(self, task):
        # 仍为 running，使下次恢复能核对可能已落盘但尚未消费的完成回执。
        self.db.execute(
            "UPDATE attempts SET result=? WHERE task=? AND attempt=?",
            (
                json.dumps(
                    {
                        "requests": task["requests"],
                        "seconds": time.perf_counter() - task["began"],
                        "commands": task.get("commands"),
                        "ply": task.get("ply"),
                    }
                ),
                task["id"],
                task["attempt"],
            ),
        )
        self.db.commit()

    def summary(self):
        return {
            "run": self.run,
            "next_id": self.next_id,
            "terminal_total": self.terminal,
            "attempts": dict(
                self.db.execute(
                    "SELECT status,count(*) FROM attempts WHERE run=? GROUP BY status", (self.run,)
                )
            ),
            "task_states": dict(
                self.db.execute("SELECT status,count(*) FROM tasks GROUP BY status")
            ),
        }

    def close(self):
        self.db.close()
        self.lock.close()


def resident(
    executable,
    checkpoint,
    output,
    *,
    environments=8,
    tasks=None,
    target=64,
    seconds=3600,
    drain_seconds=120,
    seed=20261004,
    rules="mixed",
    commands=20000,
    plies=1000,
    device="cuda",
    precision="fp32",
    batch_wait_ms=0,
    resume=False,
):
    if (tasks is not None and tasks < 1) or (target is not None and target < 1):
        raise ValueError("任务/终局目标必须为正")
    if not seconds > drain_seconds >= 0 or not 1 <= commands <= 20000 or not 1 <= plies <= 1000:
        raise ValueError("总时限必须大于排空预算；命令1至20000，ply 1至1000")
    started = time.perf_counter()
    config = {
        "version": 1,
        "seed": seed,
        "rules": rules,
        "commands": commands,
        "plies": plies,
        "model": digest(checkpoint),
        "engine": digest(executable),
        "precision": precision,
    }
    ledger = Ledger(output, config, resume=resume, tasks=tasks, target=target)
    pool = None
    output = ledger.output
    stop_file = output / "STOP"
    previous_signal = signal.getsignal(signal.SIGINT)
    try:
        if resume:
            ledger.recover(executable)
        if stop_file.exists():
            raise ValueError("请移除 STOP 文件后恢复；旧记录和尝试仍保留")
        options = {
            "environments": environments,
            "tasks": tasks,
            "target": target,
            "seconds": seconds,
            "drain_seconds": drain_seconds,
            "device": device,
            "batch_wait_ms": batch_wait_ms,
            "config": config,
        }
        ledger.begin(options)

        # SIGINT 只请求停止分配；再次 SIGINT 强制退出并保留前缀。
        def stop_signal(*_):
            if stop_file.exists():
                raise KeyboardInterrupt
            stop_file.touch()

        signal.signal(signal.SIGINT, stop_signal)
        pool = Pool(executable, checkpoint, environments, device, precision, batch_wait_ms)
        if pool.model_hash != config["model"]:
            raise ValueError("加载期间检查点内容改变")
        pool.started = started
        pool.startup_seconds = time.perf_counter() - started
        with (output / f"telemetry-{ledger.run:04d}.jsonl").open("x", encoding="utf-8") as stream:

            def snapshot(row):
                row.update(ledger.summary())
                stream.write(json.dumps(row) + "\n")
                stream.flush()

            report = pool.run(
                ledger,
                seconds=seconds,
                drain_seconds=drain_seconds,
                stop_file=stop_file,
                snapshot=snapshot,
            )
            report.update(ledger.summary())
            snapshot(report)
        report["seconds"] = time.perf_counter() - started
        ledger.db.execute("UPDATE runs SET report=? WHERE id=?", (json.dumps(report), ledger.run))
        ledger.db.commit()
        (output / f"report-{ledger.run:04d}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report
    except BaseException as error:
        report = {
            "error": repr(error),
            "seconds": time.perf_counter() - started,
            **ledger.summary(),
        }
        if pool:
            report["pool"] = pool.metrics()
        failure_id = ledger.run if ledger.run is not None else f"preflight-{time.time_ns()}"
        (output / f"failure-{failure_id}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        raise
    finally:
        if pool:
            pool.close()
        signal.signal(signal.SIGINT, previous_signal)
        ledger.close()
