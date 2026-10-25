"""冻结原生检查点，对战现有游戏 AI；结果只采用正式终局。"""

import argparse
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

from .teacher import Teacher


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def schedule(pairs):
    return [
        {
            "difficulty": level,
            "side": side,
            "rules": rules,
            "seed": 3_000_001_000 + pair * 2 + ri,
            "result": "pending",
        }
        for pair in range(pairs)
        for ri, rules in enumerate(("classic", "shrine"))
        for side in (1, 2)
        for level in ("easy", "medium", "hard")
    ]


def summarize(games):
    result = {}
    for level in ("easy", "medium", "hard"):
        rows = [g for g in games if g["difficulty"] == level]
        counts = {
            key: sum(g["result"] == key for g in rows)
            for key in ("win", "loss", "draw", "unfinished", "error", "pending")
        }
        completed = counts["win"] + counts["loss"] + counts["draw"]
        result[level] = {
            **counts,
            "completed": completed,
            "total": len(rows),
            "win_rate": counts["win"] / completed if completed else None,
        }
    return result


class Native:
    def __init__(self, command, checkpoint, device, log):
        self.process = subprocess.Popen(
            [*command, "evaluate", "--checkpoint", str(checkpoint), "--device", device],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.lines = queue.Queue()
        self.inflight = True
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()

    def read(self):
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def receive(self, cancelled):
        while not cancelled():
            try:
                line = self.lines.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError("原生测评进程退出，详见测评日志")
            self.inflight = False
            return json.loads(line)
        raise InterruptedError("测评已取消或达到时限")

    def request(self, text, cancelled):
        self.inflight = True
        self.process.stdin.write(text + "\n")
        self.process.stdin.flush()
        return self.receive(cancelled)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait()
        self.reader.join()
        self.process.stdin.close()
        self.process.stdout.close()


def run(command, checkpoint, device, directory, pairs=1, seconds=120):
    directory.mkdir(parents=True, exist_ok=True)
    plan = schedule(pairs)
    report = {
        "phase": "initializing",
        "checkpoint": str(checkpoint),
        "games": plan,
        "created": time.time(),
        "seconds_per_game": seconds,
        "max_commands": 20000,
        "max_plies": 1000,
    }

    def cancelled():
        return (directory / "cancel").exists()

    native = None

    def publish():
        report["summary"] = summarize(plan)
        write_json(directory / "status.json", report)

    publish()
    try:
        with (directory / "native.log").open("w", encoding="utf-8") as log:
            native = Native(command, checkpoint, device, log)
            deadline = time.monotonic() + 300
            report.update(
                native.receive(lambda: cancelled() or time.monotonic() >= deadline)
            )
            report["phase"] = "running"
            for index, game in enumerate(plan):
                if cancelled():
                    break
                report["current"] = index + 1
                game["result"] = "running"
                publish()
                teacher = None
                start = time.monotonic()

                def expired():
                    return cancelled() or time.monotonic() - start >= seconds

                try:
                    teacher = Teacher(game["difficulty"])
                    report["teacher_sha256"] = teacher.manifest["sha256"]
                    game["budget"] = teacher.budget
                    state = native.request(
                        f"start {game['seed']} {game['side']} {int(game['rules'] == 'shrine')}",
                        expired,
                    )
                    while state["winner"] is None:
                        game.update(
                            commands=state["commands"], ply=state["observation"]["ply"]
                        )
                        if game["commands"] >= 20000 or game["ply"] >= 1000:
                            game.update(
                                result="unfinished", reason="command-or-ply-limit"
                            )
                            break
                        if state["actor"] == game["side"]:
                            state = native.request("advance", expired)
                        else:
                            action = teacher.next(state, expired)
                            state = native.request(
                                "command " + json.dumps(action), expired
                            )
                        if time.monotonic() - report.get("progress_time", 0) >= 2:
                            report["progress_time"] = time.monotonic()
                            publish()
                    else:
                        winner = state["winner"]
                        game.update(
                            result="draw"
                            if winner == "draw"
                            else "win"
                            if winner == game["side"]
                            else "loss",
                            commands=state["commands"],
                        )
                except InterruptedError:
                    game.update(
                        result="unfinished",
                        reason="cancelled" if cancelled() else "time-budget",
                    )
                    if cancelled():
                        break
                    # 等待当前有界时间片退出，再以同一冻结模型测下一局。
                    if native.inflight:
                        drain_deadline = time.monotonic() + 30
                        native.receive(
                            lambda: cancelled() or time.monotonic() >= drain_deadline
                        )
                except Exception as error:
                    game.update(result="error", reason=str(error))
                    report["error"] = str(error)
                    break
                finally:
                    game["seconds"] = time.monotonic() - start
                    if teacher:
                        teacher.close()
                    publish()
            report["phase"] = (
                "cancelled"
                if cancelled()
                else "failed"
                if "error" in report
                else "finished"
            )
    except Exception as error:
        report.update(phase="failed", error=str(error))
    finally:
        if native:
            native.close()
        for game in plan:
            if game["result"] in ("pending", "running"):
                game.update(result="unfinished", reason="round-stopped")
        report["finished"] = time.time()
        publish()
        write_json(directory / "result.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("native", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 1 <= args.pairs <= 20 or not 1 <= args.seconds <= 3600:
        parser.error("pairs 1–20; seconds 1–3600")
    command = args.native[1:] if args.native[:1] == ["--"] else args.native
    run(command, args.checkpoint, args.device, args.directory, args.pairs, args.seconds)
