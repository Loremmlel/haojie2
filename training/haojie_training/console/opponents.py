"""独立嵌入式 V8 执行真实 TS 教师静态包；不启动 Node、浏览器或后台网络。"""

import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path

ASSETS = Path(__file__).with_name("assets")


def ready():
    return bool(
        importlib.util.find_spec("py_mini_racer")
        and (ASSETS / "teacher.js").exists()
        and (ASSETS / "teacher.json").exists()
    )


class Teacher:
    def __init__(self, difficulty, budget=None):
        if not ready():
            raise RuntimeError("评测未就绪：安装 pip install mini-racer==0.14.1，并提供随包教师JS")
        from py_mini_racer import MiniRacer

        self.manifest = json.loads((ASSETS / "teacher.json").read_text(encoding="utf-8"))
        source = (ASSETS / "teacher.js").read_bytes()
        if hashlib.sha256(source).hexdigest() != self.manifest["sha256"]:
            raise ValueError("教师静态包指纹不匹配")
        self.runner = asyncio.Runner()

        async def create():
            return MiniRacer()

        self.context = self.runner.run(create())
        try:
            self.context.set_hard_memory_limit(256 * 1024 * 1024)
            # work 模式不依赖时间分配预算；仅满足既有统计计时接口。
            self._eval("globalThis.performance = {now: () => 0}")
            self._eval(source.decode())
            self.budget = json.loads(
                self._eval(
                    f"JSON.stringify(HaojieTeacher.reset({json.dumps(difficulty)}, "
                    f"{json.dumps(budget) if budget else 'undefined'}))",
                )
            )
        except BaseException:
            self.close()
            raise

    def _eval(self, expression, cancel=None):
        async def run():
            task = asyncio.create_task(self.context.eval_cancelable(expression))
            try:
                while not task.done():
                    if cancel and cancel():
                        raise InterruptedError("教师计算已取消")
                    await asyncio.wait({task}, timeout=0.05)
                return task.result()
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

        return self.runner.run(run())

    def next(self, message, cancel=None):
        expression = (
            "JSON.stringify(HaojieTeacher.next("
            + json.dumps(message["observation"])
            + ","
            + json.dumps(message.get("actor"))
            + ","
            + json.dumps(message.get("optional", False))
            + "))"
        )
        return json.loads(self._eval(expression, cancel))

    def close(self):
        self.context.close()
        self.runner.close()


def schedule(pairs, round_number):
    # 高位独立种子空间；每组经典/神龛各一对，交换先后手。
    return [
        {
            "difficulty": difficulty,
            "side": side,
            "start": {"rules": rules, "seed": 3_000_000_000 + round_number * 1000 + pair * 2 + ri},
        }
        for pair in range(pairs)
        for ri, rules in enumerate(("classic", "shrine"))
        for side in (1, 2)
        for difficulty in ("easy", "medium", "hard")
    ]


def summarize(games):
    result = {}
    for difficulty in ("easy", "medium", "hard", "baseline"):
        rows = [g for g in games if g["difficulty"] == difficulty]
        wins = sum(g.get("result") == "win" for g in rows)
        losses = sum(g.get("result") == "loss" for g in rows)
        draws = sum(g.get("result") == "draw" for g in rows)
        completed = wins + losses + draws
        result[difficulty] = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "unfinished": sum(g.get("result") in ("unfinished", "pending") for g in rows),
            "errors": sum(g.get("result") == "error" for g in rows),
            "timeouts": sum(
                g.get("reason") in ("time-budget", "teacher-time-budget") for g in rows
            ),
            "truncated": sum(
                g.get("reason") in ("commands", "plies", "decode-budget") for g in rows
            ),
            "total": len(rows),
            "n": completed,
            "completion": completed / len(rows) if rows else 0,
            "win_rate": wins / completed if completed else None,
            "score_rate": (wins + draws / 2) / completed if completed else None,
        }
    return result
