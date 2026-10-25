"""真实游戏 AI 的嵌入式 V8；只接收公开观察，取消由执行线程检查。"""

import asyncio
import hashlib
import json
from pathlib import Path

ASSETS = Path(__file__).with_name("assets")


class Teacher:
    def __init__(self, difficulty):
        from py_mini_racer import MiniRacer

        self.manifest = json.loads(
            (ASSETS / "teacher.json").read_text(encoding="utf-8")
        )
        source = (ASSETS / "teacher.js").read_bytes()
        if hashlib.sha256(source).hexdigest() != self.manifest["sha256"]:
            raise ValueError("测评 AI 静态包指纹不匹配")
        self.runner = asyncio.Runner()

        async def create():
            return MiniRacer()

        self.context = self.runner.run(create())
        try:
            self.context.set_hard_memory_limit(256 * 1024 * 1024)
            self.evaluate("globalThis.performance = {now: () => 0}")
            self.evaluate(source.decode())
            self.budget = json.loads(
                self.evaluate(
                    f"JSON.stringify(HaojieTeacher.reset({json.dumps(difficulty)}))"
                )
            )
        except BaseException:
            self.close()
            raise

    def evaluate(self, expression, cancelled=lambda: False):
        async def run():
            task = asyncio.create_task(self.context.eval_cancelable(expression))
            try:
                while not task.done():
                    if cancelled():
                        raise InterruptedError("测评已取消或达到单局时限")
                    await asyncio.wait({task}, timeout=0.05)
                return task.result()
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

        return self.runner.run(run())

    def next(self, state, cancelled):
        arguments = ",".join(
            json.dumps(state[key]) for key in ("observation", "actor", "optional")
        )
        return json.loads(
            self.evaluate(f"JSON.stringify(HaojieTeacher.next({arguments}))", cancelled)
        )

    def close(self):
        self.context.close()
        self.runner.close()
