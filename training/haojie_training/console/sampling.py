"""常驻 Pool 的内存任务出口；双方版本按真实行动者路由，终局后才接纳当前方样本。"""

import time

import torch

from ..native.execution import PolicyInference
from .memory import Trajectory


class MemoryJobs:
    memory = True

    def __init__(self, controller, current, historical=None, evaluation=None, teacher=None):
        self.c = controller
        self.current, self.historical = current, historical
        self.evaluation, self.teacher = evaluation, teacher
        self.commands, self.plies = (
            controller.config["max_commands"],
            controller.config["max_plies"],
        )
        self.target = 1 if evaluation else controller.config["games_per_round"]
        self.assigned = self.finished = 0
        self.result = None
        self.inferences = {
            current: PolicyInference(controller.trainer.model.eval(), controller.device)
        }
        if historical:
            model, identity = historical
            self.inferences[identity] = PolicyInference(model.eval(), controller.device)

    def claim(self):
        if self.assigned >= self.target:
            return None
        self.assigned += 1
        self.c.counts["tasks"] += 1
        task_id = self.c.counts["tasks"]
        models = [self.current, self.current]
        if self.historical and task_id % 2:
            models[task_id % 4 // 2] = self.historical[1]
        start = {
            "rules": "classic" if task_id % 2 else "shrine",
            "seed": 1 + (self.c.config["seed"] + task_id) % 1_000_000_000,
        }
        if self.c.starts:
            start = self.c.starts[(task_id - 1) % len(self.c.starts)]
        if self.evaluation:
            start = self.evaluation["start"]
        task = {
            "id": task_id,
            "start": start,
            "models": models,
            "sampler_seed": (self.c.config["seed"] + task_id * 17) & 0xFFFFFFFF,
            "trajectory": Trajectory(task_id),
            "last_index": -1,
        }
        if self.evaluation:
            task["teacher"] = 3 - self.evaluation["side"]
        return task

    def stop(self):
        return False

    def example(self, task, message):
        if self.evaluation:
            raise ValueError("评测不得输出训练样本")
        actor = message["actor"]
        if actor not in (1, 2) or message["model"] != task["models"][actor - 1]:
            raise ValueError("样本实际行动者与行为版本不一致")
        if message["index"] < task["last_index"]:
            raise ValueError("样本命令序号倒退")
        task["last_index"] = message["index"]
        if message["model"] == self.current:
            task["trajectory"].add(message)

    def infer(self, messages):
        width = max(m["candidates"] for m in messages)
        result = torch.zeros((len(messages), width))
        for identity in {m["model"] for m in messages}:
            indices = [i for i, m in enumerate(messages) if m["model"] == identity]
            output = self.inferences[identity]([messages[i]["input"] for i in indices])
            if self.c.config["method"] == "decomposed-mc-q-v1":
                # 同一候选评分头训练 tanh(Q)，并直接用 Q/温度控制后续 Gumbel 排序。
                output = output.tanh() / self.c.config["temperature"]
            for row, index in enumerate(indices):
                result[index, : messages[index]["candidates"]] = output[
                    row, : messages[index]["candidates"]
                ]
        return result

    def teacher_next(self, message):
        if self.c.cancel.is_set() or self.c.eval_cancel.is_set():
            raise InterruptedError("评测已取消")
        return self.teacher.next(message)

    def finish(self, task, message):
        self.finished += 1
        outcome = message["outcome"]
        self.result = message
        if self.evaluation:
            return
        trace = task["trajectory"]
        self.c.counts["seen_samples"] += trace.seen
        self.c.counts["oversize_samples"] += trace.oversize
        if outcome["terminated"]:
            self.c.counts["games"] += 1
            self.c.counts["retained_samples"] += len(trace.rows)
            for row in trace.terminal(outcome["returns"]):
                self.c.samples.add(row)
        else:
            self.c.counts["truncated"] += 1
        self.c.last_activity = time.time()

    def unfinished(self, task):
        if not self.evaluation:
            self.c.counts["discarded"] += 1
