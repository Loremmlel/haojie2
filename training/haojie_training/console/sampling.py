"""常驻 Pool 的内存任务出口；双方版本按真实行动者路由，终局后才接纳当前方样本。"""

import hashlib
import time

import torch

from ..native.execution import PolicyInference
from .memory import Trajectory

# 持续局的协议路由与权重摘要分开；每条训练样本另记实际行为版本。
CURRENT_ROUTE = hashlib.sha256(b"haojie-continuous-current-v1").hexdigest()


def task_plan(config, task_id):
    """默认八项平衡组合；自定义比例使用独立散列流，与槽位、评测和完成顺序无关。"""
    index = task_id - 1

    def seed(stream):
        value = hashlib.sha256(f"{config['seed']}:{stream}:{task_id}".encode()).digest()
        return 1 + int.from_bytes(value[:4], "little") % 1_999_999_999

    def choose(key, bit):
        percent = config.get(key, 50)
        return (index // bit % 2 == 0) if percent == 50 else seed(key) % 100 < percent

    classic = choose("classic_percent", 1)
    history = choose("history_percent", 2)
    current_first = choose("current_first_percent", 4)

    return {
        "start": {"rules": "classic" if classic else "shrine", "seed": seed("game")},
        "history": history,
        "current_side": 1 if current_first else 2,
        "sampler_seed": seed("policy"),
    }


class MemoryJobs:
    memory = True

    def __init__(
        self,
        controller,
        current,
        historical=None,
        evaluation=None,
        teacher=None,
        *,
        continuous=False,
    ):
        self.c = controller
        self.continuous = continuous and evaluation is None
        self.mc_context = controller.config["method"] == "decomposed-mc-q-v2"
        self.temperature = controller.config["temperature"]
        self.current = CURRENT_ROUTE if self.continuous else current
        self.version, self.historical = current, historical
        self.evaluation, self.teacher = evaluation, teacher
        self.commands, self.plies = (
            controller.config["max_commands"],
            controller.config["max_plies"],
        )
        self.target = 1 if evaluation else controller.config["games_per_round"]
        self.assigned = self.finished = 0
        self.result = None
        self.resource_failure = None
        self.cancel = None
        # 推理执行器跨训练轮复用；历史对手也计入同一工作池，监控不再读取空统计。
        inference = controller.pool.inference
        inference.model.eval()
        self.inferences = {self.current: inference}
        if historical:
            model, identity = historical
            self.inferences[identity] = PolicyInference(
                model.eval(), controller.device, inference.precision, stats=inference.stats
            )

    def next_window(self, current, historical):
        """所有在途局停在新决策根后发布权重；历史对手仍按整局冻结。"""
        self.version, self.historical = current, historical
        self.target = self.finished + self.c.config["games_per_round"]
        inference = self.inferences[self.current]
        inference.model.eval()
        if historical and historical[1] not in self.inferences:
            model, version = historical
            self.inferences[version] = PolicyInference(
                model.eval(), self.c.device, inference.precision, stats=inference.stats
            )

    def release_unused(self):
        """只释放没有在途对局引用的冻结历史版本。"""
        used = {model for task in self.c.pool.slots.values() for model in task["models"]}
        for version in list(self.inferences):
            if version != self.current and version not in used:
                self.inferences.pop(version).close()

    def claim(self):
        if (self.finished if self.continuous else self.assigned) >= self.target:
            return None
        self.assigned += 1
        counter = "eval_tasks" if self.evaluation else "tasks"
        self.c.counts[counter] = self.c.counts.get(counter, 0) + 1
        task_id = self.c.counts[counter]
        plan = task_plan(self.c.config, task_id)
        models = [self.current, self.current]
        if self.historical and plan["history"] and self.historical[1] != self.version:
            models[2 - plan["current_side"]] = self.historical[1]
        start = plan["start"]
        if self.c.starts:
            start = self.c.starts[(task_id - 1) % len(self.c.starts)]
        if self.evaluation:
            start = self.evaluation["start"]
            if self.historical:
                models[2 - self.evaluation["side"]] = self.historical[1]
        matchup = "historical" if models[0] != models[1] else "self"
        context = {
            "task": task_id,
            "rules": start["rules"],
            "models": tuple(models),
            "matchup": matchup,
        }
        if self.continuous:
            context.pop("models")
            context.update(
                policy_mode="continuous-root-v1",
                initial_models=tuple(self.version if m == self.current else m for m in models),
                started_generation=self.c.generation,
            )
        task = {
            "id": task_id,
            "start": start,
            "models": models,
            "sampler_seed": self.evaluation.get("sampler_seed", start["seed"])
            if self.evaluation
            else plan["sampler_seed"],
            "trajectory": Trajectory(task_id, context=context),
            "last_index": -1,
            "last_decision": 0,
            "context": context,
        }
        if self.evaluation and self.teacher:
            task["teacher"] = 3 - self.evaluation["side"]
        if not self.evaluation:
            coverage = self.c.counts.setdefault("matchups", {})
            key = f"{start['rules']}/{matchup}/current-{plan['current_side']}"
            coverage[key] = coverage.get(key, 0) + 1
            if plan["history"] and matchup == "self":
                self.c.counts["history_fallback"] = self.c.counts.get("history_fallback", 0) + 1
        return task

    def stop(self):
        return False

    def example(self, task, message):
        if self.evaluation:
            return
        actor = message["actor"]
        if actor not in (1, 2) or message["model"] != task["models"][actor - 1]:
            raise ValueError("样本实际行动者与行为版本不一致")
        if message["index"] < task["last_index"]:
            raise ValueError("样本命令序号倒退")
        task["last_index"] = message["index"]
        if self.mc_context:
            if (
                message.get("semantics") != "mc-context-v2"
                or message["decision"] <= task["last_decision"]
            ):
                raise ValueError("决策身份重复或样本语义不匹配")
            task["last_decision"] = message["decision"]
        if message["model"] == self.current:
            behavior = (
                {"behavior_version": self.version, "behavior_generation": self.c.generation}
                if self.continuous
                else None
            )
            task["trajectory"].add(message, behavior=behavior)

    def infer(self, messages, *, completed=None):
        """同步或异步结果共享 MC 变换；已完成张量仅用于同一模型的一组请求。"""
        identities = {m["model"] for m in messages}
        if completed is not None and (len(identities) != 1 or completed.shape[0] != len(messages)):
            raise ValueError("已完成推理必须对应同一模型和完整请求组")
        width = max(m["candidates"] for m in messages)
        if len(identities) == 1:
            output = (
                completed
                if completed is not None
                else self.inferences[messages[0]["model"]]([m["input"] for m in messages])
            )
            output = (
                output.tanh() / self.temperature
                if self.c.config["method"].startswith("decomposed-mc-q-")
                else output
            )
            return output.float()
        result = torch.zeros((len(messages), width))
        for identity in identities:
            indices = [i for i, m in enumerate(messages) if m["model"] == identity]
            output = (
                completed
                if completed is not None
                else self.inferences[identity]([messages[i]["input"] for i in indices])
            )
            if self.c.config["method"].startswith("decomposed-mc-q-"):
                # 同一候选评分头训练 tanh(Q)，并直接用 Q/温度控制后续 Gumbel 排序。
                output = output.tanh() / self.temperature
            for row, index in enumerate(indices):
                result[index, : messages[index]["candidates"]] = output[
                    row, : messages[index]["candidates"]
                ]
        return result

    def teacher_next(self, message):
        if self.c.cancel.is_set() or self.c.eval_cancel.is_set():
            raise InterruptedError("评测已取消")
        return self.teacher.next(
            message,
            lambda: (
                self.c.cancel.is_set()
                or self.c.eval_cancel.is_set()
                or bool(self.cancel and self.cancel())
            ),
        )

    def finish(self, task, message):
        self.finished += 1
        outcome = message["outcome"]
        self.result = message
        if self.evaluation:
            return
        self._coverage(task)
        trace = task["trajectory"]
        if outcome["terminated"]:
            self.c.counts["games"] += 1
            self.c.counts["retained_samples"] += len(trace.rows)
            completed = self.c.counts.setdefault("completed_matchups", {})
            key = f"{task['context']['rules']}/{task['context']['matchup']}"
            completed[key] = completed.get(key, 0) + 1
            for row in trace.terminal(outcome["returns"]):
                self.c.samples.add(row)
        else:
            self.c.counts["truncated"] += 1
        self.c.last_activity = time.time()

    def _coverage(self, task):
        trace = task["trajectory"]
        self.c.counts["seen_samples"] += trace.seen
        self.c.counts["oversize_samples"] += trace.oversize
        self.c.counts["eligible_samples"] = (
            self.c.counts.get("eligible_samples", 0) + trace.eligible
        )
        coverage = self.c.counts.setdefault("stages", {})
        for stage, count in trace.coverage.items():
            coverage[stage] = coverage.get(stage, 0) + count

    def unfinished(self, task):
        if not self.evaluation:
            self._coverage(task)
            self.c.counts["discarded"] += 1

    def resource_error(self, reason):
        self.resource_failure = reason
        self.c.counts["resource_cancelled"] = self.c.counts.get("resource_cancelled", 0) + 1
        self.c.log(reason + "；已取消该局，未裁剪实体或补终局标签")
