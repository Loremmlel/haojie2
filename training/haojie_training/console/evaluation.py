"""一个冻结候选、逐局公平时间片；评测不读取可变学习器，不积累跨版本胜负。"""

import copy
import hashlib
import json
import time

import torch

from ..model import ModelConfig, PolicyValueNet
from ..native.execution import PolicyInference
from ..native.pool import Pool
from .opponents import ASSETS, Teacher, schedule, summarize
from .sampling import MemoryJobs


def frozen(weights):
    result = copy.deepcopy(weights)
    result["model"] = {k: v.detach().cpu().clone() for k, v in weights["model"].items()}
    return result


class Evaluation:
    def __init__(self, controller, weights, confirmation=False, config=None):
        self.c = controller
        self.weights = frozen(weights)
        self.confirmation = confirmation
        self.config = copy.deepcopy(config or controller.config)
        self.created = time.time()
        self.index = 0
        pairs = self.config["confirmation_pairs"] if confirmation else self.config["eval_pairs"]
        # 相同固定种子族，趋势及确认前缀一致；训练序号从不参与。
        self.plan = schedule(pairs, 1)
        base = controller.store.names("baseline")
        self.baseline = torch.load(
            controller.store.root / base[-1], weights_only=True, map_location="cpu"
        )
        teachers = self.plan
        self.plan = []
        for offset in range(0, len(teachers), 3):
            group = teachers[offset : offset + 3]
            self.plan.extend(group)
            self.plan.append({**copy.deepcopy(group[0]), "difficulty": "baseline"})
        for game in self.plan:
            game["result"] = "pending"
        manifest = json.loads((ASSETS / "teacher.json").read_text(encoding="utf-8"))
        self.conditions = {
            "teacher": manifest["sha256"],
            "scheduler": "production-owner-offturn-pass-v2",
            "baseline": self.baseline["version"],
            "rules": controller.meta,
            "seconds_per_game": self.config["eval_seconds"],
            "teacher_call_budget": "game-only",
            "commands": self.config["max_commands"],
            "plies": self.config["max_plies"],
            "temperature": self.config["temperature"],
            "seeds": "fixed-eval-family-1",
        }
        self.series = hashlib.sha256(
            json.dumps(self.conditions, sort_keys=True).encode()
        ).hexdigest()

    def summary(self):
        return {
            "model": self.weights["version"],
            "updates": self.weights["updates"],
            "completed_tasks": self.index,
            "planned": len(self.plan),
            "confirmation": self.confirmation,
            "created": self.created,
            "results": summarize(self.plan),
            "expires": self.created + 86400,
            "seconds_per_game": self.config["eval_seconds"],
        }

    def slice(self):
        c = self.c
        if time.time() - self.created > 86400:
            return self.finish("候选超过24小时保留期限")
        if c.cancel.is_set() or c.eval_cancel.is_set():
            return self.finish("已取消")
        game = self.plan[self.index]
        game["result"] = "unfinished"
        began = time.monotonic()
        teacher = None
        historical = None
        try:
            model = PolicyValueNet(ModelConfig(**self.weights["config"]))
            model.load_state_dict(self.weights["model"])
            model.to(c.device).eval()
            if game["difficulty"] == "baseline":
                old = PolicyValueNet(model.config)
                old.load_state_dict(self.baseline["model"])
                historical = (old.to(c.device).eval(), self.baseline["version"])
            else:
                teacher = Teacher(game["difficulty"])
                game["budget"] = teacher.budget
            c._close_pool()
            c.pool = Pool(
                c.engine,
                None,
                1,
                c.device,
                inference=(PolicyInference(model, c.device), self.weights["version"]),
                frame_limit=32 * 2**20,
            )
            jobs = MemoryJobs(c, self.weights["version"], historical, game, teacher)
            jobs.inferences[self.weights["version"]] = PolicyInference(model, c.device)
            jobs.commands, jobs.plies = self.config["max_commands"], self.config["max_plies"]
            jobs.temperature = self.config["temperature"]

            class Cancel:
                def is_set(inner):
                    return (
                        c.cancel.is_set()
                        or c.eval_cancel.is_set()
                        or time.monotonic() - began >= self.config["eval_seconds"]
                    )

            cancel = Cancel()
            jobs.cancel = cancel.is_set
            c.pool.run(jobs, cancel_event=cancel, snapshot=c._progress)
            if jobs.result:
                outcome = jobs.result["outcome"]
                game["reason"] = outcome["reason"]
                game["commands"] = outcome["commands"]
                game["ply"] = jobs.result["ply"]
                if outcome["terminated"]:
                    game["result"] = (
                        "draw"
                        if outcome["winner"] == "draw"
                        else "win"
                        if outcome["winner"] == game["side"]
                        else "loss"
                    )
            else:
                game["reason"] = (
                    "cancelled" if c.cancel.is_set() or c.eval_cancel.is_set() else "time-budget"
                )
        except InterruptedError:
            game["reason"] = (
                "cancelled" if c.cancel.is_set() or c.eval_cancel.is_set() else "time-budget"
            )
        except Exception as error:
            game.update(result="error", error=str(error)[:1000])
        finally:
            c._close_pool()
            if teacher:
                teacher.close()
            c.runtime = {"active": [], "inflight_bytes": 0}
        game["seconds"] = time.monotonic() - began
        self.index += 1
        if self.index == len(self.plan) or c.cancel.is_set() or c.eval_cancel.is_set():
            return self.finish("完成" if self.index == len(self.plan) else "已取消")
        return None

    def finish(self, reason):
        for game in self.plan:
            if game["result"] == "pending":
                game.update(result="unfinished", reason=reason)
        return {
            "series": self.series,
            "conditions": self.conditions,
            "model": self.weights["version"],
            "updates": self.weights["updates"],
            "games_trained": self.c.counts["games"],
            "time": time.time(),
            "results": summarize(self.plan),
            "games": self.plan,
            "fixture": False,
            "confirmation": self.confirmation,
            "reason": reason,
        }


def evaluate_slice(c):
    c.state = "evaluating"
    if not c.evaluation:
        confirming = bool(c.candidate and c.candidate.get("confirm"))
        weights = c.candidate["weights"] if confirming else c._weights()
        config = c.candidate["config"] if confirming else None
        c.evaluation = Evaluation(c, weights, confirming, config)
        c.round_number += 1
        c.eval_cancel.clear()
        c.log(f"冻结评测 {weights['version'][:12]}；每次一局时间片，档次轮转")
    c.want_eval = False
    job = c.evaluation
    report = job.slice()
    if report is None:
        return
    # 先释放已完成任务；保存失败后暂停不能再次执行越过末尾的评测局。
    c.evaluation = None
    c.evaluations.append(report)
    for old in c.evaluations[:-2]:
        old.pop("games", None)
    c.evaluations = c._compact(c.evaluations)
    c.eval_games, c.eval_step = c.counts["games"], c.trainer.updates
    results = [report["results"][key] for key in ("easy", "medium", "hard")]
    score = sum(r["score_rate"] or 0 for r in results) / 3
    if job.confirmation:
        c.candidate = None
        if all(r["n"] >= 20 and r["completion"] == 1 for r in results):
            names = c.store.names("best")
            old = (
                torch.load(c.store.root / names[-1], weights_only=True, map_location="cpu")
                if names
                else None
            )
            if not old or old["evaluation"]["series"] != report["series"] or score > old["score"]:
                c.store.save_tensor(
                    "best",
                    job.weights["updates"],
                    {**job.weights, "score": score, "evaluation": report},
                    1,
                )
                c.log("确认批次满足每档20局；已保存该系列最佳（不代表统计显著提升）")
    elif all(r["n"] for r in results):
        c.candidate = {
            "weights": job.weights,
            "model": job.weights["version"],
            "score": score,
            "created": time.time(),
            "confirm": False,
            "series": report["series"],
            "config": job.config,
        }
    c._metrics()
    c.log("评测批次结束；未完成及错误不计胜负，可在评测页确认冻结候选")
