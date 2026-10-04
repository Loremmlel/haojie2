import random
import tempfile
import threading
import time
import unittest
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from haojie_training.console.controller import Controller, defaults
from haojie_training.console.evaluation import Evaluation, evaluate_slice, frozen
from haojie_training.console.memory import SamplePool, Trajectory
from haojie_training.console.opponents import Teacher, schedule
from haojie_training.console.sampling import task_plan
from haojie_training.console.storage import QuotaError, Store
from haojie_training.data import synthetic_batch
from haojie_training.model import ModelConfig, PolicyValueNet


class DeliveryTests(unittest.TestCase):
    def test_resource_cancelled_evaluation_is_an_error_not_a_timeout_or_loss(self):
        model = PolicyValueNet(ModelConfig.tiny())
        job = Evaluation.__new__(Evaluation)
        job.c = SimpleNamespace(
            config=defaults(),
            trainer=SimpleNamespace(model=model),
            device=torch.device("cpu"),
            engine="unused",
            counts={"games": 0},
            cancel=threading.Event(),
            eval_cancel=threading.Event(),
            _close_pool=lambda: None,
            _progress=lambda _: None,
            log=lambda _: None,
        )
        job.weights = job.baseline = {
            "version": "a" * 64,
            "updates": 0,
            "config": asdict(model.config),
            "model": model.state_dict(),
        }
        job.config, job.created = defaults(), time.time()
        job.plan = [{**schedule(1, 1)[0], "difficulty": "baseline", "result": "pending"}]
        job.index, job.confirmation, job.series, job.conditions = 0, False, "resource-test", {}
        with patch("haojie_training.console.evaluation.Pool") as pool:
            pool.return_value.run.side_effect = lambda jobs, **_: jobs.resource_error("帧内存不足")
            report = job.slice()
        result = report["results"]["baseline"]
        self.assertEqual(result["errors"], 1)
        self.assertEqual((result["timeouts"], result["losses"], result["n"]), (0, 0, 0))
        self.assertEqual(report["games"][0]["reason"], "resource-budget")

    def test_due_evaluation_survives_pause_but_explicit_pending_cancel_is_respected(self):
        c = Controller.__new__(Controller)
        c.config = {**defaults(), "eval_games": 4}
        c.counts, c.trainer = {"games": 8}, SimpleNamespace(updates=256)
        c.eval_games, c.eval_step = 4, 128
        c.cancel, c.eval_cancel, c.wake = (threading.Event() for _ in range(3))
        c.lock = threading.RLock()
        c.want_run, c.want_eval = False, True
        c.evaluation = c.candidate = None
        self.assertFalse(c._due_evaluation())
        c.want_run = True
        self.assertTrue(c._due_evaluation())
        c.command("cancel-evaluation")
        self.assertFalse(c.want_eval)
        self.assertFalse(c._due_evaluation())

    def test_rare_pass_survives_retention_and_is_consumed_with_its_actor_return(self):
        torch.set_num_threads(1)
        inputs = synthetic_batch(ModelConfig.tiny(), size=1, entities=4, actions=2)
        frame = {
            "input": {
                k: v[0] for k, v in inputs.items() if k not in ("policy", "value", "value_mask")
            },
            "bytes": 2000,
            "candidates": 2,
            "selected": 0,
            "actor": 1,
            "step": 0,
        }
        trajectory = Trajectory(19, capacity=16)
        for _ in range(10000):
            trajectory.add(frame)
        trajectory.add({**frame, "selected": 1, "actor": 2, "pass": True})
        for _ in range(10000):
            trajectory.add(frame)
        self.assertEqual(len(trajectory.rows), 16)
        pool = SamplePool(1000000)
        for row in trajectory.terminal({"1": 1, "2": -1}):
            if row["_meta"]["pass"]:
                self.assertEqual(row["policy"].argmax().item(), 1)
                self.assertEqual(row["value"].item(), -1)
            pool.add(row)
        while pool.rows:
            pool.batch(8, random.Random(7))
        self.assertEqual(pool.coverage["pass"], 2)

    def test_failed_new_experiment_preserves_previous_state(self):
        c = Controller.__new__(Controller)
        c.new_request = ("fresh", torch.device("cpu"), defaults())
        c.trainer, c.experiment, c.version = None, "original", "weights"
        c._close_pool = lambda: None

        def fail(*_):
            c.experiment, c.version = "failed", "invalid"
            raise QuotaError("空间不足")

        c._activate_experiment = fail
        with self.assertRaises(QuotaError):
            c._new_experiment()
        self.assertEqual((c.experiment, c.version), ("original", "weights"))

    def test_teacher_can_cancel_inside_a_long_javascript_call(self):
        teacher = Teacher("hard")
        cancel = threading.Event()
        timer = threading.Timer(0.1, cancel.set)
        timer.start()
        began = time.monotonic()
        try:
            with self.assertRaises(InterruptedError):
                teacher._eval("while (true) {}", cancel.is_set)
            self.assertLess(time.monotonic() - began, 1.5)
        finally:
            timer.join()
            teacher.close()

    def test_combinations_and_streams_do_not_depend_on_slots_or_evaluation(self):
        config = defaults()
        plans = [task_plan(config, i) for i in range(1, 801)]
        counts = Counter((p["start"]["rules"], p["history"], p["current_side"]) for p in plans)
        self.assertEqual(len(counts), 8)
        self.assertEqual(set(counts.values()), {100})
        for concurrency in (1, 3, 16):
            scheduled = {}
            for slot in range(concurrency):
                for task in range(slot + 1, 801, concurrency):
                    schedule(1, task)
                    scheduled[task] = task_plan(config, task)
            self.assertEqual(plans, [scheduled[i] for i in range(1, 801)])
        self.assertEqual(len({p["start"]["seed"] for p in plans}), 800)
        self.assertTrue(all(p["start"]["seed"] != p["sampler_seed"] for p in plans))

    def test_automatic_consumption_age_and_pass_accounting(self):
        torch.set_num_threads(1)
        data = synthetic_batch(ModelConfig.tiny(), size=1, entities=4, actions=2)
        pool = SamplePool(100000, reuse=2, max_age=2)
        for _ in range(10):
            pool.add(
                {
                    **{k: v[0] for k, v in data.items()},
                    "_meta": {"stage": "pass", "game": {"rules": "shrine"}},
                }
            )
        self.assertEqual(pool.budget(4, 100), 5)
        rng = random.Random(4)
        while pool.rows:
            pool.batch(4, rng)
        self.assertEqual(pool.consumed, 20)
        self.assertEqual(pool.coverage["pass"], 20)
        self.assertEqual(pool.exhausted, 10)
        pool.add({k: v[0] for k, v in data.items()})
        pool.expire(3)
        self.assertEqual(pool.age_evicted, 1)
        self.assertEqual(pool.used, 0)
        self.assertEqual(pool.budget(4, 100), 0)

    def test_experiments_share_quota_and_atomic_temporary_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder, limit=100)
            store.write("experiments/a/recovery/1.pt", b"a" * 50)
            store.write("experiments/b/recovery/1.pt", b"b" * 30)
            with self.assertRaises(QuotaError):
                store.write("experiments/c/baseline/0.pt", b"c" * 21)
            orphan = Path(folder) / "experiments/b/recovery/2.pt.partial"
            orphan.write_bytes(b"x" * 10)
            store.reconcile()
            store.write("experiments/b/recovery/2.pt", b"c" * 20)
            self.assertFalse(orphan.exists())
            self.assertEqual(store.snapshot()["used"], 100)

    def test_freezing_clones_storage_and_default_confirmation_is_reachable(self):
        weights = {"model": {"x": torch.tensor([1.0])}, "version": "v", "config": {}, "updates": 7}
        held = frozen(weights)
        weights["model"]["x"].add_(9)
        self.assertEqual(float(held["model"]["x"]), 1.0)
        plan = schedule(defaults()["confirmation_pairs"], 1)
        self.assertEqual(
            Counter(g["difficulty"] for g in plan), {"easy": 20, "medium": 20, "hard": 20}
        )
        self.assertEqual([g["difficulty"] for g in plan[:3]], ["easy", "medium", "hard"])
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            report = {
                "series": "series",
                "results": {
                    d: {"n": 20, "completion": 1, "score_rate": 0.5}
                    for d in ("easy", "medium", "hard")
                },
            }
            job = SimpleNamespace(weights=held, confirmation=True, slice=lambda: report)
            c = SimpleNamespace(
                state="paused",
                evaluation=job,
                want_eval=False,
                evaluations=[],
                counts={"games": 10},
                trainer=SimpleNamespace(updates=999),
                _compact=lambda x: x,
                _metrics=lambda: None,
                store=store,
                log=lambda x: None,
                losses=[],
                logs=[],
                candidate={},
            )
            evaluate_slice(c)
            saved = torch.load(store.root / store.names("best")[-1], weights_only=True)
            self.assertEqual(saved["updates"], 7)
            self.assertEqual(float(saved["model"]["x"]), 1)
            self.assertEqual(c.trainer.updates, 999)
            self.assertIsNone(c.evaluation)
            # 最佳写入的原子副本空间不足时，学习器仍在，完成任务不能重复推进。
            c.evaluation = job
            store.remove(store.names("best")[-1])
            store.limit = 1
            with self.assertRaises(QuotaError):
                evaluate_slice(c)
            self.assertIsNone(c.evaluation)
            self.assertEqual(c.trainer.updates, 999)

    def test_candidate_expiry_and_cancel_never_turn_into_losses(self):
        job = Evaluation.__new__(Evaluation)
        job.c = SimpleNamespace(counts={"games": 0})
        job.plan = schedule(1, 1)
        for game in job.plan:
            game["result"] = "pending"
        job.series, job.conditions = "s", {}
        job.weights = {"version": "v", "updates": 0}
        job.confirmation = False
        result = job.finish("已取消")
        self.assertTrue(all(r["losses"] == 0 for r in result["results"].values()))
        self.assertTrue(all(r["n"] == 0 for r in result["results"].values()))


if __name__ == "__main__":
    unittest.main()
