"""镜像统计、独立调度与真实原生评测边界。"""

from concurrent.futures import Future
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import psutil
import torch

from training.haojie.evaluation import evaluate_round, paired_games, paired_summary
from training.haojie.model import Network
from training.haojie.runtime.config import Config
from training.haojie.runtime.gpu import Local
from training.haojie.runtime.evaluation import EvaluationSchedule, read_evaluation
from training.haojie.runtime.storage import atomic_json


class EvaluationTests(unittest.TestCase):
    def test_weight_request_failure_keeps_training_control_alive(self):
        local = Local(Mock())
        executor = Mock(published=7)
        executor.weights.get.side_effect = RuntimeError("weights unavailable")
        request = local.submit("evaluation")
        local.execute_one(executor)
        with self.assertRaisesRegex(RuntimeError, "weights unavailable"):
            request.result()
        stats = local.submit("stats")
        local.execute_one(executor)
        self.assertEqual(stats.result(), executor.report.return_value)

    def test_balanced_seeds_and_complete_pairs_only(self):
        rows = list(paired_games(12345))
        self.assertEqual(rows, list(paired_games(12345)))
        self.assertEqual(len(rows), 12)
        for difficulty in ("easy", "medium", "hard"):
            level = [r for r in rows if r["difficulty"] == difficulty]
            self.assertEqual(len(level), 4)
            for pair in (0, 1):
                mirror = [r for r in level if r["pair"] == pair]
                self.assertEqual({r["learner"] for r in mirror}, {1, 2})
                self.assertEqual(len({r["seed"] for r in mirror}), 1)
        results = [{**r, "winner": r["learner"]} for r in rows]
        results[2]["winner"] = None
        results[4]["error"] = "native failed"
        results[8]["winner"] = "draw"
        score = paired_summary(results[:-1])
        self.assertEqual(
            score["easy"], dict(pairs=1, wins=2, losses=0, draws=0, unknown=1, errors=0)
        )
        self.assertEqual(score["medium"]["pairs"], 1)
        self.assertEqual(score["medium"]["errors"], 1)
        self.assertEqual(score["hard"]["pairs"], 1)
        self.assertEqual(score["hard"]["draws"], 1)

    def test_async_snapshot_no_overlap_and_failure_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            schedule = EvaluationSchedule(directory, Config(evaluation_seconds=1))
            local = Mock()
            pending = Future()
            local.submit.return_value = pending
            schedule.next_at = 0
            schedule.tick(local)
            schedule.tick(local)
            local.submit.assert_called_once_with("evaluation")
            pending.set_exception(RuntimeError("snapshot failed"))
            schedule.tick(local)
            self.assertEqual(read_evaluation(directory)["error"], "snapshot failed")
            process = Mock()
            process.poll.return_value = None
            schedule.process = process
            schedule.next_at = 0
            schedule.tick(local)
            local.submit.assert_called_once()
            schedule.close()
            process.wait.assert_not_called()

    def test_resume_respects_live_evaluator_and_detects_dead_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evaluations"
            job = root / "round-test"
            job.mkdir(parents=True)
            status = dict(
                state="starting",
                round=job.name,
                started_at=0,
                pid=os.getpid(),
                process_created=psutil.Process().create_time(),
            )
            atomic_json(root / "status.json", status)
            schedule = EvaluationSchedule(directory, Config(evaluation_seconds=1))
            local = Mock()
            schedule.tick(local)
            local.submit.assert_not_called()
            atomic_json(
                job / "process.json", {"pid": os.getpid(), "process_created": 0}
            )
            self.assertEqual(read_evaluation(directory)["state"], "failed")
            schedule.close()

    def test_native_round_twelve_games_and_small_weights_only(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory) / "evaluations" / "round-test"
            job.mkdir(parents=True)
            torch.save(Network().state_dict(), job / "model.pt")
            atomic_json(
                job / "round.json",
                dict(
                    version=17,
                    started_at=time.time(),
                    seed=12345,
                    games=4,
                    simulations=1,
                    max_commands=2,
                    device="cpu",
                ),
            )
            evaluate_round(job)
            status = read_evaluation(directory)
            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["completed"], 12)
            self.assertEqual(len((job / "results.jsonl").read_text().splitlines()), 12)
            self.assertTrue(
                all(
                    r["winner"] is None and not r.get("error")
                    for r in status["results"]
                )
            )
            self.assertTrue(
                all(
                    s["unknown"] == 4 and s["pairs"] == 0
                    for s in status["summary"].values()
                )
            )
            self.assertLess((job / "model.pt").stat().st_size, 20_000_000)
            self.assertFalse((job / "checkpoint.pt").exists())

    def test_failed_game_keeps_other_games_and_pair_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory) / "evaluations" / "round-test"
            job.mkdir(parents=True)
            atomic_json(
                job / "round.json",
                dict(
                    version=17,
                    started_at=time.time(),
                    seed=12345,
                    games=4,
                    simulations=1,
                    max_commands=2,
                    device="cpu",
                ),
            )
            outcomes = [RuntimeError("native error")] + [dict(winner=1, steps=10)] * 11
            with patch("training.haojie.evaluation.load"), patch(
                "training.haojie.evaluation.play_game", side_effect=outcomes
            ):
                evaluate_round(job)
            status = read_evaluation(directory)
            self.assertEqual(status["completed"], 12)
            self.assertEqual(
                status["summary"]["easy"],
                dict(pairs=1, wins=1, losses=1, draws=0, unknown=0, errors=1),
            )
            self.assertEqual(status["summary"]["hard"]["pairs"], 2)


if __name__ == "__main__":
    unittest.main()
