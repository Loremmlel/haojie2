"""真实长短对局跨采样窗口、更新和评测继续；固定权重下逐样本核对有限全集。"""

import gzip
import hashlib
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from haojie_training.console.controller import Controller
from haojie_training.console.evaluation import Evaluation
from haojie_training.console.sampling import MemoryJobs
from haojie_training.console.storage import Store


class ManualController(Controller):
    def _loop(self):
        pass


@contextmanager
def limited_tasks(limit):
    original = MemoryJobs.claim

    def claim(jobs):
        if jobs.evaluation or jobs.c.counts["tasks"] < limit:
            return original(jobs)
        return None

    with patch.object(MemoryJobs, "claim", claim):
        yield


class HistoryLeaseTests(unittest.TestCase):
    def test_live_history_cap_and_current_route_ownership(self):
        current = Mock(precision="fp32", stats=object())
        frozen = Mock()
        frozen.eval.return_value = frozen
        c = SimpleNamespace(
            config={
                "method": "decomposed-mc-q-v2",
                "temperature": 0.25,
                "max_commands": 20000,
                "max_plies": 1000,
                "games_per_round": 2,
                "history_size": 4,
                "history_percent": 50,
            },
            pool=SimpleNamespace(inference=current, slots={}),
            version="a" * 64,
            generation=0,
            device="cpu",
            trainer=SimpleNamespace(model=Mock(config=object())),
            store=SimpleNamespace(names=Mock(return_value=["history/new.pt"]), root=Path("unused")),
        )
        with patch(
            "haojie_training.console.sampling.PolicyInference",
            side_effect=lambda model, *args, **kwargs: Mock(model=model),
        ):
            jobs = MemoryJobs(c, c.version, (frozen, c.version), continuous=True)
            # 初始摘要相同的历史权重仍有独立路由，更新当前模型不会借用其执行器。
            self.assertNotEqual(jobs.current, c.version)
            self.assertIs(jobs.inferences[jobs.current], current)
            self.assertIs(jobs.inferences[c.version].model, frozen)
            for name in "bcd":
                jobs.inferences[name * 64] = Mock(model=object())
            c.pool.slots = {
                i: {"models": [jobs.current, name * 64]} for i, name in enumerate("abcd")
            }
            with patch("haojie_training.console.controller.torch.load") as load:
                self.assertEqual(Controller._historical(c, jobs)[1], "a" * 64)
                load.assert_not_called()
                c.store.names.assert_not_called()
            expired = jobs.inferences["b" * 64]
            del c.pool.slots[1]
            new_model = Mock()
            new_model.to.return_value = new_model.eval.return_value = new_model
            with (
                patch(
                    "haojie_training.console.controller.torch.load",
                    return_value={"version": "e" * 64, "model": {}},
                ),
                patch("haojie_training.console.controller.PolicyValueNet", return_value=new_model),
            ):
                historical = Controller._historical(c, jobs)
            expired.close.assert_called_once()
            self.assertNotIn("b" * 64, jobs.inferences)
            jobs.next_window(c.version, historical)
            self.assertIs(jobs.inferences["e" * 64].model, new_model)
            self.assertEqual(len(jobs.inferences), 5)
            self.assertIs(jobs.inferences[jobs.current], current)


@unittest.skipUnless(os.environ.get("HAOJIE_NATIVE"), "需要真实 Rust 引擎")
class StreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.engine = os.environ["HAOJIE_NATIVE"]
        root = Path(__file__).resolve().parents[3]
        fixture = root / "tests/fixtures/native/late-starts.json.gz"
        starts = json.loads(
            gzip.decompress(fixture.read_bytes())
            if fixture.exists()
            else (root / "fixtures/late-starts.json").read_bytes()
        )
        cls.starts = [starts[0], starts[1], starts[0]]

    @contextmanager
    def controller(self, continuous=True):
        with tempfile.TemporaryDirectory() as folder:
            c = ManualController(
                self.engine, Store(folder), device="cpu", tiny=True, starts=self.starts, threads=1
            )
            c.thread.join()
            c.continuous_selfplay = continuous
            c.config.update(
                environments=2,
                games_per_round=2,
                updates_per_round=1,
                history_percent=0,
                pool_mib=128,
                eval_games=100000,
            )
            c.want_run = True
            try:
                c._initialize()
                yield c
            finally:
                c._close_pool()
                c.close()

    def test_long_game_crosses_real_update_and_keeps_behavior_labels(self):
        with self.controller() as c, limited_tasks(3):
            c._sample()
            self.assertEqual(c.counts["games"], 2)
            carried = {task["id"] for task in c.pool.slots.values()}
            self.assertTrue(carried)
            self.assertTrue(
                all(row["_meta"]["game"]["task"] not in carried for row, *_ in c.samples.rows)
            )
            old_version, pool = c.version, c.pool
            c._update()
            self.assertEqual(c.trainer.updates, 1)
            self.assertNotEqual(c.version, old_version)
            c._sample()
            self.assertIs(c.pool, pool)
            self.assertEqual(c.counts["games"], 3)
            self.assertEqual(c.counts["truncated"], 0)
            self.assertEqual(c.counts["discarded"], 0)
            rows = [row for row, *_ in c.samples.rows if row["_meta"]["game"]["task"] in carried]
            self.assertTrue(rows)
            versions = {row["_meta"]["behavior_version"] for row in rows}
            self.assertIn(c.version, versions)
            self.assertLessEqual(versions, {old_version, c.version})
            self.assertIn(1, {row["_meta"]["behavior_generation"] for row in rows})
            self.assertTrue(all(bool(row["value_mask"]) for row in rows))
            self.assertFalse(c.pool.slots)

    def test_evaluation_preserves_parked_training_games(self):
        with self.controller() as c, limited_tasks(3):
            c._sample()
            pool, jobs = c.pool, c.sampling_jobs
            self.assertTrue(pool.slots)
            task_objects = dict(pool.slots)
            progress = {i: task["trajectory"].seen for i, task in pool.slots.items()}
            pids = [client.process.pid for client in pool.clients]
            evaluation = Evaluation(c, c._weights())
            evaluation.plan = [
                {
                    "difficulty": "baseline",
                    "start": self.starts[0],
                    "side": 1,
                    "sampler_seed": 91,
                    "result": "pending",
                }
            ]
            report = evaluation.slice()
            self.assertIsNotNone(report)
            self.assertEqual(report["results"]["baseline"]["n"], 1)
            self.assertIs(c.pool, pool)
            self.assertIs(c.sampling_jobs, jobs)
            self.assertEqual([client.process.pid for client in pool.clients], pids)
            self.assertEqual(set(pool.slots), set(task_objects))
            for i, task in pool.slots.items():
                self.assertIs(task, task_objects[i])
                self.assertEqual(task["trajectory"].seen, progress[i])
            self.assertEqual(c.counts["discarded"], 0)
            c._close_pool()
            self.assertEqual(c.counts["discarded"], len(task_objects))
            c._close_pool()
            self.assertEqual(c.counts["discarded"], len(task_objects))

    def test_fixed_weights_complete_identical_finite_task_set(self):
        summaries = []
        for continuous in (False, True):
            with self.controller(continuous) as c, limited_tasks(3):
                for _ in range(3):
                    if c.counts["games"] == 3:
                        break
                    c._sample()
                    self.assertEqual(c.counts["truncated"], 0)
                    self.assertEqual(c.counts["discarded"], 0)
                    if not continuous:
                        self.assertIsNone(c.sampling_jobs)
                        self.assertEqual(c.runtime["historical_models"], 0)
                self.assertEqual(c.counts["games"], 3)
                self.assertEqual(c.counts["tasks"], 3)
                self.assertFalse(c.pool.slots)
                samples = []
                for row, *_ in c.samples.rows:
                    meta = row["_meta"]
                    digest = hashlib.sha256()
                    for key in sorted(k for k in row if not k.startswith("_")):
                        value = row[key].contiguous()
                        digest.update(f"{key}:{value.dtype}:{list(value.shape)}".encode())
                        digest.update(value.numpy().tobytes())
                    samples.append((meta["game"]["task"], meta["decision"], digest.hexdigest()))
                self.assertTrue(samples)
                summaries.append((c.counts["seen_samples"], sorted(samples)))
        self.assertEqual(summaries[0], summaries[1])


if __name__ == "__main__":
    unittest.main()
