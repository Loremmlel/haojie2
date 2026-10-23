"""真实跨进程链路：更新时推进、队列背压、伸缩及故障后的终局记账。"""

import gzip
import json
import os
import queue
import tempfile
import threading
import time
import unittest
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from haojie_training.console.controller import Controller, identity
from haojie_training.console.pipeline import ProducerJobs, encode_weights
from haojie_training.console.storage import Store
from haojie_training.model import ModelConfig, PolicyValueNet


class ManualController(Controller):
    def _loop(self):
        pass


class PublicationLeaseTests(unittest.TestCase):
    def test_publication_reuses_owned_snapshot_and_refreshes_its_identity(self):
        c = Controller.__new__(Controller)
        c.trainer = SimpleNamespace(model=PolicyValueNet(ModelConfig.tiny()), updates=0)
        c.device, c.meta = torch.device("cpu"), {}
        c.version = c.weight_snapshot = None
        first = c._weights()
        self.assertIs(first["model"], c._weights()["model"])
        self.assertEqual(first["version"], identity(c.trainer.model))
        before = {key: value.clone() for key, value in first["model"].items()}
        with torch.no_grad():
            next(c.trainer.model.parameters()).add_(0.25)
        c.trainer.updates += 1
        after = c._weights()
        self.assertNotEqual(first["version"], after["version"])
        self.assertEqual(after["version"], identity(c.trainer.model))
        for key, value in first["model"].items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_full_terminal_queue_stages_completed_slot_and_stops_new_admission(self):
        jobs = ProducerJobs.__new__(ProducerJobs)
        jobs.output = queue.Queue(1)
        jobs.output.put(b"occupied")
        jobs.completed = deque()
        jobs.blocked_slots = set()
        jobs.finished = 0
        jobs.pressure_since = None
        jobs.backpressure_seconds = jobs.transfer_seconds = 0.0
        task = dict(id=1, context={}, trajectory=None, requests=2, began=0, slot=3, pid=4)
        jobs.finish(task, {"outcome": {"reason": "terminal"}})
        self.assertEqual(jobs.finished, 1)
        self.assertEqual(len(jobs.completed), 1)
        self.assertFalse(jobs.admission_open(3))
        self.assertTrue(jobs.admission_open(4))
        self.assertTrue(jobs.waiting_for_admission())
        jobs.output.get()
        jobs.flush()
        self.assertEqual(len(jobs.completed), 0)
        self.assertTrue(jobs.output.full())
        self.assertTrue(jobs.admission_open(3))
        jobs.output.get()
        self.assertGreater(jobs.pressure_seconds(), 0)

    def test_selected_history_survives_collection_and_releases_after_plan_and_games(self):
        current, historical = Mock(precision="bf16", stats=object()), Mock()
        historical.eval.return_value = historical
        c = SimpleNamespace(
            config=dict(
                method="decomposed-mc-q-v2",
                temperature=0.25,
                max_commands=20000,
                max_plies=1000,
                games_per_round=4,
                seed=17,
                history_percent=100,
            ),
            pool=SimpleNamespace(inference=current, slots={}),
            generation=0,
            trainer=SimpleNamespace(updates=0),
            device="cpu",
            counts={"tasks": 0},
            starts=None,
        )
        with patch(
            "haojie_training.console.sampling.PolicyInference",
            side_effect=lambda model, *a, **k: Mock(model=model),
        ):
            jobs = ProducerJobs(c, "a" * 64, (historical, "b" * 64), queue.Queue(4), None, True)
            jobs.release_unused()
            jobs.release_unused()
            self.assertIs(jobs.inference_for({"model": "b" * 64}).model, historical)
            task = jobs.claim()
            self.assertIn("b" * 64, task["models"])
            c.pool.slots[0] = task
            jobs.historical = None
            jobs.release_unused()
            self.assertIn("b" * 64, jobs.inferences)
            c.pool.slots.clear()
            jobs.release_unused()
            self.assertNotIn("b" * 64, jobs.inferences)


@unittest.skipUnless(os.environ.get("HAOJIE_NATIVE"), "需要真实 Rust 引擎")
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        root = Path(__file__).resolve().parents[4]
        fixture = root / "tests/fixtures/native/late-starts.json.gz"
        cls.starts = json.loads(
            gzip.decompress(fixture.read_bytes())
            if fixture.exists()
            else (root / "fixtures/late-starts.json").read_bytes()
        )

    @contextmanager
    def controller(self, indices=(0, 1)):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(os.environ, {"HAOJIE_HOST_WORKERS": "1"}),
        ):
            c = ManualController(
                os.environ["HAOJIE_NATIVE"],
                Store(folder),
                device="cpu",
                tiny=True,
                threads=1,
                starts=[self.starts[i] for i in indices],
            )
            c.thread.join()
            c.sampling_backend = "pipeline"
            c.config.update(
                environments=2,
                games_per_round=4,
                updates_per_round=1,
                history_percent=0,
                pool_mib=128,
                eval_games=100000,
            )
            c.want_run = True
            c._initialize()
            timer = threading.Timer(90, c.cancel.set)
            timer.start()
            try:
                yield c
            finally:
                timer.cancel()
                c._close_pool()
                c.close()

    def until(self, c, condition, seconds=10):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            c.pool.refresh(c)
            if condition():
                return
            time.sleep(0.02)
        self.fail("采样状态没有在时限内推进")

    def test_sampling_progresses_without_learner_driving_requests(self):
        with self.controller() as c:
            c._sample()
            remote = c.pool
            old_version = c.version
            carried = {t["id"]: t["pid"] for t in remote.slots.values()}
            before = remote.metrics()["inference_requests"]
            c._update()
            self.assertEqual(c.trainer.updates, 1)
            self.assertNotEqual(c.version, old_version)
            self.until(
                c,
                lambda: (
                    remote.metrics()["inference_requests"] > before
                    and remote.metrics()["published_model"] == c.version
                ),
            )
            active = {t["id"]: t["pid"] for t in remote.slots.values()}
            self.assertTrue(set(carried) & set(active))
            for task in set(carried) & set(active):
                self.assertEqual(carried[task], active[task])
            self.assertEqual(c.counts["discarded"], 0)
            for _ in range(4):
                c._sample()
                versions = {row["_meta"]["behavior_version"] for row, *_ in c.samples.rows}
                if c.version in versions:
                    break
            self.assertIn(c.version, versions)
            self.assertLessEqual(versions, {old_version, c.version})
            self.assertTrue(all(bool(row["value_mask"]) for row, *_ in c.samples.rows))

    def test_bounded_backpressure_resize_and_idempotent_close(self):
        with self.controller((0, 2)) as c:
            c._sample()
            remote = c.pool
            self.until(c, lambda: remote.output.qsize() == 4, seconds=30)
            self.assertEqual(remote.metrics()["terminal_queue_capacity"], 4)
            c.command("concurrency", {"environments": 4})
            end = time.perf_counter() + 15
            while remote.environments != 4 and time.perf_counter() < end:
                c._sample()
            self.assertEqual(remote.environments, 4)
            c.command("concurrency", {"environments": 1})
            end = time.perf_counter() + 20
            while remote.environments != 1 and time.perf_counter() < end:
                c._sample()
            self.assertEqual(remote.environments, 1)
            self.assertEqual(c.counts["discarded"], 0)
            c._close_pool()
            counts = dict(c.counts)
            c._close_pool()
            self.assertEqual(c.counts, counts)
            self.assertFalse(remote.process.is_alive())
            self.assertEqual(c.sampling_totals["requests"], remote.metrics()["requests"])
            self.assertGreater(c.counts["games"], 0)
            self.assertEqual(c.counts["truncated"], 0)

    def test_failed_publication_reports_error_and_keeps_received_terminals(self):
        with self.controller() as c:
            c._sample()
            games, samples = c.counts["games"], c.counts["retained_samples"]
            c.pool.publications.put(
                encode_weights(
                    {
                        "updates": 1,
                        "generation": 1,
                        "version": "invalid",
                        "config": {},
                        "model": {},
                        "history": None,
                    }
                )
            )
            with self.assertRaises(RuntimeError):
                while True:
                    c._sample()
            self.assertGreaterEqual(c.counts["games"], games)
            self.assertGreaterEqual(c.counts["retained_samples"], samples)
            self.assertTrue(all(bool(row["value_mask"]) for row, *_ in c.samples.rows))
