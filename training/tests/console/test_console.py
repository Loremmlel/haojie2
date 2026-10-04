import gzip
import json
import os
import random
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from haojie_training.console.controller import Controller, defaults, identity, validate
from haojie_training.console.memory import SamplePool, Trajectory
from haojie_training.console.opponents import Teacher, summarize
from haojie_training.console.sampling import MemoryJobs
from haojie_training.console.storage import Lease, QuotaError, Store
from haojie_training.data import INPUT_KEYS, synthetic_batch
from haojie_training.model import ModelConfig, PolicyValueNet
from haojie_training.native.client import Client
from haojie_training.runtime import Trainer


def until(predicate, seconds=30):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("未在期限内满足条件")


class ResourceTests(unittest.TestCase):
    def test_adopted_original_counts_but_is_never_deleted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            legacy = root / "original"
            legacy.mkdir()
            (legacy / "model.pt").write_bytes(b"x" * 50)
            store = Store(root / "managed", limit=1000)
            store.adopt([legacy])
            self.assertEqual(store.snapshot()["categories"]["adopted"], 50)
            store.remove("adopted/0")
            self.assertEqual((legacy / "model.pt").stat().st_size, 50)
            loaded = Store(root / "managed", limit=1000)
            self.assertEqual(loaded.adopted, [str(legacy.resolve())])
            loaded.limit = loaded.snapshot()["used"] + 3
            with self.assertRaises(QuotaError):
                loaded.write("metrics/recent.json", b"1234")

    def test_historical_actor_routing_and_training_ownership(self):
        torch.set_num_threads(1)
        model = PolicyValueNet(ModelConfig.tiny())
        c = SimpleNamespace(
            config=defaults(),
            counts={"tasks": 0},
            starts=None,
            trainer=SimpleNamespace(model=model),
            device=torch.device("cpu"),
        )
        jobs = MemoryJobs(c, "current", (model, "historical"))
        task = jobs.claim()
        self.assertEqual(task["models"], ["historical", "current"])
        batch = synthetic_batch(model.config, size=1, entities=4, actions=2)
        frame = {
            "input": {k: batch[k][0] for k in INPUT_KEYS},
            "candidates": 2,
            "bytes": 2000,
            "selected": 0,
            "step": 0,
            "index": 0,
        }
        jobs.example(task, {**frame, "actor": 1, "model": "historical"})
        self.assertEqual(task["trajectory"].seen, 0)
        jobs.example(task, {**frame, "actor": 2, "model": "current"})
        self.assertEqual(task["trajectory"].seen, 1)
        calls = []

        def inference(name, value):
            def run(rows):
                calls.append((name, len(rows)))
                return torch.full((len(rows), 2), value)

            return run

        jobs.inferences = {
            "current": inference("current", 1.0),
            "historical": inference("historical", -1.0),
        }
        output = jobs.infer([{**frame, "model": name} for name in ("historical", "current")])
        self.assertEqual(set(calls), {("historical", 1), ("current", 1)})
        self.assertLess(output[0, 0], 0)
        self.assertGreater(output[1, 0], 0)

    def test_atomic_reservation_rotation_and_protected_last(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder, limit=100)
            store.write("recovery/0001.pt", b"a" * 30)
            store.write("recovery/0002.pt", b"b" * 30)
            with self.assertRaises(QuotaError):
                store.write("recovery/0003.pt", b"c" * 41)
            self.assertEqual(store.snapshot()["used"], 60)
            with self.assertRaises(ValueError):
                store.write(
                    "recovery/0003.pt",
                    b"c" * 30,
                    lambda _: (_ for _ in ()).throw(ValueError("校验失败")),
                )
            self.assertEqual(store.snapshot()["reserved"], 0)
            self.assertEqual(len(store.names("recovery")), 2)
            for i in range(1000):
                store.write("metrics/recent.json", b"0" * 10)
            self.assertLessEqual(store.snapshot()["used"], 100)
            self.assertEqual(len(store.files), 3)
            self.assertLessEqual(len(store.writes), 120)
            self.assertEqual((Path(folder) / "recovery/0001.pt").read_bytes(), b"a" * 30)

    def test_lock_duplicate_and_oversize_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            lease = Lease(store)
            with self.assertRaises(RuntimeError):
                Lease(store)
            lease.close()
            Lease(store).close()
        batch = synthetic_batch(ModelConfig.tiny(), size=1, entities=4, actions=2)
        message = {
            "input": {k: batch[k][0] for k in INPUT_KEYS},
            "selected": 0,
            "actor": 1,
            "step": 0,
            "bytes": 2000,
            "candidates": 2,
        }
        trajectory = Trajectory(4, capacity=16, sample_bytes=10000)
        for _ in range(10000):
            trajectory.add(message)
        self.assertEqual(len(trajectory.rows), 16)
        self.assertLessEqual(trajectory.used, 160000)
        trajectory.add({**message, "bytes": 1000000})
        self.assertEqual(trajectory.oversize, 1)
        pool = SamplePool(20000)
        for _ in range(1000):
            for row in trajectory.terminal({"1": 1, "2": -1}):
                pool.add(row)
        self.assertLessEqual(pool.used, pool.limit)
        while pool.rows:
            pool.batch(4, random.Random(9))
        self.assertEqual(pool.used, 0)

    def test_returns_change_the_actual_candidate_scores(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        model = PolicyValueNet(ModelConfig.tiny())
        trainer = Trainer(model, torch.device("cpu"), objective="decomposed-mc-q-v1", lr=0.002)
        batch = synthetic_batch(model.config, size=2, entities=4, actions=2)
        # 相同公开输入，两个实际动作分别获得 +1/-1；不是增加同策略模仿次数。
        for key in INPUT_KEYS:
            batch[key][1] = batch[key][0]
        batch["candidate_mask"][:] = True
        batch["policy"] = torch.eye(2)
        batch["value"] = torch.tensor([1.0, -1.0])
        batch["value_mask"][:] = True
        before = model.policy_logits(batch).detach().clone()
        for _ in range(24):
            trainer.step(batch)
        after = model.policy_logits(batch).detach().tanh()
        self.assertGreater(float(after[0, 0] - after[0, 1]), 0.5)
        self.assertFalse(torch.equal(before, model.policy_logits(batch)))
        self.assertEqual(trainer.updates, 24)

    def test_schedule_and_unknown_not_losses(self):
        validate(defaults())
        with self.assertRaises(ValueError):
            validate({**defaults(), "pool_mib": 5000})
        result = summarize(
            [
                {"difficulty": "easy", "result": "unfinished"},
                {"difficulty": "easy", "result": "error"},
            ]
        )
        self.assertIsNone(result["easy"]["win_rate"])
        self.assertEqual(result["easy"]["losses"], 0)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = os.environ["HAOJIE_NATIVE"]
        torch.set_num_threads(1)

    def test_memory_labels_match_authoritative_record_replay(self):
        with tempfile.TemporaryDirectory() as folder, Client(self.engine) as client:
            request = {
                "op": "sample",
                "start": {"seed": 71, "rules": "classic"},
                "model": "a" * 64,
                "samplerSeed": 91,
                "maxCommands": 10,
                "maxPlies": 10,
            }

            def collect():
                examples = []
                while True:
                    message = client.receive()
                    if message["type"] == "infer":
                        client.send(
                            {
                                "id": message["id"],
                                "model": message["model"],
                                "logits": [0.0] * message["candidates"],
                            }
                        )
                    elif message["type"] == "example":
                        examples.append(message)
                    elif message["type"] == "done":
                        return message, examples

            client.send({**request, "memory": True})
            memory_done, memory = collect()
            record = str(Path(folder) / "game.jsonl")
            client.send({**request, "record": record})
            file_done, _ = collect()
            self.assertEqual(memory_done["finalHash"], file_done["finalHash"])
            client.send({"op": "audit", "record": record, "encode": True})
            _, replay = collect()
            self.assertGreater(len(memory), 0)
            self.assertEqual(len(memory), len(replay))
            for actual, expected in zip(memory, replay, strict=True):
                for key in ("index", "actor", "step", "stage", "selected"):
                    self.assertEqual(actual[key], expected[key])
                for key in INPUT_KEYS:
                    self.assertTrue(torch.equal(actual["input"][key], expected["input"][key]))

    def test_real_teacher_all_difficulties_public_shrine(self):
        for difficulty in ("easy", "medium", "hard"):
            teacher = Teacher(difficulty)
            try:
                with Client(self.engine) as client:
                    client.send(
                        {
                            "op": "sample",
                            "memory": True,
                            "start": {"seed": 7, "rules": "shrine"},
                            "model": "a" * 64,
                            "teacher": 1,
                            "samplerSeed": 5,
                            "maxCommands": 1,
                            "maxPlies": 1,
                        }
                    )
                    request = client.receive()
                    self.assertEqual(request["type"], "teacher")
                    self.assertNotIn("rng", request["observation"])
                    command = teacher.next(request)
                    client.send({"id": request["id"], "command": command})
                    done = client.receive()
                    self.assertEqual(done["outcome"]["commands"], 1)
                    self.assertIsNone(done["error"])
            finally:
                teacher.close()

    def test_real_updates_pause_continue_save_restore_and_quota(self):
        root = Path(__file__).resolve().parents[3]
        fixture = root / "tests/fixtures/native/late-starts.json.gz"
        starts = json.loads(
            gzip.decompress(fixture.read_bytes())
            if fixture.exists()
            else (root / "fixtures/late-starts.json").read_bytes()
        )
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            c = Controller(self.engine, store, device="cpu", tiny=True, starts=starts[:2])
            try:
                c.command(
                    "settings",
                    {
                        "environments": 2,
                        "games_per_round": 2,
                        "updates_per_round": 2,
                        "pool_mib": 16,
                        "max_commands": 500,
                        "max_plies": 60,
                        "history_every": 2,
                    },
                )
                c.command("start")
                c.command("start")
                until(lambda: (c.trainer and c.trainer.updates >= 2) or c.state == "error", 60)
                self.assertNotEqual(c.state, "error", c.error)
                c.command("pause")
                until(lambda: c.state == "paused")
                self.assertGreater(c.counts["retained_samples"], 0)
                old = c.trainer.updates
                first = identity(c.trainer.model)
                optimizer = c.trainer.optimizer
                self.assertIsNone(c.pool)
                c.command("save")
                until(lambda: c.saved_at is not None or c.state == "error")
                self.assertNotEqual(c.state, "error", c.error)
                c.command("start")
                until(lambda: c.trainer.updates > old or c.state == "error", 60)
                c.command("pause")
                until(lambda: c.state == "paused")
                self.assertIs(c.trainer.optimizer, optimizer)
                self.assertNotEqual(identity(c.trainer.model), first)
                saved = c.saved_step
                saved_file = store.names("recovery")[-1]
                payload = torch.load(store.root / saved_file, weights_only=True)
                self.assertEqual(payload["format"], "haojie-stream-recovery-v1")
                self.assertNotIn("dataset_sha256", payload["trainer"]["metadata"])
            finally:
                c.close()
            restored = Controller(self.engine, Store(folder), device="cpu", tiny=True)
            try:
                restored.command("save")
                until(
                    lambda: (
                        restored.trainer is not None
                        and restored.saved_at is not None
                        or restored.state == "error"
                    )
                )
                until(lambda: restored.state in ("paused", "error"))
                self.assertNotEqual(restored.state, "error", restored.error)
                self.assertEqual(restored.trainer.updates, saved)
                self.assertEqual(len(restored.samples.rows), 0)
                self.assertTrue(restored.restored)
                self.assertEqual(identity(restored.trainer.model), first)
                # 假时钟检验10分钟仅在新更新后触发，暂停不自动补保存。
                with patch(
                    "haojie_training.console.controller.time.time",
                    return_value=restored.saved_at + 601,
                ):
                    restored.want_run = True
                    self.assertFalse(restored._due_save())
                    restored.trainer.updates += 1
                    self.assertTrue(restored._due_save())
                    restored.want_run = False
                    self.assertFalse(restored._due_save())
                restored.store.limit = restored.store.snapshot()["used"] + 1
                restored.command("save")
                until(lambda: restored.state == "error")
                self.assertIn("配额", restored.error)
                self.assertTrue((restored.store.root / saved_file).exists())
            finally:
                restored.close()
