"""以确定性零 logits 验证工作池，不把替身吞吐当模型产能。"""

import gc
import io
import json
import os
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace

import torch

from haojie_training.native.audit import audit
from haojie_training.native.client import Client
from haojie_training.native.execution import PolicyInference
from haojie_training.native.pipeline import BatchJobs
from haojie_training.native.pool import Pool
from haojie_training.native.resident import Ledger, task_identity
from haojie_training.native.stream import prepare_pool, prepare_record


class ZeroPolicy:
    def policy_logits(self, batch):
        return torch.zeros(batch["candidate_mask"].shape)


class ResidentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Path(os.environ["HAOJIE_NATIVE"])
        if not cls.engine.is_file():
            raise RuntimeError("需要真实原生引擎")
        torch.set_num_threads(1)

    def pool(self, environments=2):
        return Pool(
            self.engine,
            None,
            environments,
            "cpu",
            inference=(PolicyInference(ZeroPolicy(), "cpu"), "a" * 64),
        )

    def test_eof_during_frame_is_restartable_but_corruption_is_not(self):
        for payload, kind in (
            (b'{"type":"infer"', RuntimeError),
            (b'{"bytes":666,"entities":1,"candidates":1}\nabc', RuntimeError),
            (b'{"bytes":10,"entities":1,"candidates":1}\nabc', ValueError),
            (b"{bad json}\n", ValueError),
        ):
            client = Client.__new__(Client)
            client.frame_limit = 256 * 1024 * 1024
            client.process = SimpleNamespace(stdout=io.BytesIO(payload))
            results = []
            client._publish = results.append
            client._read()
            self.assertIsInstance(results[-1], kind)
            if kind is RuntimeError:
                self.assertEqual(str(results[-1]), "原生进程已退出")

    def test_large_valid_frame_is_distinct_from_resource_and_protocol_errors(self):
        entities = 8200
        body = (
            torch.zeros(entities * 64 + 32 + 64).numpy().tobytes()
            + torch.zeros(entities, dtype=torch.int64).numpy().tobytes()
            + torch.full((2,), -1, dtype=torch.int64).numpy().tobytes()
            + bytes([1]) * (entities + 1)
        )
        self.assertGreater(len(body), 2 * 1024 * 1024)
        header = (
            json.dumps(
                {"type": "infer", "bytes": len(body), "entities": entities, "candidates": 1}
            ).encode()
            + b"\n"
        )
        for limit, kind in ((4 * 1024 * 1024, dict), (1024 * 1024, MemoryError)):
            client = Client.__new__(Client)
            client.frame_limit = limit
            client.read_seconds = client.decode_seconds = client.tensor_bytes = 0
            client.process = SimpleNamespace(stdout=io.BytesIO(header + body))
            result = []
            client._publish = result.append
            client._read()
            self.assertIsInstance(result[0], kind)
            if kind is dict:
                self.assertEqual(result[0]["input"]["entities"].shape[0], entities)

    def test_refill_identity_and_frame_lifetime(self):
        starts = [task_identity(97, i)["start"] for i in range(9)]
        with tempfile.TemporaryDirectory() as folder:
            hashes = []
            for count in (1, 4):
                output = Path(folder) / str(count)
                output.mkdir()
                jobs = BatchJobs(starts, output, 74, 6, 100)
                pool = self.pool(count)
                pids = [c.process.pid for c in pool.clients]
                refs = []
                original = pool.inference.batch

                def batch(examples):
                    refs.extend(weakref.ref(e["entities"]) for e in examples)
                    return original(examples)

                pool.inference.batch = batch
                report = pool.run(jobs)
                gc.collect()
                self.assertTrue(all(r() is None for r in refs))
                self.assertEqual(pids, [c.process.pid for c in pool.clients])
                self.assertEqual(report["assigned"], 9)
                self.assertGreater(report["refills"], 0)
                self.assertLessEqual(report["peak_queue"], 2 * count)
                hashes.append([r["finalHash"] for r in jobs.results])
            self.assertEqual(hashes[0], hashes[1])

    def test_stale_request_and_process_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            jobs = BatchJobs([{"seed": 71, "rules": "classic"}], output, 91, 5, 100)
            pool = self.pool(1)
            pool.last_ids[0] = 999
            with self.assertRaisesRegex(ValueError, "过期"):
                pool.run(jobs)

    def test_cancel_resume_and_configuration_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            config = {"seed": 77, "rules": "mixed", "commands": 10, "plies": 100, "model": "a" * 64}
            ledger = Ledger(output, config, tasks=3)
            ledger.begin({})
            first = ledger.claim()
            ledger.db.execute(
                "UPDATE attempts SET result=? WHERE task=? AND attempt=1",
                (json.dumps({"requests": 7, "seconds": 1.25}), first["id"]),
            )
            ledger.db.commit()
            # 模拟账本提交后、宿主启动前退出；恢复必须新尝试编号、同一身份。
            ledger.close()
            ledger = Ledger(output, config, resume=True, tasks=3)
            ledger.recover(self.engine)
            previous_attempt = json.loads(
                ledger.db.execute("SELECT result FROM attempts WHERE attempt=1").fetchone()[0]
            )
            self.assertEqual(previous_attempt["requests"], 7)
            self.assertEqual(previous_attempt["seconds"], 1.25)
            ledger.begin({})
            again = ledger.claim()
            self.assertIsNone(
                ledger.db.execute("SELECT result FROM tasks WHERE id=?", (again["id"],)).fetchone()[
                    0
                ]
            )
            self.assertEqual(first["start"], again["start"])
            self.assertEqual(first["sampler_seed"], again["sampler_seed"])
            self.assertEqual(again["attempt"], 2)
            self.assertNotEqual(first["record"], again["record"])
            again["requests"] = 0
            ledger.failed(again, "test")
            pool = self.pool(2)
            report = pool.run(ledger)
            self.assertEqual(report["finished"], 3)
            ledger.close()
            ledger = Ledger(output, config, resume=True, tasks=3)
            ledger.recover(self.engine)
            self.assertIsNone(ledger.claim())
            ledger.close()
            with self.assertRaisesRegex(ValueError, "不匹配"):
                Ledger(output, {**config, "seed": 78}, resume=True)

    def test_stop_and_replay_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            config = {"seed": 81, "rules": "mixed", "commands": 20000, "plies": 1000}
            ledger = Ledger(output, config, tasks=2)
            ledger.begin({})
            pool = self.pool(2)
            report = pool.run(ledger, seconds=0.5 + pool.startup_seconds, drain_seconds=0.1)
            self.assertEqual(report["assigned"], 2)
            self.assertEqual(ledger.terminal, 0)
            self.assertGreater(len(list(output.rglob("*.jsonl*"))), 0)
            self.assertEqual(sum(ledger.summary()["attempts"].values()), 2)
            ledger.close()

    def test_streaming_audit_matches_existing_encoder(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            jobs = BatchJobs([{"seed": 71, "rules": "classic"}], output, 91, 6, 100)
            self.pool(1).run(jobs)
            path = output / "game-0.jsonl"
            reference = audit(self.engine, path)
            with Client(self.engine) as client:
                report = prepare_record(client, path, output / "prepared", shard_size=3)
            self.assertEqual(report["samples"], len(reference.inputs))
            self.assertEqual(report["audit"], reference.report)
            self.assertEqual(report["value_labels"], 0)
            index = 0
            for file in sorted((output / "prepared").glob("*.pt")):
                payload = torch.load(file, weights_only=True)
                tensors = payload["tensors"]
                for row in range(len(tensors["value"])):
                    inputs, selected = reference.inputs[index]
                    for key, value in inputs.items():
                        actual = tensors[key][row]
                        if value.ndim:
                            actual = actual[: len(value)]
                        torch.testing.assert_close(actual, value, atol=0, rtol=0)
                    self.assertEqual(int(tensors["policy"][row].argmax()), selected)
                    index += 1

    def test_worker_exit_restarts_only_its_task(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            config = {"seed": 81, "rules": "mixed", "commands": 10, "plies": 100}
            ledger = Ledger(output, config, tasks=3)
            ledger.begin({})
            pool = self.pool(2)
            killed = []

            def snapshot(_):
                if not killed:
                    killed.append(pool.clients[0].process.pid)
                    pool.clients[0].process.terminate()

            report = pool.run(ledger, snapshot=snapshot)
            self.assertEqual(report["restarts"], 1)
            self.assertEqual(report["finished"], 3)
            self.assertEqual(ledger.summary()["attempts"]["process-exit"], 1)
            ledger.close()

    def test_missing_completed_record_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            ledger = Ledger(
                output, {"seed": 81, "rules": "classic", "commands": 3, "plies": 100}, tasks=1
            )
            ledger.begin({})
            self.pool(1).run(ledger)
            path = Path(ledger.db.execute("SELECT path FROM attempts").fetchone()[0])
            ledger.close()
            prepared = Path(folder) / "audit-complete"
            prepare_pool(self.engine, output, prepared, shard_size=3)
            progress = json.loads((prepared / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["attempts"], 1)
            self.assertEqual(progress["pending_attempts"], 0)
            path.unlink()
            with self.assertRaisesRegex(ValueError, "已完成尝试缺少记录"):
                prepare_pool(self.engine, output, Path(folder) / "audit")
            progress = json.loads(
                (Path(folder) / "audit/progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(progress["pending_attempts"], 1)

    def test_published_record_recovered_and_error_not_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            config = {"seed": 81, "rules": "classic", "commands": 6, "plies": 100}
            ledger = Ledger(output, config, tasks=1)
            ledger.begin({})
            finish = ledger.finish

            def interrupted_finish(task, result):
                raise RuntimeError("模拟发布完成文件后退出")

            ledger.finish = interrupted_finish
            with self.assertRaisesRegex(RuntimeError, "模拟发布"):
                self.pool(1).run(ledger)
            ledger.finish = finish
            ledger.close()
            ledger = Ledger(output, config, resume=True, tasks=1)
            ledger.recover(self.engine)
            self.assertIsNone(ledger.claim())
            self.assertEqual(ledger.summary()["task_states"], {"commands": 1})
            ledger.db.execute("UPDATE tasks SET status='error'")
            ledger.db.commit()
            with self.assertRaisesRegex(ValueError, "拒绝跳过"):
                ledger.recover(self.engine)
            ledger.close()


if __name__ == "__main__":
    unittest.main()
