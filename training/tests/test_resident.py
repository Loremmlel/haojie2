"""以确定性零 logits 验证工作池，不把替身吞吐当模型产能。"""

import gc
import os
import tempfile
import unittest
import weakref
from pathlib import Path

import torch

from haojie_training.native.audit import audit
from haojie_training.native.client import Client
from haojie_training.native.execution import PolicyInference
from haojie_training.native.pipeline import BatchJobs
from haojie_training.native.pool import Pool
from haojie_training.native.resident import Ledger, task_identity
from haojie_training.native.stream import prepare_record


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
            # 模拟账本提交后、宿主启动前退出；恢复必须新尝试编号、同一身份。
            ledger.close()
            ledger = Ledger(output, config, resume=True, tasks=3)
            ledger.recover(self.engine)
            ledger.begin({})
            again = ledger.claim()
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


if __name__ == "__main__":
    unittest.main()
