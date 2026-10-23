"""整个学习会话的在线伸缩、发布与未完成工作恢复。"""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from haojie_training.online.model import Config
from haojie_training.online.session import MemoryBudgetExceeded, Session, Settings


class SessionRecovery(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False),
        "需要真实原生 BF16 CUDA",
    )
    def test_frozen_policy_preserves_native_choices(self):
        config = Config(width=32, layers=1, heads=4, ffn=64, history=16)
        a = Session(Settings(concurrency=4, workers=2), "cuda", config, training=False)
        b = Session(Settings(concurrency=4, workers=2), "cuda", config, training=False)
        try:
            for _ in range(30):
                for session in (a, b):
                    session.tick()
                    for i, worker in enumerate(session.workers):
                        if worker.busy:
                            session._accept(i, worker.wait())
                self.assertEqual(a.metrics["microsteps"], b.metrics["microsteps"])
            for wa, wb in zip(a.workers, b.workers):
                wa.submit(1)
                wb.submit(1)
                self.assertEqual(json.loads(wa.wait()), json.loads(wb.wait()))
        finally:
            a.close()
            b.close()

    def test_impossible_budget_preserves_work_before_random_choice(self):
        session = Session(
            Settings(concurrency=1, workers=1),
            "cpu",
            Config(width=32, layers=1, heads=4, ffn=64, history=16),
        )
        try:
            session.tick()
            session._accept(0, session.workers[0].wait())
            session.workers[0].submit(1)
            before = session.workers[0].wait()
            session.settings.memory_bytes = session.memory() // 2
            with self.assertRaises(MemoryBudgetExceeded):
                session.tick()
            session.workers[0].submit(1)
            self.assertEqual(before, session.workers[0].wait())
            self.assertEqual(session.learner.steps, 0)
        finally:
            session.close()

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False),
        "需要真实原生 BF16 CUDA",
    )
    def test_cuda_two_workers_saved_work_and_optimizer_continue(self):
        settings = Settings(
            concurrency=4, workers=2, block=4, logical_batch=8, microbatch=8, publish_steps=2
        )
        config = Config(width=32, layers=1, heads=4, ffn=64, history=16)
        a, b = Session(settings, "cuda", config), None

        def advance(session):
            session.tick()
            for i, worker in enumerate(session.workers):
                if worker.busy:
                    session._accept(i, worker.wait())

        try:
            for _ in range(80):
                advance(a)
                if a.learner.steps >= 3:
                    break
            self.assertGreaterEqual(a.learner.steps, 3)
            with tempfile.TemporaryDirectory() as folder:
                checkpoint = Path(folder) / "cuda.pt"
                a.save(checkpoint)
                b = Session.restore(checkpoint, "cuda")
                for pa, pb in zip(a.learner.model.parameters(), b.learner.model.parameters()):
                    torch.testing.assert_close(pa, pb, rtol=0, atol=0)
                for sa, sb in zip(
                    a.learner.optimizer.state.values(), b.learner.optimizer.state.values()
                ):
                    for key in sa:
                        if isinstance(sa[key], torch.Tensor):
                            torch.testing.assert_close(sa[key], sb[key], rtol=0, atol=0)
                        else:
                            self.assertEqual(sa[key], sb[key])
                a.settings.concurrency = b.settings.concurrency = 1
                for _ in range(12):
                    advance(a)
                    advance(b)
                    self.assertEqual(a.metrics["microsteps"], b.metrics["microsteps"])
                    self.assertEqual(a.learner.consumed, b.learner.consumed)
                    self.assertEqual(a.traces.acknowledged, b.traces.acknowledged)
                for pa, pb in zip(a.learner.model.parameters(), b.learner.model.parameters()):
                    # 重建缓存可改变 BF16 内核分批舍入，保存边界本身仍逐位检查。
                    torch.testing.assert_close(pa, pb, rtol=0.02, atol=0.001)
                for wa, wb in zip(a.workers, b.workers):
                    wa.submit(1)
                    wb.submit(1)
                    self.assertEqual(json.loads(wa.wait()), json.loads(wb.wait()))
        finally:
            a.close()
            if b:
                b.close()

    def test_saved_work_is_consumed_once(self):
        settings = Settings(
            concurrency=2, workers=1, block=4, logical_batch=8, microbatch=8, publish_steps=2
        )
        config = Config(width=32, layers=1, heads=4, ffn=64, history=16)
        a = Session(settings, "cpu", config)
        b = None
        try:
            for _ in range(200):
                a.tick()
                if a.workers[0].busy:
                    a._accept(0, a.workers[0].wait())
                if a.learner.steps >= 3:
                    break
            self.assertGreaterEqual(a.learner.steps, 3)
            with tempfile.TemporaryDirectory() as folder:
                checkpoint = Path(folder) / "checkpoint.pt"
                a.save(checkpoint)
                b = Session.restore(checkpoint, "cpu")
                a.settings.concurrency = b.settings.concurrency = 1
                for _ in range(30):
                    for session in (a, b):
                        session.tick()
                        if session.workers[0].busy:
                            session._accept(0, session.workers[0].wait())
                    self.assertEqual(a.metrics["commands"], b.metrics["commands"])
                    self.assertEqual(a.learner.consumed, b.learner.consumed)
                    self.assertEqual(a.traces.acknowledged, b.traces.acknowledged)
                    self.assertLessEqual(len(a.models), 2)
                for pa, pb in zip(a.learner.model.parameters(), b.learner.model.parameters()):
                    torch.testing.assert_close(pa, pb, rtol=0, atol=0)
        finally:
            a.close()
            if b:
                b.close()


if __name__ == "__main__":
    unittest.main()
