"""训练权重、冻结推理与未完评测共用一份模型契约。"""

import tempfile
import unittest
from pathlib import Path

import torch

from haojie_training.online.model import Config
from haojie_training.online.policy import export, frozen_session, load, start_pairs
from haojie_training.online.session import Session, Settings


class FrozenPolicy(unittest.TestCase):
    def test_export_inference_and_resume_use_identical_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            trainer = Session(
                Settings(concurrency=0, workers=1),
                "cpu",
                Config(width=32, layers=1, heads=4, ffn=64, history=16),
            )
            try:
                trainer.save(path / "train.pt")
                export(path / "train.pt", path / "policy.pt")
                model, _ = load(path / "policy.pt")
                for a, b in zip(model.parameters(), trainer.learner.model.parameters()):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            finally:
                trainer.close()
            session = frozen_session(
                path / "policy.pt", path / "policy.pt", device="cpu", workers=2
            )
            restored = None
            try:
                start_pairs(session, [173])
                weights = {k: v.clone() for k, v in session.models[0].state_dict().items()}
                for _ in range(30):
                    session.tick(admit=False)
                    for i, worker in enumerate(session.workers):
                        if worker.busy:
                            session._accept(i, worker.wait())
                self.assertEqual(session.learner.steps, 0)
                self.assertEqual(session.learner.consumed, 0)
                self.assertFalse(session.traces.pending or session.traces.ready)
                session.save(path / "inflight.pt")
                restored = Session.restore(path / "inflight.pt", "cpu")
                self.assertFalse(restored.training)
                self.assertEqual(session.games, restored.games)
                for k, value in restored.models[0].state_dict().items():
                    torch.testing.assert_close(value, weights[k], rtol=0, atol=0)
            finally:
                session.close()
                if restored:
                    restored.close()
