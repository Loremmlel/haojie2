"""验证填充/候选语义、未决价值遮罩、真实优化更新和检查点续训。"""

import tempfile
import unittest
from pathlib import Path

import torch

from haojie_training.data import FORMAT, load_dataset, synthetic_batch, validate_batch
from haojie_training.model import ModelConfig, PolicyValueNet, policy_value_loss
from haojie_training.runtime import Trainer


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)
        self.config = ModelConfig.tiny()
        self.batch = synthetic_batch(self.config, 3, 8, 6)

    def test_masks_and_value_perspective_boundary(self):
        model = PolicyValueNet(self.config).eval()
        expected = model(self.batch)
        changed = {key: value.clone() for key, value in self.batch.items()}
        changed["entities"][~changed["entity_mask"]] = 100
        changed["kinds"][~changed["entity_mask"]] = 200
        actual = model(changed)
        for before, after in zip(expected, actual):
            torch.testing.assert_close(before, after)
        probabilities = expected[0].softmax(-1)
        self.assertEqual(float(probabilities[~self.batch["candidate_mask"]].detach().sum()), 0)
        self.batch["value_mask"].zero_()
        loss = policy_value_loss(model(self.batch), self.batch)
        self.batch["value"].neg_()
        torch.testing.assert_close(policy_value_loss(model(self.batch), self.batch), loss)
        loss.backward()
        self.assertTrue(
            all(p.grad is not None and not p.grad.any() for p in model.value.parameters())
        )
        changed["entity_mask"].zero_()
        changed["sources"].fill_(-1)
        changed["targets"].fill_(-1)
        validate_batch(changed, self.config)
        self.assertTrue(all(t.isfinite().all() for t in model(changed)))

    def test_data_validation_and_dataset_roundtrip(self):
        with self.assertRaises(ValueError):
            validate_batch({**self.batch, "rng": torch.tensor(7)}, self.config)
        broken = {key: value.clone() for key, value in self.batch.items()}
        broken["candidate_mask"].zero_()
        with self.assertRaises(ValueError):
            validate_batch(broken, self.config)
        broken = {key: value.clone() for key, value in self.batch.items()}
        broken["targets"][0, 0] = 100
        with self.assertRaises(ValueError):
            validate_batch(broken, self.config)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dataset.pt"
            metadata = {"synthetic": True, "ruleset": "test-only", "encoding": "test-v1"}
            torch.save({"format": FORMAT, "metadata": metadata, "tensors": self.batch}, path)
            loaded, provenance = load_dataset(path, self.config)
            self.assertEqual(provenance, metadata)
            torch.testing.assert_close(loaded["policy"], self.batch["policy"])

    def test_optimizer_learns_and_resume_reproduces_next_update(self):
        trainer = Trainer(PolicyValueNet(self.config), torch.device("cpu"), lr=0.002)
        initial = float(trainer.step(self.batch))
        for _ in range(9):
            final = float(trainer.step(self.batch))
        self.assertLess(final, initial * 0.7)
        self.assertEqual(trainer.updates, 10)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            metadata = {"synthetic": True, "ruleset": "test-only", "encoding": "test-v1"}
            trainer.save(path, metadata)
            resumed = Trainer(PolicyValueNet(self.config), torch.device("cpu"), lr=0.002)
            resumed.restore(path, metadata)
            torch.testing.assert_close(
                trainer.step(self.batch), resumed.step(self.batch), atol=0, rtol=0
            )
            for before, after in zip(trainer.model.parameters(), resumed.model.parameters()):
                torch.testing.assert_close(before, after, atol=0, rtol=0)
            with self.assertRaises(ValueError):
                resumed.restore(path, {**metadata, "ruleset": "changed-rules"})

    def test_default_parameter_budget_and_variable_entity_counts(self):
        with torch.device("meta"):
            count = sum(p.numel() for p in PolicyValueNet().parameters())
        self.assertGreater(count, 10_000_000)
        self.assertLess(count, 15_000_000)
        model = PolicyValueNet(self.config)
        for entities in (1, 3, 17):
            batch = synthetic_batch(self.config, 2, entities, 5)
            self.assertEqual(tuple(model(batch)[0].shape), (2, 5))


if __name__ == "__main__":
    unittest.main()
