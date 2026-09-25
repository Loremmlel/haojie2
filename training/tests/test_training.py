"""验证填充/候选语义、未决价值遮罩、真实优化更新和检查点续训。"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from haojie_training.data import FORMAT, load_dataset, synthetic_batch, validate_batch
from haojie_training.evaluate import value_baselines, value_diagnostics
from haojie_training.model import ModelConfig, PolicyValueNet, policy_value_loss
from haojie_training.runtime import Trainer
from haojie_training.train import initialize_weights


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)
        self.config = ModelConfig.tiny()
        self.batch = synthetic_batch(self.config, 3, 8, 6)

    def test_new_generation_transfers_weights_but_not_optimizer_or_data_identity(self):
        previous = Trainer(PolicyValueNet(self.config), torch.device("cpu"))
        previous.step(self.batch)
        metadata = {
            "synthetic": True,
            "ruleset": "test-only",
            "encoding": "test-v1",
            "dataset_sha256": "old",
            "groups": ["old-train"],
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "parent.pt"
            previous.save(path, metadata)
            current = Trainer(PolicyValueNet(self.config), torch.device("cpu"))
            new_metadata = {**metadata, "dataset_sha256": "new", "groups": ["new-train"]}
            initialize_weights(current, path, new_metadata)
            for old, new in zip(previous.model.parameters(), current.model.parameters()):
                torch.testing.assert_close(old, new, atol=0, rtol=0)
            self.assertEqual(current.updates, 0)
            self.assertFalse(current.optimizer.state)
            self.assertEqual(new_metadata["dataset_sha256"], "new")
            self.assertEqual(new_metadata["initialized_from"]["parent_updates"], 1)
            self.assertEqual(new_metadata["seen_training_groups"], ["new-train", "old-train"])
            with self.assertRaisesRegex(ValueError, "验证泄漏"):
                initialize_weights(current, path, new_metadata, ["old-train"])
            with self.assertRaisesRegex(ValueError, "不一致"):
                initialize_weights(current, path, {**new_metadata, "ruleset": "changed"})
            current.step(self.batch)
            with self.assertRaisesRegex(ValueError, "尚未开始"):
                initialize_weights(current, path, new_metadata)

    def test_initialized_generation_can_resume_through_cli_with_lineage_intact(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent, child = root / "parent.pt", root / "child.pt"
            base = [
                sys.executable,
                "-m",
                "haojie_training.train",
                "--synthetic",
                "--tiny",
                "--device",
                "cpu",
                "--threads",
                "2",
                "--steps",
                "1",
                "--batch-size",
                "2",
                "--entities",
                "3",
                "--actions",
                "3",
            ]

            def run(arguments):
                subprocess.run(
                    [*base, *arguments],
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )

            run(["--seed", "7", "--checkpoint", str(parent)])
            run(["--seed", "8", "--initialize-from", str(parent), "--checkpoint", str(child)])
            initialized = torch.load(child, map_location="cpu", weights_only=True)
            self.assertEqual(initialized["updates"], 1)
            self.assertEqual(initialized["metadata"]["seed"], 8)
            origin = initialized["metadata"]["initialized_from"]
            self.assertEqual(origin["parent_updates"], 1)
            run(["--seed", "8", "--resume", str(child), "--checkpoint", str(child)])
            resumed = torch.load(child, map_location="cpu", weights_only=True)
            self.assertEqual(resumed["updates"], 2)
            self.assertEqual(resumed["metadata"]["initialized_from"], origin)

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
            # 旧检查点没有权重字段，必须继续按原策略加价值目标恢复。
            legacy = torch.load(path, weights_only=True)
            legacy.pop("value_weight")
            torch.save(legacy, path)
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

    def test_value_ablation_and_checkpoint_objective_boundary(self):
        model = PolicyValueNet(self.config)
        output = model(self.batch)
        policy_only = policy_value_loss(output, self.batch, 0)
        changed = {**self.batch, "value": -self.batch["value"]}
        torch.testing.assert_close(policy_only, policy_value_loss(output, changed, 0))
        policy_only.backward()
        self.assertTrue(all(not p.grad.any() for p in model.value.parameters()))
        trainer = Trainer(model, torch.device("cpu"), value_weight=0)
        trainer.step(self.batch)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.pt"
            trainer.save(path, {})
            same = Trainer(PolicyValueNet(self.config), torch.device("cpu"), value_weight=0)
            same.restore(path, {})
            torch.testing.assert_close(trainer.step(self.batch), same.step(self.batch))
            with self.assertRaises(ValueError):
                Trainer(PolicyValueNet(self.config), torch.device("cpu")).restore(path, {})
        for weight in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                Trainer(model, torch.device("cpu"), value_weight=weight)

    def test_value_baselines_use_training_labels_and_exclude_unknown_values(self):
        training = {
            "value": torch.tensor([1.0, 1.0, -1.0, -1.0]),
            "value_mask": torch.tensor([True, True, True, False]),
        }
        records = [{"actor": a, "game_id": g} for a, g in [(1, "a"), (1, "a"), (2, "b"), (2, "c")]]
        baselines = value_baselines(training, records)
        self.assertAlmostEqual(baselines["mean"], 1 / 3)
        self.assertEqual(baselines["by_actor"], {"1": 1.0, "2": -1.0})
        validation = {**training, "value": -training["value"]}
        result = value_diagnostics(
            torch.tensor([1.0, 0.0, -1.0, 100.0]), validation, records, baselines
        )
        self.assertEqual(result["overall"]["labels"], 3)
        self.assertEqual(result["overall"]["train_actor_mean_mse"], 4)
        self.assertEqual(result["overall"]["zero_mse"], 1)
        self.assertEqual(result["overall"]["model_mse"], 3)
        self.assertEqual(result["game_macro_mse"], 3.25)
        self.assertEqual(result["by_actor"]["2"]["labels"], 1)


if __name__ == "__main__":
    unittest.main()
