"""验证填充/候选语义、未决价值遮罩、真实优化更新和检查点续训。"""

import copy
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from haojie_training.data import FORMAT, load_dataset, synthetic_batch, validate_batch
from haojie_training.evaluate import value_baselines, value_diagnostics, value_quality
from haojie_training.model import ModelConfig, PolicyValueNet, policy_value_loss, stable_tanh
from haojie_training.runtime import BF16AdamW, Trainer, _finite
from haojie_training.train import initialize_weights


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)
        self.config = ModelConfig.tiny()
        self.batch = synthetic_batch(self.config, 3, 8, 6)

    def test_grouped_health_keeps_finite_extremes_and_detects_nonfinite_values(self):
        for device in ("cpu", *(["cuda"] if torch.cuda.is_available() else [])):
            with self.subTest(device=device):
                values = [
                    torch.empty(0, dtype=torch.bfloat16, device=device),
                    torch.tensor(
                        torch.finfo(torch.bfloat16).max, dtype=torch.bfloat16, device=device
                    ),
                    torch.ones((4, 6), dtype=torch.bfloat16, device=device)[:, ::2],
                ]
                self.assertTrue(_finite(values))
                for invalid in (float("nan"), float("inf"), -float("inf")):
                    values[-1][2, 1] = invalid
                    self.assertFalse(_finite(values))
                    values[-1][2, 1] = 1
                self.assertTrue(_finite(iter(values)))
                self.assertTrue(_finite([]))

    def test_bf16_adam_retains_small_updates_moment_decay_and_integer_steps(self):
        parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
        optimizer = BF16AdamW([parameter], weight_decay=0)
        for _ in range(1000):
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
        state = optimizer.state[parameter]
        self.assertAlmostEqual(parameter.item(), 0.7, delta=0.004)
        self.assertAlmostEqual(state["exp_avg_sq"].item(), 1 - 0.999**1000, delta=0.004)
        self.assertEqual(state["step"], 1000)
        self.assertIs(type(state["step"]), int)
        self.assertTrue(
            all(v.dtype == torch.bfloat16 for v in state.values() if isinstance(v, torch.Tensor))
        )
        restored_parameter = torch.nn.Parameter(parameter.detach().clone())
        restored = BF16AdamW([restored_parameter])
        restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        parameter.grad = torch.ones_like(parameter)
        restored_parameter.grad = torch.ones_like(restored_parameter)
        optimizer.step()
        restored.step()
        torch.testing.assert_close(parameter, restored_parameter, rtol=0, atol=0)

    def test_bf16_tanh_preserves_small_gradients_after_output_rounds_to_one(self):
        values = torch.tensor([0, 5, -5, 10, 20], dtype=torch.bfloat16, requires_grad=True)
        output = stable_tanh(values)
        torch.testing.assert_close(output, values.tanh(), rtol=0, atol=0)
        self.assertEqual(float(output[1].detach()), 1)
        output.sum().backward()
        self.assertEqual(values.grad.dtype, torch.bfloat16)
        for value, gradient in zip(values.tolist(), values.grad.tolist()):
            exponential = math.exp(-2 * abs(value))
            expected = 4 * exponential / (1 + exponential) ** 2
            self.assertAlmostEqual(gradient / expected, 1, delta=0.02)

    def test_bf16_storage_and_kernel_auxiliary_dtype_audit(self):
        class Audit(TorchDispatchMode):
            def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                result = function(*args, **(kwargs or {}))
                # 融合核的内部归一化统计单列；主输出、矩阵、梯度和优化器仍须为 BF16。
                auxiliary = {
                    "aten.native_layer_norm.default": {1, 2},
                    "aten._scaled_dot_product_flash_attention_for_cpu.default": {1},
                    "aten._scaled_dot_product_flash_attention.default": {1},
                    "aten._scaled_dot_product_efficient_attention.default": {1},
                }.get(str(function), set())
                for index, value in enumerate(result if isinstance(result, tuple) else (result,)):
                    if index in auxiliary:
                        continue
                    for tensor in tree_flatten(value)[0]:
                        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                            self_test.assertEqual(tensor.dtype, torch.bfloat16, str(function))
                return result

        self_test = self
        for name in ("cpu", *(["cuda"] if torch.cuda.is_available() else [])):
            with self.subTest(device=name), Audit():
                trainer = Trainer(PolicyValueNet(self.config), torch.device(name))
                batch = {key: value.to(name) for key, value in self.batch.items()}
                trainer.step(batch)
                payload = trainer.checkpoint({})
                self.assertEqual(payload["precision"], "bf16")
                self.assertNotIn("scaler", payload)
                self.assertTrue(all(p.dtype == torch.bfloat16 for p in trainer.model.parameters()))
                self.assertTrue(
                    all(type(state["step"]) is int for state in trainer.optimizer.state.values())
                )

    def test_bf16_adam_migrates_old_moments_without_mutating_source(self):
        parameter = torch.nn.Parameter(torch.ones(2))
        old = torch.optim.AdamW([parameter])
        parameter.grad = torch.ones_like(parameter)
        old.step()
        payload = old.state_dict()
        new_parameter = torch.nn.Parameter(parameter.detach().to(torch.bfloat16))
        new = BF16AdamW([new_parameter])
        new.load_state_dict(payload)
        self.assertEqual(payload["state"][0]["exp_avg"].dtype, torch.float32)
        self.assertEqual(new.state[new_parameter]["exp_avg"].dtype, torch.bfloat16)
        self.assertEqual(new.state[new_parameter]["step"], 1)
        self.assertNotIn("fused", new.param_groups[0])
        self.assertNotIn("foreach", new.param_groups[0])
        new_parameter.grad = torch.ones_like(new_parameter)
        new.step()
        self.assertEqual(new.state[new_parameter]["step"], 2)

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

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "需要 CUDA BF16"
    )
    def test_bf16_head_learns_adjacent_coordinates_and_resumes(self):
        device = torch.device("cuda")
        model = PolicyValueNet(self.config)
        trainer = Trainer(model, device, lr=0.002, objective="decomposed-mc-q-v2")
        batch = synthetic_batch(self.config, 2, 4, 1)
        # 两行公开上下文完全相同，监督仅依赖相邻格坐标。
        batch = {
            key: value[:1].expand(2, *value.shape[1:]).clone().to(device)
            for key, value in batch.items()
        }
        batch["candidates"].zero_()
        batch["candidates"][:, 0, 10] = torch.tensor(
            [5 / 12, 6 / 12], device=device, dtype=torch.bfloat16
        )
        batch["candidate_mask"].fill_(True)
        batch["policy"].fill_(1)
        batch["value_mask"].fill_(True)
        batch["value"] = torch.tensor([-0.8, 0.8], device=device, dtype=torch.bfloat16)
        dtypes = []
        handle = model.policy[0].register_forward_hook(
            lambda _module, _inputs, output: dtypes.append(output.dtype)
        )
        try:
            initial = float(trainer.step(batch))
        finally:
            handle.remove()
        self.assertEqual(dtypes, [torch.bfloat16])
        for _ in range(199):
            final = float(trainer.step(batch))
        self.assertLess(final, initial * 0.6)
        with torch.inference_mode():
            scores = model.policy_logits(batch).squeeze(1).tanh()
        self.assertLess(float(scores[0]), -0.5)
        self.assertGreater(float(scores[1]), 0.5)
        self.assertTrue(
            all(
                trainer.health()[key]
                for key in ("finite_parameters", "finite_gradients", "finite_optimizer")
            )
        )
        restored = Trainer(
            PolicyValueNet(self.config),
            device,
            lr=0.002,
            objective="decomposed-mc-q-v2",
        )
        restored.restore_payload(trainer.checkpoint({}), {})
        self.assertEqual(restored.updates, trainer.updates)
        for expected, actual in zip(model.parameters(), restored.model.parameters()):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        expected_states = trainer.optimizer.state_dict()["state"]
        for key, state in restored.optimizer.state_dict()["state"].items():
            for field, actual in state.items():
                torch.testing.assert_close(actual, expected_states[key][field], atol=0, rtol=0)
        torch.testing.assert_close(trainer.step(batch), restored.step(batch), atol=0, rtol=0)
        self.assertEqual(restored.updates, trainer.updates)
        self.assertTrue(restored.health()["finite_optimizer"])

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

    def test_value_gate_rejects_perfect_single_winner_and_seat_shortcut(self):
        records = [
            {"game_id": f"g{i}", "group": f"seed{i}", "actor": actor}
            for i in range(8)
            for actor in (1, 2)
        ]
        only_p2 = torch.tensor([-1.0, 1.0] * 8)
        data = {"value": only_p2, "value_mask": torch.ones(16, dtype=torch.bool)}
        baseline = {"mean": 0.0, "by_actor": {"1": -1.0, "2": 1.0}}
        report = value_diagnostics(only_p2, data, records, baseline)
        self.assertEqual(report["game_macro_mse"], 0)
        self.assertFalse(value_quality(report)["passed"])
        data["value"] = torch.tensor([1.0, -1.0] * 4 + [-1.0, 1.0] * 4)
        report = value_diagnostics(only_p2, data, records, baseline)
        self.assertFalse(value_quality(report)["passed"])
        report = value_diagnostics(
            data["value"] * 0.5, data, records, {"mean": 0.0, "by_actor": {}}
        )
        self.assertTrue(value_quality(report)["passed"])
        records[1]["actor"] = 1
        with self.assertRaisesRegex(ValueError, "收益相互矛盾"):
            value_diagnostics(only_p2, data, records, baseline)


if __name__ == "__main__":
    unittest.main()
