"""执行路径的边界、缓冲所有权、CUDA输出/梯度与固定随机取样回归。"""

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from haojie_training.batching import training_batches
from haojie_training.data import INPUT_KEYS, collate_inputs, synthetic_batch, validate_inputs
from haojie_training.model import ModelConfig, PolicyValueNet, policy_value_loss
from haojie_training.native.client import decode
from haojie_training.native.execution import PolicyInference
from haojie_training.runtime import Trainer


class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_frame_ownership_and_rejection(self):
        batch = synthetic_batch(ModelConfig.tiny(), 1, 4, 5)
        keys = (
            "entities",
            "globals",
            "candidates",
            "kinds",
            "sources",
            "targets",
            "entity_mask",
            "candidate_mask",
        )
        data = bytearray(b"".join(batch[k][0].numpy().tobytes() for k in keys))
        inputs = decode(data, 4, 5)
        data[:] = b"\0" * len(data)
        for k in keys:
            torch.testing.assert_close(inputs[k], batch[k][0], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            decode(data[:-1], 4, 5)
        with self.assertRaises(ValueError):
            decode(data, 4, 5)
        for key, value in (
            ("entities", float("nan")),
            ("globals", float("inf")),
            ("kinds", 256),
            ("sources", -2),
            ("targets", 4),
            ("candidate_mask", False),
        ):
            invalid = {k: v.clone() for k, v in batch.items()}
            invalid[key].fill_(value)
            raw = bytearray(b"".join(invalid[k][0].numpy().tobytes() for k in keys))
            with self.assertRaises(ValueError):
                decode(raw, 4, 5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA专项回归，需要GPU执行器")
    def test_policy_buffers_full_value_and_gradients(self):
        torch.manual_seed(71)
        config = ModelConfig.tiny()
        model = PolicyValueNet(config).cuda().eval()
        runner = PolicyInference(model, "cuda")
        saved = []
        for entities, candidates, size in ((9, 6, 3), (3, 2, 1), (12, 8, 4)):
            batch = synthetic_batch(config, size, entities, candidates)
            examples = [{k: v[i] for k, v in batch.items() if k in INPUT_KEYS} for i in range(size)]
            inputs = collate_inputs(examples, config)
            validate_inputs(inputs, config)
            buffered = runner.batch(examples)
            self.assertTrue(all(t.is_contiguous() and t.is_pinned() for t in buffered.values()))
            with torch.inference_mode():
                logits, _ = model({k: v.cuda() for k, v in inputs.items()})
            result = runner(examples)
            torch.testing.assert_close(result, logits.cpu(), atol=2e-6, rtol=2e-5)
            saved.append((result, result.clone()))
        for output, snapshot in saved:
            torch.testing.assert_close(output, snapshot, atol=0, rtol=0)
        batch = {k: v.cuda() for k, v in synthetic_batch(config, 2, 9, 6).items()}
        other = copy.deepcopy(model)
        logits, value = model(batch)
        policy_value_loss((logits, value), batch).backward()
        x = other.encode(batch)
        policy_value_loss(
            (other.policy_logits(batch), other.value(x[:, 0]).squeeze(-1)), batch
        ).backward()
        for a, b in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA专项回归，需要GPU执行器")
    def test_prefetch_and_fused_update_restore(self):
        config = ModelConfig.tiny()
        dataset = synthetic_batch(config, 9, 7, 6)
        kwargs = dict(seed=17, start=3, steps=4, size=3, bucket_size=4)
        for plain, prefetched in zip(
            training_batches(dataset, config, **kwargs),
            training_batches(dataset, config, **kwargs, prefetch=True, pin_memory=True),
        ):
            for key in plain:
                torch.testing.assert_close(plain[key], prefetched[key], rtol=0, atol=0)
        a = Trainer(PolicyValueNet(config), torch.device("cuda"), optimizer="foreach")
        b = Trainer(copy.deepcopy(a.model), torch.device("cuda"), optimizer="fused")
        batch = {k: v[:3].cuda() for k, v in dataset.items()}
        a.step(batch)
        b.step(batch)
        for x, y in zip(a.model.parameters(), b.model.parameters()):
            torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            b.save(path, {"test": True})
            restored = Trainer(PolicyValueNet(config), torch.device("cuda"))
            restored.restore(path, {"test": True})
            b.step(batch)
            restored.step(batch)
            for x, y in zip(b.model.parameters(), restored.model.parameters()):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
            self.assertEqual(restored.updates, 2)


if __name__ == "__main__":
    unittest.main()
