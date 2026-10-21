"""执行路径的边界、缓冲所有权、CUDA输出/梯度与固定随机取样回归。"""

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from haojie_training.batching import training_batches
from haojie_training.data import INPUT_KEYS, collate_inputs, synthetic_batch, validate_inputs
from haojie_training.model import ModelConfig, PolicyValueNet, policy_value_loss
from haojie_training.native.client import Client, decode
from haojie_training.native.execution import InferenceStats, PolicyInference
from haojie_training.runtime import Trainer, read_checkpoint


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

    def test_cpu_buffer_reuse_and_shared_statistics(self):
        model = PolicyValueNet(ModelConfig.tiny()).eval()
        stats = InferenceStats()
        runners = [PolicyInference(model, "cpu", stats=stats) for _ in range(2)]
        saved = []
        for index, sizes in enumerate(((4, 4), (4, 7, 4), (3,))):
            rows = [
                {
                    k: v[0]
                    for k, v in synthetic_batch(model.config, 1, n, 5).items()
                    if k in INPUT_KEYS
                }
                for n in sizes
            ]
            original = [{k: v.clone() for k, v in row.items()} for row in rows]
            expected = collate_inputs(rows, model.config)
            actual = runners[index % 2].batch(rows)
            for key in INPUT_KEYS:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            logits = runners[index % 2](rows)
            with torch.inference_mode():
                torch.testing.assert_close(logits, model.policy_logits(expected), rtol=0, atol=0)
            saved.append((logits, logits.clone()))
            for row, before in zip(rows, original):
                for key in INPUT_KEYS:
                    torch.testing.assert_close(row[key], before[key], rtol=0, atol=0)
        for value, before in saved:
            torch.testing.assert_close(value, before, rtol=0, atol=0)
        self.assertEqual(stats.snapshot()["inference_requests"], 6)
        self.assertEqual(stats.snapshot()["forwards"], 3)
        self.assertEqual(stats.snapshot()["mean_batch"], 2)

    def test_input_reference_ownership_and_stale_rejection(self):
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
        data = b"".join(batch[k][0].numpy().tobytes() for k in keys)
        infer = {
            "type": "infer",
            "id": 1,
            "model": "a" * 64,
            "actor": 1,
            "commandIndex": 0,
            "entities": 4,
            "candidates": 5,
            "bytes": len(data),
        }
        example = {"type": "example", "inputRef": 1, "model": "a" * 64, "actor": 1, "index": 0}

        def read(messages, enabled=True):
            client = Client.__new__(Client)
            client._input_refs = enabled
            client.read_seconds = client.decode_seconds = client.tensor_bytes = 0
            client.frame_limit = len(data)
            client.process = SimpleNamespace(
                stdout=io.BytesIO(
                    json.dumps(infer).encode()
                    + b"\n"
                    + data
                    + b"".join(json.dumps(message).encode() + b"\n" for message in messages)
                )
            )
            values = []
            client._publish = values.append
            client._read()
            self.assertEqual(client.tensor_bytes, len(data))
            return values

        result = read([example])
        self.assertEqual(len(result), 3)
        for key in INPUT_KEYS:
            self.assertEqual(result[0]["input"][key].data_ptr(), result[1]["input"][key].data_ptr())
        for change in (
            {"inputRef": 2},
            {"inputRef": True},
            {"inputRef": -1},
            {"index": 1},
            {"actor": 2},
            {"model": "b" * 64},
            {"bytes": 0},
        ):
            with self.subTest(change=change):
                self.assertIsInstance(read([{**example, **change}])[-1], ValueError)
        self.assertIsInstance(read([example], enabled=False)[-1], ValueError)
        self.assertIsInstance(read([example, example])[-1], ValueError)
        self.assertIsInstance(read([{"type": "done"}, example])[-1], ValueError)

    def test_complete_and_exported_checkpoint_readers(self):
        trainer = Trainer(PolicyValueNet(ModelConfig.tiny()), torch.device("cpu"))
        self.assertEqual(trainer.last_losses, {})
        trainer.step(synthetic_batch(trainer.model.config, 2, 4, 5))
        self.assertEqual(set(trainer.last_losses), {"policy", "value"})
        self.assertTrue(trainer.health()["finite_optimizer"])
        payload = trainer.checkpoint({"test": True})
        exported = {k: payload[k] for k in ("config", "model", "metadata", "network")}
        exported["format"] = "haojie-stream-weights-v1"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            for value in (
                payload,
                {"format": "haojie-stream-recovery-v2", "trainer": payload},
                exported,
            ):
                torch.save(value, path)
                config, actual = read_checkpoint(path)
                self.assertEqual(config, trainer.model.config)
                for key, tensor in payload["model"].items():
                    torch.testing.assert_close(actual["model"][key], tensor, rtol=0, atol=0)
            for network in (None, "entity-transformer-v2"):
                old = {**exported, "network": network}
                if network is None:
                    del old["network"]
                torch.save(old, path)
                with self.assertRaisesRegex(ValueError, "网络版本不兼容"):
                    read_checkpoint(path)

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
