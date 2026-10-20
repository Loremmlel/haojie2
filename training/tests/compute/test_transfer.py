"""状态传回 CPU 后即可安全哈希和保存，保留容器元数据与恢复后的优化器行为。"""

import unittest
from unittest.mock import patch

import torch

from haojie_training.console.controller import identity
from haojie_training.data import synthetic_batch
from haojie_training.model import ModelConfig, PolicyValueNet
from haojie_training.runtime import Trainer, cpu_state


class TransferTests(unittest.TestCase):
    def test_cpu_state_preserves_model_metadata_and_does_not_modify_containers(self):
        model = PolicyValueNet(ModelConfig.tiny())
        original = {"model": model.state_dict(), "items": [torch.tensor(7)], "pair": (3, None)}
        actual = cpu_state(original)
        self.assertIsNot(actual, original)
        self.assertIsNot(actual["items"], original["items"])
        self.assertEqual(actual["model"]._metadata, original["model"]._metadata)
        self.assertEqual(actual["pair"], original["pair"])
        for key, tensor in original["model"].items():
            self.assertEqual(actual["model"][key].data_ptr(), tensor.data_ptr())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA 状态传回测试需要 GPU")
    def test_cuda_checkpoint_and_identity_wait_once_and_restore_exactly(self):
        torch.set_num_threads(1)
        config = ModelConfig.tiny()
        torch.manual_seed(61020)
        trainer = Trainer(PolicyValueNet(config), torch.device("cuda"))
        batch = {k: v.cuda() for k, v in synthetic_batch(config, 2, 9, 5).items()}
        trainer.step(batch)
        expected = {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()}
        with patch("haojie_training.runtime.synchronize", wraps=torch.cuda.synchronize) as wait:
            saved = trainer.checkpoint({})
        self.assertEqual(wait.call_count, 1)
        self.assertEqual(saved["model"]._metadata, trainer.model.state_dict()._metadata)
        for key, value in saved["model"].items():
            torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
        cpu_model = PolicyValueNet(config)
        cpu_model.load_state_dict(saved["model"])
        self.assertEqual(identity(trainer.model), identity(cpu_model))
        restored = Trainer(PolicyValueNet(config), torch.device("cuda"))
        restored.restore_payload(saved, {})
        trainer.step(batch)
        restored.step(batch)
        for a, b in zip(trainer.model.parameters(), restored.model.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
