"""空间试验只改变候选上下文，保持公开输入、填充遮罩及实验检查点隔离。"""

import sys
import tempfile
import unittest
from pathlib import Path

import torch

from haojie_training.data import synthetic_batch
from haojie_training.model import ModelConfig
from haojie_training.runtime import checkpoint_config

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/training/search/recovery"))
from spatial import SpatialPolicyNet  # noqa: E402


class RecoveryTests(unittest.TestCase):
    def test_spatial_candidates_keep_order_masks_and_input_immutable(self):
        torch.set_num_threads(2)
        torch.manual_seed(5)
        config = ModelConfig.tiny()
        model = SpatialPolicyNet(config).eval()
        batch = synthetic_batch(config, 1, 4, 3)
        batch["entities"].zero_()
        batch["entities"][0, :2, 0] = 2 / 32
        batch["entities"][0, :2, 8:10] = torch.log1p(torch.tensor([[3.0, 4.0], [6.0, 9.0]])) / 8
        batch["entity_mask"][0] = torch.tensor([True, True, False, False])
        batch["sources"].fill_(-1)
        batch["targets"].fill_(-1)
        batch["candidate_mask"].fill_(True)
        batch["candidates"][0, :, 36:38] = torch.tensor(
            [[3 / 9, 4 / 13], [6 / 9, 9 / 13], [5 / 9, 7 / 13]]
        )
        before = {k: v.clone() for k, v in batch.items()}
        with torch.inference_mode():
            logits, value = model(batch)
            for key in batch:
                torch.testing.assert_close(batch[key], before[key])
            batch["entities"][0, 2:] = 100
            changed, again = model(batch)
            torch.testing.assert_close(logits, changed)
            torch.testing.assert_close(value, again)
            order = torch.tensor([2, 0, 1])
            for key in ("candidates", "sources", "targets", "candidate_mask"):
                batch[key] = batch[key][:, order]
            permuted, _ = model(batch)
            torch.testing.assert_close(permuted, logits[:, order])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experimental.pt"
            torch.save({"format": "haojie-recovery-policy-pilot-v1"}, path)
            with self.assertRaisesRegex(ValueError, "网络版本不兼容"):
                checkpoint_config(path)


if __name__ == "__main__":
    unittest.main()
