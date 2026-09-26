"""结构消融的真实前向边界：关系重编号、候选等变、填充遮罩与输入不变。"""

import sys
import unittest
from pathlib import Path

import torch
from haojie_training.data import synthetic_batch
from haojie_training.model import ModelConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.training.improvement.model import (  # noqa: E402
    ImprovementNet,
    expanded_fields,
    relation_edges,
)
from scripts.training.improvement.value.train import features  # noqa: E402
from scripts.training.search.recovery.spatial import SpatialPolicyNet  # noqa: E402


class ImprovementTests(unittest.TestCase):
    def test_spatial_baseline_and_new_relationship_boundaries(self):
        torch.set_num_threads(2)
        config = ModelConfig.tiny()
        batch = synthetic_batch(config, 1, 4, 3)
        batch["entities"].zero_()
        batch["entities"][0, :, 0] = torch.tensor([2, 2, 7, 0]) / 32
        batch["entities"][0, :, 2] = torch.tensor([1, 2, 3, 0]) / 256
        batch["entities"][0, 2, 3] = 1 / 256
        batch["entities"][0, :2, 8:10] = torch.log1p(torch.tensor([[3.0, 4.0], [6.0, 9.0]])) / 8
        batch["entity_mask"][0] = torch.tensor([True, True, True, False])
        batch["sources"].fill_(-1)
        batch["targets"].fill_(-1)
        batch["candidate_mask"].fill_(True)
        batch["candidates"][..., 56:58].zero_()
        batch["candidates"][0, :, 36:38] = torch.tensor(
            [[3 / 9, 4 / 13], [6 / 9, 9 / 13], [5 / 9, 7 / 13]]
        )
        torch.manual_seed(17)
        old = SpatialPolicyNet(config).eval()
        torch.manual_seed(17)
        base = ImprovementNet(config, "spatial").eval()
        for a, b in zip(old(batch), base(batch)):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        model = ImprovementNet(config, "combined").eval()
        before = {k: v.clone() for k, v in batch.items()}
        expected = model(batch)
        for key in batch:
            torch.testing.assert_close(before[key], batch[key])
        batch["entities"][0, 3] = 100
        for a, b in zip(expected, model(batch)):
            torch.testing.assert_close(a, b)
        batch = before
        edges = relation_edges(batch)
        refs = batch["entities"][..., 2:7]
        refs[refs > 0] = refs[refs > 0] * 3 + 7 / 256
        torch.testing.assert_close(edges, relation_edges(batch))
        for a, b in zip(expected, model(batch)):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        order = torch.tensor([2, 0, 1])
        for key in ("candidates", "sources", "targets", "candidate_mask"):
            batch[key] = batch[key][:, order]
        torch.testing.assert_close(model(batch)[0], expected[0][:, order], atol=2e-6, rtol=2e-5)

    def test_presence_bits_remain_independent_and_padding_safe(self):
        batch = synthetic_batch(ModelConfig.tiny(), 1, 2, 1)
        batch["entities"].zero_()
        batch["entity_mask"][0] = torch.tensor([True, False])
        batch["entities"][0, 0, 60] = (1 + 4096) / 8191
        batch["entities"][0, 1] = 100
        result = expanded_fields(batch)
        self.assertEqual(result.shape[-1], 112)
        self.assertEqual(result[0, 0, :52].sum(), 2)
        self.assertEqual(result[0, 0, 0], 1)
        self.assertEqual(result[0, 0, 12], 1)
        self.assertTrue(result.isfinite().all())

    def test_frozen_representation_fits_value_without_changing_policy(self):
        torch.set_num_threads(2)
        config = ModelConfig.tiny()
        model = ImprovementNet(config, "spatial").eval()
        batch = synthetic_batch(config, 4, 3, 2)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        original_policy, original_value = model(batch)
        encoded = features(model, batch).clone()
        torch.testing.assert_close(model.value(encoded).squeeze(-1), original_value)
        for parameter in model.value.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(model.value.parameters(), lr=0.01)
        loss = (model.value(encoded).squeeze(-1) - 1).square().mean()
        loss.backward()
        optimizer.step()
        policy, value = model(batch)
        torch.testing.assert_close(original_policy, policy, atol=0, rtol=0)
        self.assertFalse(torch.equal(original_value, value))


if __name__ == "__main__":
    unittest.main()
