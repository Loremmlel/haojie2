"""长度分组不重抽样；每步终局归一化、梯度累积、优化器计数和恢复语义保持一致。"""

import copy
import unittest

import torch

from haojie_training.batching.stream import update_groups
from haojie_training.data import collate_examples, synthetic_batch
from haojie_training.model import ModelConfig, PolicyValueNet
from haojie_training.runtime import Trainer


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_partition_preserves_multiplicity_and_skips_low_padding(self):
        config = ModelConfig.tiny()
        rows = [{"entities": torch.empty(n, 64)} for n in (5, 7, 100, 102)]
        rows.append(rows[0])
        groups = update_groups(rows, config)
        self.assertGreater(len(groups), 1)
        self.assertEqual(sorted(map(id, rows)), sorted(id(r) for g in groups for r in g))
        self.assertEqual(update_groups(rows[2:4], config), [rows[2:4]])
        self.assertEqual(update_groups(rows[:1], config), [rows[:1]])

    def test_grouped_loss_gradients_and_restore_are_one_logical_step(self):
        torch.manual_seed(29)
        config = ModelConfig.tiny()
        model = PolicyValueNet(config)
        rows = []
        for n in (9, 15, 160, 190):
            batch = synthetic_batch(config, 1, n, 23)
            rows.append({k: v[0] for k, v in batch.items()})
        # 第一组无终局标签，第二组有标签；不能简单平均两组损失。
        rows[0]["value_mask"].fill_(False)
        rows[1]["value_mask"].fill_(False)
        for row in rows[2:]:
            row["value_mask"].fill_(True)
        dense = Trainer(model, torch.device("cpu"), objective="decomposed-mc-q-v2")
        grouped = Trainer(copy.deepcopy(model), torch.device("cpu"), objective=dense.objective)
        batches = [collate_examples(g, config) for g in update_groups(rows, config)]
        loss = dense.step(collate_examples(rows, config))
        actual = grouped.step_batches(batches)
        torch.testing.assert_close(actual, loss, atol=2e-6, rtol=2e-5)
        self.assertEqual(grouped.updates, 1)
        self.assertEqual(grouped.steps, 1)
        for (name, a), (_, b) in zip(
            dense.model.named_parameters(), grouped.model.named_parameters()
        ):
            torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=2e-4, msg=name)
            torch.testing.assert_close(a, b, atol=3e-5, rtol=2e-4, msg=name)
        restored = Trainer(PolicyValueNet(config), torch.device("cpu"), objective=dense.objective)
        restored.restore_payload(copy.deepcopy(grouped.checkpoint({})), {})
        torch.testing.assert_close(
            grouped.step_batches(batches), restored.step_batches(batches), atol=0, rtol=0
        )
        for a, b in zip(grouped.model.parameters(), restored.model.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        self.assertEqual(grouped.updates, 2)

    def test_nonfinite_later_group_never_updates_parameters(self):
        trainer = Trainer(
            PolicyValueNet(ModelConfig.tiny()), torch.device("cpu"), objective="decomposed-mc-q-v2"
        )
        batch = synthetic_batch(trainer.model.config, 1, 5, 3)
        bad = {k: v.clone() for k, v in batch.items()}
        bad["value"].fill_(float("nan"))
        before = copy.deepcopy(trainer.model.state_dict())
        with self.assertRaises(FloatingPointError):
            trainer.step_batches([batch, bad])
        self.assertEqual(trainer.updates, 0)
        for key, value in before.items():
            torch.testing.assert_close(trainer.model.state_dict()[key], value, atol=0, rtol=0)
