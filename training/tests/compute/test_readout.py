"""稀疏末层只改变执行量；全实体注意力、候选评分、输入所有权和参数梯度对照。"""

import copy
import unittest

import torch

from haojie_training.data import INPUT_KEYS, collate_inputs, synthetic_batch
from haojie_training.model import ModelConfig, PolicyValueNet
from haojie_training.native.execution import PolicyInference
from haojie_training.research.spatial import SpatialPolicyNet


class ReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_sparse_inference_matches_full_with_holes_global_and_duplicate_pointers(self):
        torch.manual_seed(321)
        for config in (ModelConfig.tiny(), ModelConfig()):
            model = PolicyValueNet(config).eval()
            runner = PolicyInference(model, "cpu")
            rows = []
            for n, actions in ((79, 3), (147, 11), (98, 8)):
                batch = synthetic_batch(config, 1, n, actions)
                batch["sources"].fill_(-1)
                batch["targets"].fill_(0)
                batch["entity_mask"][:, 8:13] = False
                batch["candidate_mask"][:, -1] = False
                rows.append({k: v[0] for k, v in batch.items() if k in INPUT_KEYS})
            original = [{k: v.clone() for k, v in row.items()} for row in rows]
            full = collate_inputs(rows, config)
            positions, sources, targets = runner.readout(rows, full)
            self.assertEqual(positions.shape, (3, 2))
            with torch.inference_mode():
                expected = model.policy_logits(full)
                actual = runner(rows)
            torch.testing.assert_close(expected, actual, atol=2e-6, rtol=2e-5)
            for row, prior in zip(rows, original):
                for key in INPUT_KEYS:
                    torch.testing.assert_close(row[key], prior[key], atol=0, rtol=0)
            snapshot = actual.clone()
            runner(rows[:1])
            torch.testing.assert_close(actual, snapshot, atol=0, rtol=0)
            self.assertTrue((sources >= 0).all() and (targets >= 0).all())

    def test_mc_selected_outputs_and_all_gradients_match_dense(self):
        torch.manual_seed(79)
        for config in (ModelConfig.tiny(), ModelConfig()):
            dense = PolicyValueNet(config)
            sparse = copy.deepcopy(dense)
            batch = synthetic_batch(config, 3, 97, 41)
            batch["sources"][0].fill_(-1)
            batch["targets"][0].fill_(-1)
            selected = batch["policy"].argmax(-1)
            logits, value = dense(batch)
            chosen = logits.gather(1, selected[:, None]).squeeze(1)
            other_chosen, other_value = sparse.selected_value(batch, selected)
            torch.testing.assert_close(chosen, other_chosen, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(value, other_value, atol=2e-6, rtol=2e-5)
            (chosen.tanh().square().sum() + value.square().sum()).backward()
            (other_chosen.tanh().square().sum() + other_value.square().sum()).backward()
            for (name, a), (_, b) in zip(dense.named_parameters(), sparse.named_parameters()):
                torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=2e-4, msg=name)
            self.assertEqual(list(dense.state_dict()), list(sparse.state_dict()))

    def test_research_model_keeps_full_context_path(self):
        model = SpatialPolicyNet(ModelConfig.tiny()).eval()
        batch = synthetic_batch(model.config, 2, 7, 4)
        selected = batch["policy"].argmax(-1)
        with torch.inference_mode():
            logits, value = model(batch)
            chosen, other_value = model.selected_value(batch, selected)
        torch.testing.assert_close(
            chosen, logits.gather(1, selected[:, None]).squeeze(1), atol=0, rtol=0
        )
        torch.testing.assert_close(other_value, value, atol=0, rtol=0)
        rows = [{k: v[i] for k, v in batch.items() if k in INPUT_KEYS} for i in range(2)]
        self.assertIsNone(PolicyInference(model, "cpu").readout(rows, batch))

    def test_dense_fallback_when_most_entities_are_referenced(self):
        model = PolicyValueNet(ModelConfig.tiny()).eval()
        runner = PolicyInference(model, "cpu")
        batch = synthetic_batch(model.config, 1, 40, 40)
        batch["sources"][0] = torch.arange(40)
        rows = [{k: v[0] for k, v in batch.items() if k in INPUT_KEYS}]
        self.assertIsNone(runner.readout(rows, batch))
        with torch.inference_mode():
            torch.testing.assert_close(runner(rows), model.policy_logits(batch), atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA 稀疏读取专项需要 GPU 执行器")
    def test_cuda_sparse_forward_and_selected_gradients(self):
        config = ModelConfig.tiny()
        torch.manual_seed(811)
        model = PolicyValueNet(config).to("cuda")
        batch = synthetic_batch(config, 2, 97, 11)
        rows = [{k: v[i] for k, v in batch.items() if k in INPUT_KEYS} for i in range(2)]
        device_batch = {k: v.to("cuda") for k, v in batch.items()}
        precisions = ["fp32"] + (["bf16"] if torch.cuda.is_bf16_supported() else [])
        for precision in precisions:
            from haojie_training.runtime import autocast

            with torch.inference_mode(), autocast(torch.device("cuda"), precision):
                expected = model.policy_logits(device_batch).cpu()
            runner = PolicyInference(model, "cuda", precision)
            torch.testing.assert_close(runner(rows), expected, atol=3e-3, rtol=3e-3)
        dense = copy.deepcopy(model)
        selected = device_batch["policy"].argmax(-1)
        logits, values = dense(device_batch)
        chosen = logits.gather(1, selected[:, None]).squeeze(1)
        actual, actual_values = model.selected_value(device_batch, selected)
        (chosen.tanh().square().sum() + values.square().sum()).backward()
        (actual.tanh().square().sum() + actual_values.square().sum()).backward()
        for a, b in zip(dense.parameters(), model.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=3e-5, rtol=5e-3)
