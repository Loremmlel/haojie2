"""历史保留逐条可读上下文；完整、稀疏读出及已选候选的输出和梯度共用同一网络。"""

import copy
import unittest

import torch

from haojie_training.data import INPUT_KEYS, collate_inputs, synthetic_batch
from haojie_training.model import ModelConfig, PolicyValueNet, active_tokens
from haojie_training.native.execution import PolicyInference


def history_input(batch):
    """显式覆盖亡者、日志和普通实体；随机浮点角色通常不会触发亡者路径。"""
    batch["entities"][..., 0] = 2 / 32
    batch["kinds"].fill_(1)
    batch["entities"][:, 16:48, 0] = 5 / 32
    batch["kinds"][:, 48:80] = 250
    batch["entities"][:, 48:80, 0] = torch.arange(batch["kinds"][:, 48:80].shape[1]) / 4096
    batch["entities"][:, 8, 0] = 5 / 32
    batch["kinds"][:, 9] = 250
    batch["entity_mask"].fill_(True)
    batch["entity_mask"][:, 8:13] = False
    batch["sources"].fill_(-1)
    batch["targets"].fill_(0)
    batch["targets"][:, 0] = 16
    # 遮罩候选的历史指针不能改变其他有效候选的上下文。
    batch["candidate_mask"][:, -1] = False
    batch["targets"][:, -1] = 17
    batch["policy"].zero_()
    batch["policy"][:, 0] = 1
    return batch


class ReadoutTests(unittest.TestCase):
    def assert_gradient_close(self, actual, expected, name=""):
        # BF16 的重复引用归并顺序不同；逐层检查梯度误差范数，保留全部梯度与方向约束。
        error = float(torch.linalg.vector_norm(actual - expected))
        reference = float(torch.linalg.vector_norm(expected))
        self.assertLessEqual(error, reference * 0.04 + 1e-7, name)

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_sparse_inference_matches_full_with_holes_global_and_duplicate_pointers(self):
        torch.manual_seed(321)
        for config in (ModelConfig(width=64, layers=4, heads=4, ffn=256), ModelConfig()):
            model = PolicyValueNet(config).eval()
            runner = PolicyInference(model, "cpu")
            rows = []
            for n, actions in ((79, 3), (147, 11), (98, 8)):
                batch = history_input(synthetic_batch(config, 1, n, actions))
                rows.append({k: v[0] for k, v in batch.items() if k in INPUT_KEYS})
            original = [{k: v.clone() for k, v in row.items()} for row in rows]
            full = collate_inputs(rows, config)
            positions, sources, targets = runner.readout(rows, full)
            self.assertEqual(positions.shape, (3, 4))
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
        for config in (ModelConfig(width=64, layers=4, heads=4, ffn=256), ModelConfig()):
            dense = PolicyValueNet(config)
            sparse = copy.deepcopy(dense)
            batch = history_input(synthetic_batch(config, 3, 97, 41))
            batch["sources"][0].fill_(-1)
            batch["targets"][0].fill_(-1)
            batch["kinds"][1, 48:56] = 1
            # 未被选中但仍有效的候选，其旧日志引用也必须在完整主干中提升。
            batch["targets"][1:, 1] = 56
            selected = batch["policy"].argmax(-1)
            logits, value = dense(batch)
            chosen = logits.gather(1, selected[:, None]).squeeze(1)
            other_chosen, other_value = sparse.selected_value(batch, selected)
            torch.testing.assert_close(chosen, other_chosen, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(value, other_value, atol=2e-6, rtol=2e-5)
            (chosen.tanh().square().sum() + value.square().sum()).backward()
            (other_chosen.tanh().square().sum() + other_value.square().sum()).backward()
            for (name, a), (_, b) in zip(dense.named_parameters(), sparse.named_parameters()):
                self.assert_gradient_close(a.grad, b.grad, name)
            self.assertEqual(list(dense.state_dict()), list(sparse.state_dict()))

    def test_history_context_retains_information_gradients_and_mask_boundaries(self):
        torch.manual_seed(140)
        config = ModelConfig(width=64, layers=4, heads=4, ffn=256)
        model = PolicyValueNet(config).eval()
        batch = history_input(synthetic_batch(config, 1, 80, 4))
        batch["candidate_mask"][0, 2] = False
        batch["targets"][0, 2] = 8
        batch["sources"][0, 1] = 49
        batch["entity_mask"][0, [77, 79]] = False
        # kind 决定日志身份；同样的第零列不能把近期日志误当作亡者冻结。
        batch["entities"][0, 48, 0] = 5 / 32
        batch["entities"][0, 64, 0] = 5 / 32
        active = active_tokens(batch)
        self.assertTrue(active[0, [0, 1, 17, 50]].all())
        self.assertFalse(active[0, [9, 10, 18, 19, 49, 62, 78, 80]].any())
        # 最新十六条有效日志全部活跃，遮罩洞不消耗窗口；较早候选引用另行提升。
        self.assertTrue(active[0, 63:78].all())
        self.assertTrue(active[0, 79])
        batch["entities"].requires_grad_()
        original, original_value = model(batch)
        original[0, 0].backward()
        masked_pointers = {key: value.detach().clone() for key, value in batch.items()}
        for key, entity in (("sources", 18), ("targets", 48)):
            masked_pointers[key][~batch["candidate_mask"]] = entity
        torch.testing.assert_close(active_tokens(masked_pointers), active, atol=0, rtol=0)
        with torch.inference_mode():
            changed_logits, changed_value = model(masked_pointers)
            torch.testing.assert_close(changed_logits, original, atol=0, rtol=0)
            torch.testing.assert_close(changed_value, original_value, atol=0, rtol=0)
        for entity in (18, 48):
            self.assertGreater(batch["entities"].grad[0, entity].abs().sum().item(), 0)
            changed = {key: value.detach().clone() for key, value in batch.items()}
            changed["entities"][0, entity, 8 if entity == 18 else 3] += 1
            with torch.inference_mode():
                self.assertFalse(torch.equal(model.policy_logits(changed), original.detach()))
        self.assertEqual(batch["entities"].grad[:, 8:13].abs().sum().item(), 0)
        self.assertEqual(batch["entities"].grad[:, [77, 79]].abs().sum().item(), 0)
        masked = {key: value.detach().clone() for key, value in batch.items()}
        masked["entities"][:, 8:13] += 10
        masked["kinds"][:, 8:13] = 249
        masked["entities"][:, [77, 79]] += 10
        masked["kinds"][:, [77, 79]] = 249
        with torch.inference_mode():
            torch.testing.assert_close(model.policy_logits(masked), original, atol=0, rtol=0)

        # 无历史时对照原来的全量图，避免普通局面意外改变计算语义。
        plain = {key: value.detach().clone() for key, value in batch.items()}
        plain["entities"][..., 0] = 2 / 32
        plain["kinds"].fill_(1)
        plain["entity_mask"].fill_(True)
        with torch.inference_mode():
            entities = model.entity_projection(plain["entities"]) + model.kind_embedding(
                plain["kinds"]
            )
            x = torch.cat((model.global_projection(plain["globals"]).unsqueeze(1), entities), 1)
            allowed = torch.ones(1, 1, 1, x.shape[1], dtype=torch.bool)
            for block in model.blocks:
                x = block(x, allowed)
            expected = model.score_policy(plain, model.norm(x))
            torch.testing.assert_close(model.policy_logits(plain), expected, atol=0, rtol=0)

    def test_extended_model_keeps_full_context_path(self):
        class ExtendedPolicy(PolicyValueNet):
            pass

        model = ExtendedPolicy(ModelConfig.tiny()).eval()
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
        config = ModelConfig(width=64, layers=4, heads=4, ffn=256)
        torch.manual_seed(811)
        model = PolicyValueNet(config).to("cuda")
        batch = history_input(synthetic_batch(config, 2, 97, 11))
        rows = [{k: v[i] for k, v in batch.items() if k in INPUT_KEYS} for i in range(2)]
        device_batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.inference_mode():
            expected = model.policy_logits(device_batch).cpu()
        runner = PolicyInference(model, "cuda")
        torch.testing.assert_close(runner(rows), expected, atol=3e-3, rtol=3e-3)
        dense = copy.deepcopy(model)
        selected = device_batch["policy"].argmax(-1)
        logits, values = dense(device_batch)
        chosen = logits.gather(1, selected[:, None]).squeeze(1)
        actual, actual_values = model.selected_value(device_batch, selected)
        (chosen.tanh().square().sum() + values.square().sum()).backward()
        (actual.tanh().square().sum() + actual_values.square().sum()).backward()
        for a, b in zip(dense.parameters(), model.parameters()):
            self.assert_gradient_close(a.grad, b.grad)
