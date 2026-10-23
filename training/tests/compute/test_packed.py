"""紧凑推理的序列边界、指针映射、参数更新与回退行为。"""

import os
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch
from torch.nn import functional as F

from haojie_training.data import INPUT_KEYS, collate_inputs, synthetic_batch, validate_inputs
from haojie_training.model import RECENT_JOURNAL, ModelConfig, PolicyValueNet
from haojie_training.native.execution import InferenceStats, PolicyInference
from haojie_training.native.packed import ENCODER_KEYS, POLICY_KEYS, PackedPolicy, supports_packed


def examples(config):
    rows = []
    for i, (entities, candidates) in enumerate(((31, 6), (7, 3), (1, 1))):
        batch = synthetic_batch(config, 1, entities, candidates, seed=17 + i)
        row = {key: value[0] for key, value in batch.items() if key in INPUT_KEYS}
        row["entity_mask"].fill_(True)
        row["candidate_mask"].fill_(True)
        row["sources"].fill_(-1)
        row["targets"].fill_(entities - 1)
        row["entities"][:, 0] = 2 / 32
        row["kinds"].clamp_max_(127)
        if i == 0:
            row["entities"][[1, 2, 3, 9, 12], 0] = 5 / 32
            row["kinds"][[4, 5]] = 250
            row["kinds"][13:] = 250
            # 日志第零列不是角色；被屏蔽候选引用的有效亡者仍保持冻结。
            row["entities"][[4, 14], 0] = 5 / 32
            # 近期窗口只数有效日志；中间孔洞不能挤掉第16条有效记录。
            row["entity_mask"][17] = False
            row["targets"].fill_(12)
            row["sources"][-1] = 9
            row["candidate_mask"][-1] = False
        elif i == 1:
            row["entity_mask"][[1, 3]] = False
            row["entities"][[2, 4, 6], 0] = 5 / 32
            row["kinds"][[0, 5]] = 250
            row["sources"][0] = 4
            row["sources"][-1] = 1
            row["candidate_mask"][-1] = False
        elif i == 2:
            row["entity_mask"].fill_(False)
            row["targets"].fill_(-1)
        rows.append(row)
    return rows


def reference_attention(q, k, v, batch, layout, query):
    """CPU 小测试按累积长度切开注意力，检查生产打包结果，不提供生产替代后端。"""
    qstarts = batch[f"cu_{query}"].tolist()
    kstarts = batch["cu_k"].tolist()
    result = []
    for qa, qb, ka, kb in zip(qstarts, qstarts[1:], kstarts, kstarts[1:]):
        result.append(
            F.scaled_dot_product_attention(
                q[qa:qb].transpose(0, 1).unsqueeze(0),
                k[ka:kb].transpose(0, 1).unsqueeze(0),
                v[ka:kb].transpose(0, 1).unsqueeze(0),
                dropout_p=0.0,
            )
            .squeeze(0)
            .transpose(0, 1)
        )
    return torch.cat(result)


class PackedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @unittest.skipUnless(os.environ.get("HAOJIE_ATEN") == "1", "需要已构建 ATen C++ 执行器")
    def test_aten_cpu_collation_matches_reference_without_cuda(self):
        from haojie_training.native.aten import operator
        from haojie_training.native.client import decode

        operator()
        model = PolicyValueNet().eval()
        rows = examples(model.config)
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
        arrays = [
            decode(
                b"".join(row[key].view(torch.uint8).numpy().tobytes() for key in keys),
                len(row["entities"]),
                len(row["sources"]),
                _arrays=True,
                _defer=True,
            )
            for row in rows
        ]
        _, tensors, values = torch.ops.haojie.pack(
            [row.address for row in arrays],
            [row.entity_count for row in arrays],
            [row.candidate_count for row in arrays],
            [],
            False,
            RECENT_JOURNAL,
        )
        actual = dict(zip(ENCODER_KEYS + POLICY_KEYS, tensors))
        reference = PackedPolicy(model, {}, False)
        reference.operator = operator()
        expected, layout = reference.batch(rows)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        self.assertEqual(actual["candidate_positions"].tolist(), [0, 1, 2, 3, 4, 6, 7, 12])
        self.assertEqual(
            values,
            [
                layout[key]
                for key in (
                    "max_k",
                    "max_q",
                    "max_active",
                    "size",
                    "entities",
                    "candidates",
                    "candidate_slots",
                )
            ],
        )

    def test_compaction_sequence_boundaries_weights_and_ownership(self):
        torch.manual_seed(73)
        model = PolicyValueNet(ModelConfig(width=64, layers=4, heads=4, ffn=256)).eval()
        rows = examples(model.config)
        saved = [{key: value.clone() for key, value in row.items()} for row in rows]
        dense = collate_inputs(rows, model.config)
        validate_inputs(dense, model.config)
        stats = InferenceStats()
        with patch("haojie_training.native.execution.supports_packed", return_value=True):
            runner = PolicyInference(model, "cpu", stats=stats)
        self.assertTrue(runner.supports_packed_batches)
        batch, layout = runner.packed.batch(rows)
        self.assertEqual(batch["cu_k"].tolist(), [0, 31, 37, 38])
        self.assertEqual(batch["cu_q"].tolist(), [0, 2, 5, 6])
        self.assertEqual(batch["readout"].tolist(), [0, 13, 31, 34, 36, 37])
        self.assertEqual(batch["cu_active"].tolist(), [0, 24, 29, 30])
        self.assertEqual(
            batch["active"].tolist(),
            [0, 1, 7, 8, 9, 11, 12, 13, *range(15, 31), 31, 32, 34, 35, 36, 37],
        )
        self.assertEqual(layout["max_active"], 24)
        self.assertEqual(layout["entities"], 35)
        self.assertEqual(batch["entities"].shape[0], 38)
        self.assertEqual(batch["cu_k"].dtype, torch.int32)
        with patch("haojie_training.native.packed._attention", side_effect=reference_attention):
            with torch.inference_mode():
                expected = model.policy_logits(dense)
            actual = runner(rows)
            torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.008)
            self.assertTrue(
                torch.equal(
                    actual.argsort(dim=-1, stable=True), expected.argsort(dim=-1, stable=True)
                )
            )
            retained = actual.clone()
            # 无效候选改指其他亡者或旧日志，不得改变有效候选的上下文和输出。
            masked = [{key: value.clone() for key, value in row.items()} for row in rows]
            masked[0]["sources"][-1], masked[0]["targets"][-1] = 1, 4
            masked[1]["sources"][-1], masked[1]["targets"][-1] = 2, 2
            masked_output = runner(masked)
            torch.testing.assert_close(
                masked_output[dense["candidate_mask"]],
                actual[dense["candidate_mask"]],
                atol=2e-6,
                rtol=2e-5,
            )
            # 更新原模块而非重建执行器；前向必须即时读取新参数。
            with torch.no_grad():
                model.policy[-1].weight.mul_(0.5)
            updated = runner(rows)
            with torch.inference_mode():
                expected = model.policy_logits(dense)
            torch.testing.assert_close(updated, expected, atol=0.001, rtol=0.008)
            self.assertFalse(torch.equal(updated, actual))
            runner(rows[:1])
        torch.testing.assert_close(actual, retained, atol=0, rtol=0)
        for row, before in zip(rows, saved):
            for key in INPUT_KEYS:
                torch.testing.assert_close(row[key], before[key], atol=0, rtol=0)
        # 普通及紧凑执行器共用统计，返回结果不依赖下一批主机容量。
        PolicyInference(model, "cpu", stats=stats)(rows[:1])
        self.assertEqual(stats.snapshot()["forwards"], 5)
        self.assertEqual(stats.snapshot()["packed_forwards"], 4)
        self.assertEqual(stats.snapshot()["inference_requests"], 11)

    def test_fallbacks_and_device_capability(self):
        model = PolicyValueNet(ModelConfig.tiny()).eval()
        self.assertFalse(PolicyInference(model, "cpu").supports_packed_batches)
        self.assertFalse(PolicyInference(model, "cpu").supports_async)
        self.assertFalse(supports_packed(model, torch.device("xpu")))
        self.assertFalse(supports_packed(model, torch.device("cuda")))

        class ExtendedPolicy(PolicyValueNet):
            pass

        research = ExtendedPolicy(model.config).eval()
        self.assertFalse(supports_packed(research, torch.device("cuda")))
        # 使用小配置代替正式配置检查版本和后端门槛，无须建立 CUDA 上下文。
        with (
            patch("haojie_training.native.packed.ModelConfig", return_value=model.config),
            patch("torch.backends.cuda.is_built", return_value=True),
            patch("torch.backends.cuda.mem_efficient_sdp_enabled", return_value=True),
            patch("torch.__version__", "2.14.0+cu130"),
            patch("torch.version.cuda", "13.0"),
            patch("torch.cuda.device", return_value=nullcontext()),
            patch("torch.backends.cuda.can_use_efficient_attention", return_value=True) as usable,
        ):
            self.assertTrue(supports_packed(model, torch.device("cuda")))
            query = usable.call_args.args[0].query
            self.assertEqual(
                query.shape, (1, model.config.heads, 1, model.config.width // model.config.heads)
            )
            self.assertEqual(query.data_ptr(), model.entity_projection.weight.data_ptr())
            self.assertEqual(query.dtype, torch.bfloat16)
            self.assertFalse(query.requires_grad)
            with patch("torch.backends.cuda.can_use_efficient_attention", return_value=False):
                self.assertFalse(supports_packed(model, torch.device("cuda")))
                self.assertFalse(PolicyInference(model, "cuda").supports_packed_batches)
            with patch("torch.version.cuda", None):
                self.assertFalse(supports_packed(model, torch.device("cuda")))
                self.assertFalse(PolicyInference(model, "cuda").supports_packed_batches)
            with patch("torch.__version__", "2.15.0"):
                self.assertFalse(supports_packed(model, torch.device("cuda")))
            with patch("torch.backends.cuda.mem_efficient_sdp_enabled", return_value=False):
                self.assertFalse(supports_packed(model, torch.device("cuda")))

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "CUDA BF16 紧凑推理需要支持的 GPU 执行器",
    )
    def test_cuda_bf16_packed_logits_update_and_timing(self):
        self._cuda_packed_logits_update_and_timing(3e-3, 3e-3)

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        and os.environ.get("HAOJIE_ATEN") == "1",
        "需要 CUDA BF16 和已构建 ATen C++ 执行器",
    )
    def test_fused_residual_norm_matches_reference_and_preserves_grad_fallback(self):
        from haojie_training.native.aten import operator
        from haojie_training.native.kernels import residual_norm

        operator()
        kernel, error = residual_norm(torch.cuda.current_device())
        self.assertIsNotNone(kernel, error)
        pointers = kernel[:2]
        stream = torch.cuda.current_stream().cuda_stream
        torch.manual_seed(719)
        weight, bias = (torch.randn(384, device="cuda", dtype=torch.bfloat16) for _ in range(2))
        for count in (1, 37, 8193):
            with torch.no_grad():
                left = torch.randn(count, 384, device="cuda", dtype=torch.bfloat16)
                right = torch.randn_like(left, dtype=torch.bfloat16)
                summed, normalized = torch.ops.haojie.add_norm(
                    left, right, weight, bias, 1e-5, *pointers, stream
                )
                expected = left + right
                torch.testing.assert_close(summed, expected, atol=0, rtol=0)
                torch.testing.assert_close(
                    normalized,
                    F.layer_norm(expected, (384,), weight, bias, 1e-5),
                    atol=0.016,
                    rtol=0.008,
                )
        left = torch.randn(3, 384, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        right = torch.randn(3, 384, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        _, actual = torch.ops.haojie.add_norm(left, right, weight, bias, 1e-5, *pointers, stream)
        expected = F.layer_norm(left + right, (384,), weight, bias, 1e-5)
        for actual_grad, expected_grad in zip(
            torch.autograd.grad(actual.square().sum(), (left, right)),
            torch.autograd.grad(expected.square().sum(), (left, right)),
            strict=True,
        ):
            torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)

    @unittest.skipUnless(
        torch.cuda.is_available() and os.environ.get("HAOJIE_ATEN") == "1",
        "需要已构建 ATen C++ 执行器",
    )
    def test_aten_encoder_shares_live_parameters_and_variable_shapes(self):
        torch.manual_seed(991)
        model = PolicyValueNet().cuda().eval()
        with torch.no_grad():
            for index, layer in enumerate(
                layer for layer in model.modules() if isinstance(layer, torch.nn.LayerNorm)
            ):
                # 各层使用不同仿射参数和 epsilon，检查原生路径没有错用归一化参数。
                layer.weight.mul_(1 + index * 0.02)
                layer.bias.fill_(index * 0.003)
                layer.eps *= index + 1
        keys = list(model.state_dict())
        rows = examples(model.config)
        modes = [(False, False), (False, True)]
        if torch.backends.cuda.is_flash_attention_available():
            modes.extend(((True, False), (True, True)))
        for flash, fused in modes:
            atol, rtol = 3e-3, 3e-3
            runner = PolicyInference(model, "cuda")
            self.assertIsNotNone(runner.packed.operator)
            runner.packed.flash = flash
            runner.packed.fused = fused
            for group in (rows, rows[:1], rows[1:]):
                batch = {k: v.cuda() for k, v in collate_inputs(group, model.config).items()}
                with torch.no_grad():
                    expected = model.policy_logits(batch).cpu()
                torch.testing.assert_close(runner(group), expected, atol=atol, rtol=rtol)
            self.assertEqual(
                runner.packed.weights[7].data_ptr(), model.blocks[0].qkv.weight.data_ptr()
            )
            with torch.no_grad():
                model.blocks[0].qkv.weight.mul_(0.75)
            batch = {k: v.cuda() for k, v in collate_inputs(rows, model.config).items()}
            with torch.no_grad():
                expected = model.policy_logits(batch).cpu()
            torch.testing.assert_close(runner(rows), expected, atol=atol, rtol=rtol)
        self.assertEqual(list(model.state_dict()), keys)

    @unittest.skipUnless(
        torch.cuda.is_available() and os.environ.get("HAOJIE_ATEN") == "1",
        "需要已构建 ATen C++ 执行器",
    )
    def test_aten_collation_matches_reference_and_owns_copied_frames(self):
        from haojie_training.native.client import decode

        model = PolicyValueNet().cuda().eval()
        runner = PolicyInference(model, "cuda")
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
        rows = examples(model.config)
        for group in (rows, rows[:1], rows[1:]):
            frames = [
                bytearray(b"".join(row[key].view(torch.uint8).numpy().tobytes() for key in keys))
                for row in group
            ]
            arrays = [
                decode(
                    frame,
                    len(row["entities"]),
                    len(row["sources"]),
                    _owned=True,
                    _arrays=True,
                    _defer=True,
                )
                for frame, row in zip(frames, group)
            ]
            self.assertTrue(all(not row.validated for row in arrays))
            self.assertTrue(all(row._values is None for row in arrays))
            expected, layout = runner.packed.batch(group)
            expected = {key: value.clone() for key, value in expected.items()}
            actual, actual_layout = runner.packed.batch(arrays)
            self.assertTrue(all(row.validated for row in arrays))
            self.assertTrue(all(row._values is None for row in arrays))
            self.assertEqual(actual_layout, layout)
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
            snapshot = actual["entities"].clone()
            for frame in frames:
                frame[:] = bytes(len(frame))
            torch.testing.assert_close(actual["entities"], snapshot, atol=0, rtol=0)
        frame = bytearray(
            b"".join(rows[0][key].view(torch.uint8).numpy().tobytes() for key in keys)
        )
        malformed = decode(
            frame,
            len(rows[0]["entities"]),
            len(rows[0]["sources"]),
            _owned=True,
            _arrays=True,
            _defer=True,
        )
        malformed["sources"][0] = len(rows[0]["entities"])
        with self.assertRaisesRegex(RuntimeError, "invalid packed frame values"):
            runner.packed.batch([malformed])
        self.assertFalse(malformed.validated)

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        and os.environ.get("HAOJIE_ATEN") == "1",
        "需要 CUDA BF16 和已构建 ATen C++ 执行器",
    )
    def test_factored_policy_preserves_full_candidates_and_live_weights(self):
        torch.manual_seed(927)
        model = PolicyValueNet().cuda().eval()
        rows = []
        for entities, candidates in ((37, 512), (67, 41)):
            batch = synthetic_batch(model.config, 1, entities, candidates, seed=entities)
            row = {key: value[0] for key, value in batch.items() if key in INPUT_KEYS}
            row["entity_mask"].fill_(True)
            row["candidate_mask"].fill_(True)
            row["sources"].fill_(-1)
            row["targets"].fill_(entities - 1)
            row["candidate_mask"][-1] = False
            rows.append(row)
        runner = PolicyInference(model, "cuda")
        batch = {key: value.cuda() for key, value in collate_inputs(rows, model.config).items()}
        for updated in (False, True):
            if updated:
                with torch.no_grad():
                    model.policy[0].weight.mul_(0.75)
                    model.policy[0].bias.add_(0.1)
            with torch.no_grad():
                expected = model.policy_logits(batch).cpu()
            for factored, fused in ((False, False), (False, True), (True, True)):
                runner.packed.factored = factored
                runner.packed.fused = fused
                actual = runner(rows)
                torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)
                self.assertEqual(actual.shape, (2, 512))
                self.assertTrue(
                    torch.equal(
                        actual[~batch["candidate_mask"].cpu()],
                        expected[~batch["candidate_mask"].cpu()],
                    )
                )

    def _cuda_packed_logits_update_and_timing(self, atol, rtol):
        torch.manual_seed(73)
        model = PolicyValueNet().cuda().eval()
        runner = PolicyInference(model, "cuda")
        if not runner.supports_packed_batches:
            self.skipTest("本 Torch/CUDA 后端沿用普通推理")
        rows = examples(model.config)
        batch = {key: value.cuda() for key, value in collate_inputs(rows, model.config).items()}
        with torch.inference_mode():
            expected = model.policy_logits(batch).cpu()
        dtypes = []
        handle = model.blocks[0].qkv.register_forward_hook(
            lambda _module, _inputs, output: dtypes.append(output.dtype)
        )
        head_handle = model.policy[0].register_forward_hook(
            lambda _module, _inputs, output: dtypes.append(output.dtype)
        )
        try:
            actual = runner(rows)
        finally:
            handle.remove()
            head_handle.remove()
        self.assertEqual(dtypes, [torch.bfloat16] * 2)
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        self.assertTrue(
            torch.equal(actual.argsort(dim=-1, stable=True), expected.argsort(dim=-1, stable=True))
        )
        retained = actual.clone()
        with torch.no_grad():
            model.policy[-1].weight.mul_(0.5)
        updated = runner(rows)
        with torch.inference_mode():
            expected = model.policy_logits(batch).cpu()
        torch.testing.assert_close(updated, expected, atol=atol, rtol=rtol)
        self.assertTrue(
            torch.equal(updated.argsort(dim=-1, stable=True), expected.argsort(dim=-1, stable=True))
        )
        self.assertFalse(torch.equal(actual, updated))
        torch.testing.assert_close(actual, retained, atol=0, rtol=0)
        self.assertEqual(actual.device.type, "cpu")
        stats = runner.stats.snapshot()
        self.assertEqual(stats["packed_forwards"], 2)
        self.assertEqual(stats["cuda_timing_sample"]["batches"], 1)
        self.assertEqual(stats["entity_fill"], 1)


if __name__ == "__main__":
    unittest.main()
